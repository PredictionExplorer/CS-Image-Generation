//! V66 `rose-window` -- Gravity Builds a Cathedral.
//!
//! The winding-number stained glass *built in time*: V41's row-bucket
//! crossings are tagged by step, so every video frame holds the exact
//! winding state of the path-so-far (closed dynamically back to the start
//! points). When a region's triple changes -- a true winding increment --
//! the new boundary fractures into existence tip-to-tip over 12 frames
//! under a bright cutting head, and the pane color eases from its parent
//! over 20 frames (`OkLab` lerp). Cames accumulate at 1.5 px, thickened at
//! triple-points. The last eight seconds tilt the finished window into 3D:
//! a parallel light throws its colors onto a dark floor (projective
//! caustic) while the camera cranes from the glass down to the pool of
//! light. One choir voice joins per fracture, capped at 24.

use crate::error::Result;
use crate::oklab::oklab_to_linear_rec2020;
use crate::render::context::RenderContext;
use crate::render::{VideoEncodingOptions, VideoOutputSpec, create_videos_from_frames_singlepass};
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::audio::{
    SAMPLE_RATE, fade_ends, mux, normalize_to_lufs, saw_bandlimited, soft_limit,
    write_wav_stereo_24bit,
};
use crate::viz::common::display::{encode_linear_rec2020_png16, encode_linear_rec2020_to_u16};
use crate::viz::common::raster::Rgb64;
use crate::viz::context::VizContext;
use crate::viz::modes::chord_progression::JI_LATTICE;
use crate::viz::modes::winding_glass::{
    RowCrossing, anchor_hue, pane_oklab, row_crossings, winding_triples,
};
use crate::viz::sink::ArtifactSink;
use nalgebra::Vector3;
use rayon::prelude::*;
use tracing::{info, warn};

/// Film length in seconds (30 fps).
const FILM_SECONDS: f64 = 45.0;
/// The final light-through-glass shot length in seconds.
const FINALE_SECONDS: f64 = 8.0;
/// Frame rate.
const FPS: u32 = 30;
/// Fracture reveal length in frames.
const FRACTURE_REVEAL_FRAMES: f64 = 12.0;
/// Pane color ease length in frames.
const COLOR_EASE_FRAMES: f64 = 20.0;
/// Maximum choir voices.
const MAX_VOICES: usize = 24;
/// Minimum new boundary pixels for a fracture to earn a voice.
const VOICE_PIXEL_THRESHOLD: usize = 8;
/// Minimum spacing between voices in seconds.
const VOICE_SPACING_S: f64 = 0.5;
/// Mask radius around the dynamic closure segments (transient boundaries).
const CLOSURE_MASK_PX: f32 = 2.5;

/// The rose-window mode.
pub struct RoseWindow;

/// Distance from a point to a segment (pixel space).
fn segment_distance(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let ab = (b.0 - a.0, b.1 - a.1);
    let ap = (p.0 - a.0, p.1 - a.1);
    let len_sq = ab.0 * ab.0 + ab.1 * ab.1;
    let t =
        if len_sq > 1e-12 { ((ap.0 * ab.0 + ap.1 * ab.1) / len_sq).clamp(0.0, 1.0) } else { 0.0 };
    let closest = (a.0 + ab.0 * t, a.1 + ab.1 * t);
    (p.0 - closest.0).hypot(p.1 - closest.1)
}

/// One recorded fracture event (the audit log).
struct Fracture {
    frame: usize,
    new_pixels: usize,
    voiced: bool,
    transition: ((i32, i32, i32), (i32, i32, i32)),
}

impl VizMode for RoseWindow {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("rose-window").expect("rose-window is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 1_000 {
            warn!("rose-window skipped: trajectory too short");
            return Ok(());
        }
        let anchor = anchor_hue(ctx);

        // --- Working grid at video resolution.
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16) as usize;
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16) as usize;
        let render_ctx = RenderContext::new(
            video_w as u32,
            video_h as u32,
            ctx.positions,
            ctx.settings.aspect_correction,
        );
        let rows = row_crossings(ctx.positions, &render_ctx, video_h);

        // Static crossings sorted by completion step (the incremental feed);
        // closure crossings (u32::MAX) are handled dynamically instead.
        let mut feed: Vec<(usize, RowCrossing)> = rows
            .iter()
            .enumerate()
            .flat_map(|(row, bucket)| {
                bucket.iter().filter(|event| event.step != u32::MAX).map(move |&event| (row, event))
            })
            .collect();
        feed.sort_by_key(|&(_, event)| event.step);

        let frame_count = ctx.quality.scale_count((FILM_SECONDS * f64::from(FPS)) as usize);
        let finale_frames = ctx
            .quality
            .scale_count((FINALE_SECONDS * f64::from(FPS)) as usize)
            .min(frame_count / 3);
        let build_frames = frame_count - finale_frames;

        // --- Per-pixel state.
        let pixel_count = video_w * video_h;
        let mut winding_static = vec![(0i32, 0i32, 0i32); pixel_count];
        let mut winding_now = vec![(0i32, 0i32, 0i32); pixel_count];
        let mut winding_prev = vec![(0i32, 0i32, 0i32); pixel_count];
        // Pane ease state (OkLab), seeded with the zero-winding paper color.
        let paper = pane_oklab((0, 0, 0), anchor);
        let mut pane_now: Vec<(f32, f32, f32)> =
            vec![(paper.0 as f32, paper.1 as f32, paper.2 as f32); pixel_count];
        // Came activation: frame at which a boundary pixel finishes cutting.
        let mut came_activation: Vec<f32> = vec![f32::INFINITY; pixel_count];
        let mut fractures: Vec<Fracture> = Vec::new();
        let mut voices: Vec<usize> = Vec::new(); // voice start frames
        let mut feed_cursor = 0usize;
        let ease_rate = 1.0f32 / COLOR_EASE_FRAMES as f32;

        // Apply one crossing to a winding image (pixels right of x).
        let apply = |image: &mut [(i32, i32, i32)], row: usize, event: &RowCrossing| {
            let first = (f64::from(event.x) - 0.5).ceil().max(0.0) as usize;
            let delta = i32::from(event.delta);
            let base = row * video_w;
            for triple in &mut image[base + first.min(video_w)..base + video_w] {
                match event.body {
                    0 => triple.0 += delta,
                    1 => triple.1 += delta,
                    _ => triple.2 += delta,
                }
            }
        };

        // --- The build phase, streamed straight into the encoders.
        let outputs = [
            VideoOutputSpec {
                output_file: sink.path("rose_window_silent.mp4"),
                options: VideoEncodingOptions::web_compatible(),
            },
            VideoOutputSpec {
                output_file: sink.path("rose_window_silent_hq.mp4"),
                options: if ctx.fast_encode {
                    VideoEncodingOptions::fast_encode()
                } else {
                    VideoEncodingOptions::high_quality()
                },
            },
        ];
        // The finale needs the finished window; render the build frames
        // first, capturing the final composed window for the 3D shot.
        let mut final_window: Vec<Rgb64> = vec![(0.0, 0.0, 0.0); pixel_count];
        let started = std::time::Instant::now();
        let mut logged = false;
        create_videos_from_frames_singlepass(
            video_w as u32,
            video_h as u32,
            FPS,
            |out| {
                let mut rgb: Vec<Rgb64> = vec![(0.0, 0.0, 0.0); pixel_count];
                let mut bytes: Vec<u16> = Vec::new();
                let mut boundary_prev = vec![false; pixel_count];
                for frame in 0..build_frames {
                    let max_step = ((frame + 1) as f64 / build_frames as f64 * steps as f64) as u32;
                    // Feed new static crossings into the persistent image.
                    while feed_cursor < feed.len() && feed[feed_cursor].1.step <= max_step {
                        let (row, event) = feed[feed_cursor];
                        apply(&mut winding_static, row, &event);
                        feed_cursor += 1;
                    }
                    // Dynamic closure: current positions back to the starts.
                    winding_now.copy_from_slice(&winding_static);
                    let cursor_step = (max_step as usize).min(steps - 1);
                    let closure: [((f32, f32), (f32, f32)); 3] = std::array::from_fn(|body| {
                        let path = &ctx.positions[body];
                        (
                            render_ctx.to_pixel(path[cursor_step].x, path[cursor_step].y),
                            render_ctx.to_pixel(path[0].x, path[0].y),
                        )
                    });
                    let mut closure_rows: Vec<Vec<RowCrossing>> = vec![Vec::new(); video_h];
                    for (body, &(from, to)) in closure.iter().enumerate() {
                        push_closure_crossings(&mut closure_rows, from, to, body as u8);
                    }
                    for (row, bucket) in closure_rows.iter().enumerate() {
                        for event in bucket {
                            apply(&mut winding_now, row, event);
                        }
                    }

                    // Fracture detection: new boundary pixels, closure-masked.
                    let mut new_pixels: Vec<usize> = Vec::new();
                    let mut example = None;
                    for row in 0..video_h {
                        for col in 0..video_w {
                            let index = row * video_w + col;
                            let differs_left =
                                col > 0 && winding_now[index] != winding_now[index - 1];
                            let differs_up =
                                row > 0 && winding_now[index] != winding_now[index - video_w];
                            let boundary = differs_left || differs_up;
                            if boundary && !boundary_prev[index] {
                                let point = (col as f32 + 0.5, row as f32 + 0.5);
                                let masked = closure
                                    .iter()
                                    .any(|&(a, b)| segment_distance(point, a, b) < CLOSURE_MASK_PX);
                                if !masked && came_activation[index].is_infinite() {
                                    if example.is_none() && col > 0 {
                                        example = Some((winding_prev[index], winding_now[index]));
                                    }
                                    new_pixels.push(index);
                                }
                            }
                            boundary_prev[index] = boundary;
                        }
                    }

                    // Tip-to-tip reveal: BFS order from already-cut cames.
                    if !new_pixels.is_empty() {
                        let order =
                            bfs_reveal_order(&new_pixels, &came_activation, video_w, video_h);
                        let scale = FRACTURE_REVEAL_FRAMES
                            / f64::from(order.iter().copied().fold(1.0f32, f32::max));
                        for (&index, &rank) in new_pixels.iter().zip(order.iter()) {
                            came_activation[index] =
                                frame as f32 + (f64::from(rank) * scale) as f32;
                        }
                        // Voice assignment (rate-limited, capped).
                        let spaced = voices.last().is_none_or(|&last| {
                            (frame - last) as f64 / f64::from(FPS) >= VOICE_SPACING_S
                        });
                        let voiced = new_pixels.len() >= VOICE_PIXEL_THRESHOLD
                            && voices.len() < MAX_VOICES
                            && spaced;
                        if voiced {
                            voices.push(frame);
                        }
                        fractures.push(Fracture {
                            frame,
                            new_pixels: new_pixels.len(),
                            voiced,
                            transition: example.unwrap_or(((0, 0, 0), (0, 0, 0))),
                        });
                    }
                    winding_prev.copy_from_slice(&winding_now);

                    // Pane easing toward each pixel's current triple color.
                    pane_now.par_iter_mut().zip(winding_now.par_iter()).for_each(
                        |(pane, &triple)| {
                            let target = pane_oklab(triple, anchor);
                            pane.0 += (target.0 as f32 - pane.0) * ease_rate;
                            pane.1 += (target.1 as f32 - pane.1) * ease_rate;
                            pane.2 += (target.2 as f32 - pane.2) * ease_rate;
                        },
                    );

                    // Compose: panes, cames (activated), cutting heads.
                    let frame_f = frame as f32;
                    rgb.par_iter_mut().enumerate().for_each(|(index, pixel)| {
                        let pane = pane_now[index];
                        let mut color = oklab_to_linear_rec2020(
                            f64::from(pane.0),
                            f64::from(pane.1),
                            f64::from(pane.2),
                        );
                        color.0 = color.0.max(0.0);
                        color.1 = color.1.max(0.0);
                        color.2 = color.2.max(0.0);
                        let activation = came_activation[index];
                        if frame_f >= activation {
                            let age = frame_f - activation;
                            if age < 2.0 {
                                // The cutting head: white-hot for two frames.
                                color = (1.6, 1.5, 1.3);
                            } else {
                                color = (0.006, 0.006, 0.008);
                            }
                        }
                        *pixel = color;
                    });
                    // Triple-point thickening on settled cames.
                    thicken_triple_points(
                        &mut rgb,
                        &winding_now,
                        &came_activation,
                        frame_f,
                        video_w,
                        video_h,
                    );

                    if frame + 1 == build_frames {
                        final_window.copy_from_slice(&rgb);
                    }
                    encode_linear_rec2020_to_u16(&rgb, &mut bytes);
                    out.write_all(bytemuck::cast_slice(&bytes))
                        .map_err(crate::render::error::RenderError::VideoEncoding)?;
                    if frame == 0 && !logged {
                        logged = true;
                        let per_frame = started.elapsed().as_secs_f64();
                        info!(
                            "   rose-window: {per_frame:.2}s first frame, projected {:.1} min",
                            per_frame * frame_count as f64 / 60.0
                        );
                    }
                }

                // --- The finale: light through glass, crane to the floor.
                let window_texture = &final_window;
                let aspect = video_w as f64 / video_h as f64;
                let vfov = 0.9f64; // radians, matches the head-on framing
                let head_on_distance = 1.0 / (vfov * 0.5).tan();
                let light_dir = Vector3::new(0.10, -0.42, 1.0).normalize();
                let floor_y = -1.35f64;
                for frame in 0..finale_frames {
                    let t = frame as f64 / (finale_frames - 1).max(1) as f64;
                    let ease = t * t * (3.0 - 2.0 * t);
                    // Crane: head-on (seamless with the build) down to the pool.
                    let eye = Vector3::new(
                        0.55 * ease,
                        0.10 - 1.05 * ease,
                        head_on_distance - 0.85 * ease,
                    );
                    let look = Vector3::new(0.0, -1.35 * ease, 0.35 * ease);
                    let forward = (look - eye).normalize();
                    let right = forward.cross(&Vector3::new(0.0, 1.0, 0.0)).normalize();
                    let up = right.cross(&forward);
                    let tan_half = (vfov * 0.5).tan();
                    rgb.par_chunks_mut(video_w).enumerate().for_each(|(py, line)| {
                        for (px, pixel) in line.iter_mut().enumerate() {
                            let ndc_x = ((px as f64 + 0.5) / video_w as f64 * 2.0 - 1.0)
                                * tan_half
                                * aspect;
                            let ndc_y = (1.0 - (py as f64 + 0.5) / video_h as f64 * 2.0) * tan_half;
                            let dir = (forward + right * ndc_x + up * ndc_y).normalize();
                            *pixel = trace_finale(
                                eye,
                                dir,
                                window_texture,
                                video_w,
                                video_h,
                                aspect,
                                light_dir,
                                floor_y,
                            );
                        }
                    });
                    encode_linear_rec2020_to_u16(&rgb, &mut bytes);
                    out.write_all(bytemuck::cast_slice(&bytes))
                        .map_err(crate::render::error::RenderError::VideoEncoding)?;
                }
                Ok(())
            },
            &outputs,
        )?;
        info!(
            "   rose-window: {} fractures ({} voiced) over {build_frames} build frames",
            fractures.len(),
            voices.len()
        );

        // --- Audio: drone + one choir voice per fracture.
        let duration = frame_count as f64 / f64::from(FPS);
        let sample_rate = f64::from(SAMPLE_RATE);
        let total_samples = (duration * sample_rate) as usize;
        let mut left = vec![0.0f64; total_samples];
        let mut right = vec![0.0f64; total_samples];
        let root = 82.4; // E2: cathedral register
        let mut drone_phase = (0.0f64, 0.0f64);
        let voice_data: Vec<(f64, f64, f64)> = voices
            .iter()
            .enumerate()
            .map(|(index, &frame)| {
                let (num, den) = JI_LATTICE[(index * 5) % JI_LATTICE.len()];
                let octave = 1.0 + f64::from((index % 3) as u32);
                let frequency = root * f64::from(num) / f64::from(den) * octave;
                let start = frame as f64 / f64::from(FPS);
                let pan = ((index as f64 * 0.618).fract() - 0.5) * 1.2;
                (frequency, start, pan.clamp(-0.8, 0.8))
            })
            .collect();
        let mut voice_phases = vec![(0.0f64, 0.0f64); voice_data.len()];
        for sample in 0..total_samples {
            let t = sample as f64 / sample_rate;
            // Drone: root + fifth, quiet and stable.
            drone_phase.0 = (drone_phase.0 + root / sample_rate) % 1.0;
            drone_phase.1 = (drone_phase.1 + root * 1.5 / sample_rate) % 1.0;
            let drone = (saw_bandlimited(drone_phase.0, root / sample_rate) * 0.5
                + saw_bandlimited(drone_phase.1, root * 1.5 / sample_rate) * 0.3)
                * 0.10;
            let mut frame = (drone, drone);
            for ((frequency, start, pan), phase) in voice_data.iter().zip(voice_phases.iter_mut()) {
                if t < *start {
                    continue;
                }
                let held = t - start;
                // Choir: detuned saw pair, slow 1.2 s attack, gentle sustain.
                let attack = (held / 1.2).clamp(0.0, 1.0);
                let envelope = attack * attack * (0.20 + 0.05 * (held * 0.4).sin());
                phase.0 = (phase.0 + frequency / sample_rate) % 1.0;
                phase.1 = (phase.1 + frequency * 1.004 / sample_rate) % 1.0;
                let tone = saw_bandlimited(phase.0, frequency / sample_rate)
                    + saw_bandlimited(phase.1, frequency * 1.004 / sample_rate);
                let (l, r) =
                    crate::viz::common::audio::equal_power_pan(tone * envelope * 0.045, *pan);
                frame.0 += l;
                frame.1 += r;
            }
            left[sample] = frame.0;
            right[sample] = frame.1;
        }
        soft_limit(&mut left, &mut right, 1.2);
        normalize_to_lufs(&mut left, &mut right, -17.0, sample_rate);
        fade_ends(&mut left, (sample_rate * 0.4) as usize);
        fade_ends(&mut right, (sample_rate * 0.4) as usize);
        let mix_path = sink.path("rose_mix.wav");
        write_wav_stereo_24bit(&mix_path, &left, &right)?;
        for (silent, scored) in [
            ("rose_window_silent.mp4", "rose_window.mp4"),
            ("rose_window_silent_hq.mp4", "rose_window_hq.mp4"),
        ] {
            let silent_path = sink.path(silent);
            match mux(&silent_path, &mix_path, &sink.path(scored)) {
                Ok(()) => {
                    sink.record(scored, "video");
                    let _ = std::fs::remove_file(&silent_path);
                }
                Err(error) => {
                    warn!("rose-window mux failed for {scored}: {error}; keeping silent cut");
                    let _ = std::fs::rename(&silent_path, sink.path(scored));
                    sink.record(scored, "video");
                }
            }
        }
        let _ = std::fs::remove_file(&mix_path);

        // --- Full-resolution final window (consistency with V41's core).
        {
            let full_w = ctx.quality.scale_dim(ctx.width) as usize;
            let full_h = ctx.quality.scale_dim(ctx.height) as usize;
            let full_ctx = RenderContext::new(
                full_w as u32,
                full_h as u32,
                ctx.positions,
                ctx.settings.aspect_correction,
            );
            let full_rows = row_crossings(ctx.positions, &full_ctx, full_h);
            let triples = winding_triples(&full_rows, full_w, u32::MAX, None);
            let mut pixels: Vec<Rgb64> = triples
                .par_iter()
                .map(|&triple| {
                    let (l, a, b) = pane_oklab(triple, anchor);
                    let color = oklab_to_linear_rec2020(l, a, b);
                    (color.0.max(0.0), color.1.max(0.0), color.2.max(0.0))
                })
                .collect();
            let lead: Rgb64 = (0.006, 0.006, 0.008);
            for row in 0..full_h {
                for col in 0..full_w {
                    let index = row * full_w + col;
                    let differs_left = col > 0 && triples[index] != triples[index - 1];
                    let differs_up = row > 0 && triples[index] != triples[index - full_w];
                    if differs_left || differs_up {
                        pixels[index] = lead;
                        if differs_left {
                            pixels[index - 1] = lead;
                        }
                        if differs_up {
                            pixels[index - full_w] = lead;
                        }
                    }
                }
            }
            let image = encode_linear_rec2020_png16(&pixels, full_w as u32, full_h as u32);
            sink.save_png16(&image, "rose_window.png")?;
        }

        // --- The audit log.
        let events: Vec<serde_json::Value> = fractures
            .iter()
            .map(|fracture| {
                serde_json::json!({
                    "frame": fracture.frame,
                    "new_boundary_pixels": fracture.new_pixels,
                    "voiced": fracture.voiced,
                    "example_transition": [
                        [fracture.transition.0.0, fracture.transition.0.1, fracture.transition.0.2],
                        [fracture.transition.1.0, fracture.transition.1.1, fracture.transition.1.2],
                    ],
                })
            })
            .collect();
        let meta = serde_json::json!({
            "build_frames": build_frames,
            "finale_frames": finale_frames,
            "fracture_reveal_frames": FRACTURE_REVEAL_FRAMES,
            "color_ease_frames": COLOR_EASE_FRAMES,
            "voices": voices.len(),
            "fractures": events,
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("fractures.json", &json, "data")?;
        Ok(())
    }
}

/// Scanline crossings of one closure segment (same convention as the
/// bucket builder, without step tags).
fn push_closure_crossings(
    rows: &mut [Vec<RowCrossing>],
    from: (f32, f32),
    to: (f32, f32),
    body: u8,
) {
    let height = rows.len();
    let (x0, y0) = from;
    let (x1, y1) = to;
    if (y1 - y0).abs() < f32::EPSILON {
        return;
    }
    let delta: i8 = if y1 > y0 { 1 } else { -1 };
    let (top, bottom) = if y0 < y1 { (y0, y1) } else { (y1, y0) };
    let row_start = (f64::from(top) - 0.5).ceil().max(0.0) as usize;
    let row_end_f = (f64::from(bottom) - 0.5).floor().min((height - 1) as f64);
    if row_end_f < 0.0 {
        return;
    }
    for (row, bucket) in rows.iter_mut().enumerate().take(row_end_f as usize + 1).skip(row_start) {
        let scan_y = row as f32 + 0.5;
        let t = (scan_y - y0) / (y1 - y0);
        bucket.push(RowCrossing { x: x0 + t * (x1 - x0), body, delta, step: 0 });
    }
}

/// BFS reveal ranks over the new boundary pixels: growth starts adjacent to
/// already-activated cames (or at the first pixel of each disconnected
/// piece) and spreads tip-to-tip.
fn bfs_reveal_order(
    new_pixels: &[usize],
    came_activation: &[f32],
    width: usize,
    height: usize,
) -> Vec<f32> {
    use std::collections::VecDeque;
    let position_of: std::collections::HashMap<usize, usize> =
        new_pixels.iter().enumerate().map(|(rank, &index)| (index, rank)).collect();
    let mut order = vec![f32::NEG_INFINITY; new_pixels.len()];
    let mut queue: VecDeque<(usize, f32)> = VecDeque::new();
    // Seeds: new pixels touching an activated came.
    for (slot, &index) in new_pixels.iter().enumerate() {
        let col = index % width;
        let row = index / width;
        let mut seeded = false;
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                let nc = col as i64 + dx;
                let nr = row as i64 + dy;
                if nc < 0 || nr < 0 || nc >= width as i64 || nr >= height as i64 {
                    continue;
                }
                if came_activation[nr as usize * width + nc as usize].is_finite() {
                    seeded = true;
                }
            }
        }
        if seeded {
            order[slot] = 0.0;
            queue.push_back((index, 0.0));
        }
    }
    // Flood through the new set.
    let flood = |queue: &mut VecDeque<(usize, f32)>, order: &mut Vec<f32>| {
        while let Some((index, rank)) = queue.pop_front() {
            let col = index % width;
            let row = index / width;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let nc = col as i64 + dx;
                    let nr = row as i64 + dy;
                    if nc < 0 || nr < 0 || nc >= width as i64 || nr >= height as i64 {
                        continue;
                    }
                    let neighbor = nr as usize * width + nc as usize;
                    if let Some(&slot) = position_of.get(&neighbor)
                        && order[slot] == f32::NEG_INFINITY
                    {
                        order[slot] = rank + 1.0;
                        queue.push_back((neighbor, rank + 1.0));
                    }
                }
            }
        }
    };
    flood(&mut queue, &mut order);
    // Disconnected pieces start at their first pixel in scan order.
    for slot in 0..new_pixels.len() {
        if order[slot] == f32::NEG_INFINITY {
            order[slot] = 0.0;
            queue.push_back((new_pixels[slot], 0.0));
            flood(&mut queue, &mut order);
        }
    }
    order
}

/// Thicken settled cames at triple-points (three distinct triples in the
/// 3x3 neighborhood): real leaded-glass joints.
fn thicken_triple_points(
    rgb: &mut [Rgb64],
    winding: &[(i32, i32, i32)],
    came_activation: &[f32],
    frame: f32,
    width: usize,
    height: usize,
) {
    let lead: Rgb64 = (0.006, 0.006, 0.008);
    let mut joints: Vec<usize> = Vec::new();
    for row in 1..height - 1 {
        for col in 1..width - 1 {
            let index = row * width + col;
            if frame < came_activation[index] + 2.0 {
                continue;
            }
            let mut seen = [winding[index], winding[index], winding[index]];
            let mut distinct = 1usize;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let neighbor =
                        ((row as i64 + dy) as usize) * width + (col as i64 + dx) as usize;
                    let triple = winding[neighbor];
                    if !seen[..distinct].contains(&triple) {
                        if distinct < 3 {
                            seen[distinct] = triple;
                        }
                        distinct += 1;
                    }
                }
            }
            if distinct >= 3 {
                joints.push(index);
            }
        }
    }
    for index in joints {
        let col = index % width;
        let row = index / width;
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                let neighbor = ((row as i64 + dy).clamp(0, height as i64 - 1) as usize) * width
                    + (col as i64 + dx).clamp(0, width as i64 - 1) as usize;
                rgb[neighbor] = lead;
            }
        }
    }
}

/// Trace one finale ray: the emissive window pane, the projective caustic
/// pool on the floor, black elsewhere.
#[allow(clippy::too_many_arguments)]
fn trace_finale(
    eye: Vector3<f64>,
    dir: Vector3<f64>,
    window: &[Rgb64],
    tex_w: usize,
    tex_h: usize,
    aspect: f64,
    light_dir: Vector3<f64>,
    floor_y: f64,
) -> Rgb64 {
    let sample_window = |x: f64, y: f64| -> Option<Rgb64> {
        // Window rect: x in [-aspect, aspect], y in [-1, 1] at z = 0.
        if x.abs() > aspect || y.abs() > 1.0 {
            return None;
        }
        let u = (x / aspect * 0.5 + 0.5).clamp(0.0, 1.0);
        let v = (0.5 - y * 0.5).clamp(0.0, 1.0);
        let px = ((u * (tex_w - 1) as f64) as usize).min(tex_w - 1);
        let py = ((v * (tex_h - 1) as f64) as usize).min(tex_h - 1);
        Some(window[py * tex_w + px])
    };

    // Window plane z = 0 (only from the front).
    if dir.z.abs() > 1e-9 {
        let t = -eye.z / dir.z;
        if t > 1e-6 {
            let hit = eye + dir * t;
            if let Some(texel) = sample_window(hit.x, hit.y) {
                // Emissive glass, brighter where the light grazes through.
                let glow = 1.0 + 0.6 * light_dir.dot(&-dir).max(0.0);
                return (texel.0 * glow, texel.1 * glow, texel.2 * glow);
            }
        }
    }
    // Floor plane y = floor_y: the projective caustic.
    if dir.y < -1e-9 {
        let t = (floor_y - eye.y) / dir.y;
        if t > 1e-6 {
            let hit = eye + dir * t;
            // March back along the light to the window plane.
            if light_dir.z.abs() > 1e-9 {
                let back = -hit.z / light_dir.z;
                let source = hit + light_dir * back;
                if back < 0.0
                    && let Some(texel) = sample_window(source.x, source.y)
                {
                    // Cone falloff with distance from the glass.
                    let fade = (1.0 / (1.0 + 0.25 * back.abs())).powi(2);
                    let ambient = 0.0025;
                    return (
                        texel.0 * fade * 0.85 + ambient,
                        texel.1 * fade * 0.85 + ambient,
                        texel.2 * fade * 0.85 + ambient,
                    );
                }
            }
            return (0.0022, 0.0022, 0.003);
        }
    }
    (0.0008, 0.0008, 0.0012)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reveal_order_grows_from_existing_cames() {
        // A horizontal run of new pixels next to one activated came pixel
        // on the left: ranks must increase left to right.
        let width = 16usize;
        let height = 4usize;
        let mut activation = vec![f32::INFINITY; width * height];
        activation[width + 1] = 0.0; // activated came at (1, 1)
        let new_pixels: Vec<usize> = (2..10).map(|col| width + col).collect();
        let order = bfs_reveal_order(&new_pixels, &activation, width, height);
        for pair in order.windows(2) {
            assert!(pair[1] >= pair[0], "reveal must grow tip-to-tip: {order:?}");
        }
        assert_eq!(order[0], 0.0, "growth starts at the seed");
        assert!(order[order.len() - 1] >= (order.len() - 1) as f32 - 1.0);
    }

    #[test]
    fn closure_crossings_match_the_shared_convention() {
        let mut rows: Vec<Vec<RowCrossing>> = vec![Vec::new(); 8];
        // A vertical segment crossing scanlines 2..6 at x = 3.5.
        push_closure_crossings(&mut rows, (3.5, 1.0), (3.5, 6.9), 1);
        for (row, bucket) in rows.iter().enumerate() {
            if (1..=6).contains(&row) {
                assert_eq!(bucket.len(), 1, "row {row} should hold one crossing");
                assert!((bucket[0].x - 3.5).abs() < 1e-6);
                assert_eq!(bucket[0].delta, 1);
            } else {
                assert!(bucket.is_empty(), "row {row} should be empty");
            }
        }
    }

    #[test]
    fn finale_floor_receives_the_window_colors() {
        // A uniform red window: a floor ray under the glass must read red
        // through the projective caustic.
        let tex_w = 8usize;
        let tex_h = 8usize;
        let window = vec![(0.8, 0.05, 0.05); tex_w * tex_h];
        let light = Vector3::new(0.0, -0.4, 1.0).normalize();
        let eye = Vector3::new(0.0, -0.2, 2.0);
        // Aim at a floor point in front of the window (z > 0 side).
        let target = Vector3::new(0.0, -1.35, 1.0);
        let dir = (target - eye).normalize();
        let color = trace_finale(eye, dir, &window, tex_w, tex_h, 1.0, light, -1.35);
        assert!(color.0 > color.1 * 5.0, "the pool must carry the glass color, got {color:?}");
        // A ray to the void stays black.
        let void = trace_finale(
            eye,
            Vector3::new(0.9, 0.4, -0.2).normalize(),
            &window,
            tex_w,
            tex_h,
            1.0,
            light,
            -1.35,
        );
        assert!(void.0 < 0.01);
    }

    #[test]
    fn segment_distance_is_zero_on_and_positive_off_the_segment() {
        let a = (0.0f32, 0.0f32);
        let b = (10.0f32, 0.0f32);
        assert!(segment_distance((5.0, 0.0), a, b) < 1e-6);
        assert!((segment_distance((5.0, 3.0), a, b) - 3.0).abs() < 1e-6);
        assert!((segment_distance((-4.0, 0.0), a, b) - 4.0).abs() < 1e-6);
    }
}
