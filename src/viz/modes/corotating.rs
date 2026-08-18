//! V26 `corotating` -- The Same Dance from the Dance Floor.
//!
//! Transforms the run into the co-rotating frame of the tightest pair, where
//! chaos untangles into horseshoes and temporary moons. Ships a diptych
//! (master beside the co-rotating exposure), a solo still, and a reveal
//! video that morphs mid-film: the inertial image crossfades energy-linearly
//! into a co-rotating exposure that re-grows time-compressed 6x, with L4/L5
//! sigils appearing in the rotated frame.

use crate::error::Result;
use crate::oklab::{oklab_to_oklch, oklch_to_oklab};
use crate::render::context::PixelBuffer;
use crate::render::{ImageBuffer, Rgb, SpectralRenderSettings, SpectralScene};
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::{Accumulator, resized_config, scene_levels, stream_video};
use crate::viz::common::display::{SpdCanvas, grade_auto_levels};
use crate::viz::common::kinematics::PAIRS;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use nalgebra::Vector3;
use tracing::{info, warn};

/// Video frames at 60 fps (30 s reveal).
const VIDEO_FRAMES: usize = 1800;
/// Inertial segment fraction of the reveal (10 s of 30 s).
const INERTIAL_FRACTION: f64 = 10.0 / 30.0;
/// Morph window fraction of the reveal (3 s of 30 s).
const MORPH_FRACTION: f64 = 3.0 / 30.0;
/// Regrow time compression during and after the morph.
const REGROW_COMPRESSION: usize = 6;
/// Diptych gutter as a fraction of panel width.
const GUTTER_FRACTION: f64 = 1.0 / 48.0;
/// Sigil stroke energy (quiet presence, not a UI marker).
const SIGIL_ENERGY: f64 = 0.02;

/// The co-rotating frame mode.
pub struct Corotating;

/// The tightest pair: minimum time-averaged separation (canonical order).
#[must_use]
pub fn tightest_pair(ctx: &VizContext<'_>) -> (usize, usize) {
    let kinematics = ctx.kinematics();
    PAIRS
        .iter()
        .enumerate()
        .min_by(|(index_a, _), (index_b, _)| {
            let mean = |index: usize| {
                let series = &kinematics.pairwise[index];
                let count = series.iter().step_by(97).count().max(1);
                series.iter().step_by(97).sum::<f64>() / count as f64
            };
            mean(*index_a).total_cmp(&mean(*index_b))
        })
        .map_or((0, 1), |(_, &pair)| pair)
}

/// Co-rotating transform of the whole trajectory for a pair: per step,
/// rotate all bodies by `-theta(t)` about the pair's mass-weighted
/// barycenter and translate that barycenter to the origin. The pair ends up
/// on the x axis (primary on the negative side); z is preserved.
///
/// Exported for V37 `roche` (and the later V49/V65 combos).
#[must_use]
pub fn corotating_positions(
    positions: &[Vec<Vector3<f64>>],
    masses: [f64; 3],
    pair: (usize, usize),
) -> Vec<Vec<Vector3<f64>>> {
    let steps = positions.first().map_or(0, Vec::len);
    let (a, b) = pair;
    let total = (masses[a] + masses[b]).max(1e-12);

    // Per-step frame: pair barycenter and the rotation undoing theta(t).
    let frames: Vec<(Vector3<f64>, f64, f64)> = (0..steps)
        .map(|step| {
            let barycenter =
                (positions[a][step] * masses[a] + positions[b][step] * masses[b]) / total;
            let delta = positions[b][step] - positions[a][step];
            let theta = delta.y.atan2(delta.x);
            let (sin, cos) = (-theta).sin_cos();
            (barycenter, sin, cos)
        })
        .collect();

    positions
        .iter()
        .map(|body| {
            body.iter()
                .zip(&frames)
                .map(|(position, &(barycenter, sin, cos))| {
                    let relative = position - barycenter;
                    Vector3::new(
                        relative.x * cos - relative.y * sin,
                        relative.x * sin + relative.y * cos,
                        relative.z,
                    )
                })
                .collect()
        })
        .collect()
}

/// Smoothstep ramp on [0, 1].
fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// L4/L5 positions in the co-rotating frame for the mean pair separation.
fn lagrange_points(
    masses: [f64; 3],
    pair: (usize, usize),
    mean_separation: f64,
) -> [(f64, f64); 2] {
    let (a, b) = pair;
    let total = (masses[a] + masses[b]).max(1e-12);
    let primary_x = -mean_separation * masses[b] / total;
    let apex_x = primary_x + mean_separation * 0.5;
    let apex_y = mean_separation * 3.0f64.sqrt() * 0.5;
    [(apex_x, apex_y), (apex_x, -apex_y)]
}

/// Draw a small triangular sigil (energy strokes) at a pixel position.
fn draw_sigil(canvas: &mut SpdCanvas, center: (f32, f32), radius: f32, color: (f64, f64, f64)) {
    let mut corners = [(0.0f32, 0.0f32); 3];
    for (index, corner) in corners.iter_mut().enumerate() {
        let angle = -std::f64::consts::FRAC_PI_2 + index as f64 * std::f64::consts::TAU / 3.0;
        *corner = (center.0 + radius * angle.cos() as f32, center.1 + radius * angle.sin() as f32);
    }
    for index in 0..3 {
        canvas.draw_stroke(
            corners[index],
            corners[(index + 1) % 3],
            color,
            color,
            0.9,
            SIGIL_ENERGY,
            0.8,
        );
    }
}

/// Box-downscale a 16-bit image by 2x.
fn downscale_half(image: &ImageBuffer<Rgb<u16>, Vec<u16>>) -> ImageBuffer<Rgb<u16>, Vec<u16>> {
    let half_w = (image.width() / 2).max(1);
    let half_h = (image.height() / 2).max(1);
    let mut out: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::new(half_w, half_h);
    for (x, y, pixel) in out.enumerate_pixels_mut() {
        let mut sum = [0u32; 3];
        for dy in 0..2u32 {
            for dx in 0..2u32 {
                let source = image.get_pixel(
                    (x * 2 + dx).min(image.width() - 1),
                    (y * 2 + dy).min(image.height() - 1),
                );
                for (slot, &value) in sum.iter_mut().zip(source.0.iter()) {
                    *slot += u32::from(value);
                }
            }
        }
        *pixel = Rgb([(sum[0] / 4) as u16, (sum[1] / 4) as u16, (sum[2] / 4) as u16]);
    }
    out
}

impl VizMode for Corotating {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("corotating").expect("corotating is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps == 0 {
            warn!("corotating skipped: empty trajectory");
            return Ok(());
        }
        let clip_black = ctx.settings.resolved_config.clip_black;
        let clip_white = ctx.settings.resolved_config.clip_white;
        let masses = ctx.kinematics().masses;
        let pair = tightest_pair(ctx);
        let pair_index = PAIRS
            .iter()
            .position(|&candidate| candidate == pair)
            .expect("tightest pair is canonical");
        let mean_separation = {
            let series = &ctx.kinematics().pairwise[pair_index];
            let count = series.iter().step_by(97).count().max(1);
            series.iter().step_by(97).sum::<f64>() / count as f64
        };
        info!("   corotating: pair ({}, {}), mean separation {mean_separation:.3}", pair.0, pair.1);

        let corot = corotating_positions(ctx.positions, masses, pair);
        let lagrange = lagrange_points(masses, pair, mean_separation);
        let sigil_color = {
            let (l, a, b) = ctx.mean_color(pair.0);
            let (_, chroma, hue) = oklab_to_oklch(l, a, b);
            oklch_to_oklab(0.8, (chroma * 0.5).min(0.06), hue)
        };

        // --- Solo still: full accumulation in the co-rotating frame.
        let still_w = ctx.quality.scale_dim(ctx.width);
        let still_h = ctx.quality.scale_dim(ctx.height);
        let corot_image = {
            let mut accumulator = Accumulator::new(
                corot.clone(),
                ctx.colors.to_vec(),
                ctx.body_alphas.to_vec(),
                still_w,
                still_h,
                ctx.settings.aspect_correction,
                ctx.settings.traits,
                ctx.settings.render_config.hdr_scale,
            );
            accumulator.accumulate(0..steps);

            // L4/L5 sigils as quiet energy strokes.
            let mut sigils = SpdCanvas::new(still_w, still_h);
            let radius = (f64::from(still_w.min(still_h)) * 0.006).max(4.0) as f32;
            for &(x, y) in &lagrange {
                let px = accumulator.render_ctx().to_pixel(x, y);
                draw_sigil(&mut sigils, px, radius, sigil_color);
            }
            let mut rgba = accumulator.convert();
            let mut sigil_rgba: PixelBuffer = Vec::new();
            sigils.convert_into(&mut sigil_rgba);
            for (pixel, extra) in rgba.iter_mut().zip(sigil_rgba.iter()) {
                pixel.0 += extra.0;
                pixel.1 += extra.1;
                pixel.2 += extra.2;
                pixel.3 = (pixel.3 + extra.3).min(1.0);
            }
            grade_auto_levels(&rgba, still_w, still_h, clip_black, clip_white, 1.0)
        };
        sink.save_png16(&corot_image, "corotating.png")?;

        // --- Diptych: the finished master beside the co-rotating exposure.
        let master_path = format!("{}/images/source/master.png", ctx.seed_dir);
        match image::ImageReader::open(&master_path).map(image::ImageReader::decode) {
            Ok(Ok(decoded)) => {
                let master_half = downscale_half(&decoded.into_rgb16());
                let corot_half = downscale_half(&corot_image);
                let panel_w = master_half.width().min(corot_half.width());
                let panel_h = master_half.height().min(corot_half.height());
                let gutter = (f64::from(panel_w) * GUTTER_FRACTION) as u32;
                let total_w = panel_w * 2 + gutter * 3;
                let total_h = panel_h + gutter * 2;
                let mut diptych: ImageBuffer<Rgb<u16>, Vec<u16>> =
                    ImageBuffer::new(total_w, total_h);
                for (index, panel) in [&master_half, &corot_half].into_iter().enumerate() {
                    let origin_x = gutter + index as u32 * (panel_w + gutter);
                    for y in 0..panel_h {
                        for x in 0..panel_w {
                            diptych.put_pixel(origin_x + x, gutter + y, *panel.get_pixel(x, y));
                        }
                    }
                }
                sink.save_png16(&diptych, "diptych.png")?;
            }
            Ok(Err(error)) => warn!("corotating: master decode failed ({error}); no diptych"),
            Err(error) => warn!("corotating: master missing ({error}); no diptych"),
        }

        // --- Reveal video: inertial -> 3 s morph (energy-linear crossfade
        // while the co-rotating exposure regrows 6x compressed) -> co-rotating.
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16);
        let frame_count = ctx.quality.scale_count(VIDEO_FRAMES);
        let inertial_frames = ((frame_count as f64 * INERTIAL_FRACTION) as usize).max(1);
        let morph_frames = ((frame_count as f64 * MORPH_FRACTION) as usize).max(1);

        let mut inertial_acc = Accumulator::new(
            ctx.positions.to_vec(),
            ctx.colors.to_vec(),
            ctx.body_alphas.to_vec(),
            video_w,
            video_h,
            ctx.settings.aspect_correction,
            ctx.settings.traits,
            ctx.settings.render_config.hdr_scale,
        );
        let mut corot_acc = Accumulator::new(
            corot.clone(),
            ctx.colors.to_vec(),
            ctx.body_alphas.to_vec(),
            video_w,
            video_h,
            ctx.settings.aspect_correction,
            ctx.settings.traits,
            ctx.settings.render_config.hdr_scale,
        );

        // One fixed grade for the whole reveal, from the co-rotating scene.
        let resized = resized_config(ctx.settings.resolved_config, video_w, video_h);
        let render_config = *ctx.settings.render_config;
        let video_settings =
            SpectralRenderSettings::new(&resized, &render_config, ctx.settings.aspect_correction)
                .with_traits(ctx.settings.traits);
        let corot_scene = SpectralScene::new(&corot, ctx.colors, ctx.body_alphas);
        let levels = scene_levels(corot_scene, video_settings, 1.0);

        // Sigil overlay for the co-rotating half of the film.
        let mut sigil_rgba: PixelBuffer = Vec::new();
        {
            let mut sigils = SpdCanvas::new(video_w, video_h);
            let radius = (f64::from(video_w.min(video_h)) * 0.006).max(3.0) as f32;
            for &(x, y) in &lagrange {
                let px = corot_acc.render_ctx().to_pixel(x, y);
                draw_sigil(&mut sigils, px, radius, sigil_color);
            }
            sigils.convert_into(&mut sigil_rgba);
        }

        let per_frame_budget = (steps / frame_count).max(1);
        let mut inertial_cursor = 0usize;
        let mut corot_cursor = 0usize;
        let mut inertial_frozen: PixelBuffer = Vec::new();
        stream_video(
            video_w,
            video_h,
            60,
            &sink.path("reveal.mp4"),
            &sink.path("reveal_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            &levels,
            |frame, rgba| {
                let cursor = ((frame + 1) * steps / frame_count).min(steps);
                if frame < inertial_frames {
                    // Segment A: the inertial exposure grows normally.
                    inertial_acc.accumulate(inertial_cursor..cursor);
                    inertial_cursor = cursor;
                    inertial_acc.convert_into(rgba);
                } else {
                    // Segment B regrows from step 0, time-compressed.
                    let target = (corot_cursor + per_frame_budget * REGROW_COMPRESSION).min(cursor);
                    corot_acc.accumulate(corot_cursor..target);
                    corot_cursor = target;

                    let morph_t = (frame - inertial_frames) as f64 / morph_frames as f64;
                    let weight = smoothstep(morph_t);
                    corot_acc.convert_into(rgba);
                    if weight < 1.0 {
                        if inertial_frozen.is_empty() {
                            inertial_acc.convert_into(&mut inertial_frozen);
                        }
                        for (pixel, frozen) in rgba.iter_mut().zip(inertial_frozen.iter()) {
                            pixel.0 = pixel.0 * weight + frozen.0 * (1.0 - weight);
                            pixel.1 = pixel.1 * weight + frozen.1 * (1.0 - weight);
                            pixel.2 = pixel.2 * weight + frozen.2 * (1.0 - weight);
                            pixel.3 = pixel.3.max(frozen.3 * (1.0 - weight));
                        }
                    }
                    // L4/L5 sigils fade in with the rotated frame.
                    for (pixel, extra) in rgba.iter_mut().zip(sigil_rgba.iter()) {
                        pixel.0 += extra.0 * weight;
                        pixel.1 += extra.1 * weight;
                        pixel.2 += extra.2 * weight;
                        pixel.3 = (pixel.3 + extra.3 * weight).min(1.0);
                    }
                }
            },
        )?;
        sink.record("reveal.mp4", "video");
        sink.record("reveal_hq.mp4", "video");

        let meta = serde_json::json!({
            "pair": [pair.0, pair.1],
            "mean_separation": mean_separation,
            "lagrange_points_frame": lagrange,
            "inertial_frames": inertial_frames,
            "morph_frames": morph_frames,
            "regrow_compression": REGROW_COMPRESSION,
            "render_size": [video_w, video_h],
            "note": "morph = energy-linear SPD crossfade into a 6x regrown co-rotating exposure",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("corotating.json", &json, "data")?;
        Ok(())
    }
}
