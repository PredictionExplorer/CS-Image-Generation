//! V44 `neon` -- Signage from the End of the Universe.
//!
//! The trajectory fabricated as physical neon: curvature-clamped glass
//! strands (real neon cannot fold tighter than ~2 tube diameters), at most
//! three strands per body with electrode caps and mounting standoffs, hung
//! in front of a seeded procedural brick wall that catches the tube wash.
//! Still at full res with shadowed wall light; 8 s seamless flicker loop.

use crate::error::Result;
use crate::oklab::{oklab_to_linear_rec2020, oklab_to_oklch, oklch_to_oklab};
use crate::render::context::PixelBuffer;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::stream_video;
use crate::viz::common::display::{auto_levels, grade_auto_levels};
use crate::viz::common::tube_render::{
    Capsule, Plane, PlaneFinish, RenderParams, Scene, Vec3, fit_distance, orbit_camera,
    rdp_simplify_3d, render,
};
use crate::viz::context::VizContext;
use crate::viz::modes::worldtube::fill_rgba;
use crate::viz::sink::ArtifactSink;
use tracing::{info, warn};

/// Tube radius as a fraction of the sign extent.
const TUBE_RADIUS_FRACTION: f64 = 0.009;
/// Minimum bend radius in tube diameters (glasswork limit).
const MIN_BEND_DIAMETERS: f64 = 2.0;
/// Maximum strands per body.
const MAX_STRANDS: usize = 3;
/// Total glass length trimmed away from strand ends.
const TRIM_FRACTION: f64 = 0.20;
/// Loop length in seconds at 30 fps.
const LOOP_SECONDS: usize = 8;
/// Wall distance behind the tubes as a fraction of the extent.
const WALL_GAP_FRACTION: f64 = 0.4;
/// Emission gain for the gas glow.
const EMISSION_GAIN: f64 = 5.0;

/// Uniformly resample a polyline to a fixed spacing.
fn resample(points: &[Vec3], spacing: f64) -> Vec<Vec3> {
    if points.len() < 2 || spacing <= 0.0 {
        return points.to_vec();
    }
    let mut result = vec![points[0]];
    let mut carry = 0.0f64;
    for pair in points.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let length = (b - a).norm();
        if length <= 1e-12 {
            continue;
        }
        let mut travelled = carry;
        while travelled + spacing <= length {
            travelled += spacing;
            result.push(a + (b - a) * (travelled / length));
        }
        carry = travelled - length;
    }
    if result.len() < 2 {
        result.push(*points.last().expect("nonempty"));
    }
    result
}

/// Clamp curvature by Laplacian smoothing of joints sharper than the
/// minimum bend radius allows at the given sample spacing.
fn clamp_curvature(points: &mut [Vec3], spacing: f64, bend_radius: f64, passes: usize) {
    if points.len() < 3 {
        return;
    }
    let max_turn = (spacing / bend_radius.max(1e-9)).min(1.2);
    for _ in 0..passes {
        let mut adjusted = 0usize;
        for index in 1..points.len() - 1 {
            let previous = points[index - 1];
            let next = points[index + 1];
            let incoming = (points[index] - previous).normalize();
            let outgoing = (next - points[index]).normalize();
            let turn = incoming.dot(&outgoing).clamp(-1.0, 1.0).acos();
            if turn > max_turn {
                points[index] = points[index] * 0.4 + (previous + next) * 0.3;
                adjusted += 1;
            }
        }
        if adjusted == 0 {
            break;
        }
    }
}

/// Split a polyline into at most `MAX_STRANDS` strands at its sharpest
/// remaining joints, then trim the fastest-looking ends to the glass cap.
fn split_strands(points: &[Vec3]) -> Vec<Vec<Vec3>> {
    if points.len() < 8 {
        return vec![points.to_vec()];
    }
    let mut turns: Vec<(f64, usize)> = (1..points.len() - 1)
        .map(|index| {
            let incoming = (points[index] - points[index - 1]).normalize();
            let outgoing = (points[index + 1] - points[index]).normalize();
            (incoming.dot(&outgoing).clamp(-1.0, 1.0).acos(), index)
        })
        .collect();
    turns.sort_by(|lhs, rhs| rhs.0.total_cmp(&lhs.0).then(lhs.1.cmp(&rhs.1)));

    let min_gap = points.len() / 5;
    let mut cuts: Vec<usize> = Vec::new();
    for &(turn, index) in &turns {
        if cuts.len() + 1 >= MAX_STRANDS || turn < 0.35 {
            break;
        }
        if cuts.iter().all(|&cut| cut.abs_diff(index) > min_gap)
            && index > min_gap
            && points.len() - index > min_gap
        {
            cuts.push(index);
        }
    }
    cuts.sort_unstable();

    let mut strands = Vec::new();
    let mut start = 0usize;
    for &cut in &cuts {
        strands.push(points[start..=cut].to_vec());
        start = cut;
    }
    strands.push(points[start..].to_vec());

    // Trim ~TRIM_FRACTION of the total length from strand ends (the glass
    // budget cap; ends are the dimmest, fastest passages after clamping).
    let per_end = TRIM_FRACTION / (2.0 * strands.len() as f64);
    strands
        .into_iter()
        .map(|strand| {
            let drop_each = ((strand.len() as f64) * per_end) as usize;
            let end = strand.len().saturating_sub(drop_each).max(drop_each + 2);
            strand[drop_each..end].to_vec()
        })
        .filter(|strand| strand.len() >= 2)
        .collect()
}

/// The neon mode.
pub struct Neon;

impl VizMode for Neon {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("neon").expect("neon is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 2 {
            warn!("neon skipped: trajectory too short");
            return Ok(());
        }
        let mut rng = ctx.fork_rng("neon");
        let brick_seed = rng.next_u64();
        let jitter_seed = rng.next_u64();
        let dropout_strand_roll = rng.next_f64();

        // --- Sign-plane geometry: trajectory xy centered at the origin.
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for body in ctx.positions {
            for point in body.iter().step_by(31) {
                if point.x.is_finite() && point.y.is_finite() {
                    min_x = min_x.min(point.x);
                    max_x = max_x.max(point.x);
                    min_y = min_y.min(point.y);
                    max_y = max_y.max(point.y);
                }
            }
        }
        if !(min_x.is_finite() && min_y.is_finite()) {
            warn!("neon skipped: degenerate bounds");
            return Ok(());
        }
        let center = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5);
        let extent = (max_x - min_x).max(max_y - min_y).max(1e-9);
        let tube_radius = TUBE_RADIUS_FRACTION * extent;
        let bend_radius = MIN_BEND_DIAMETERS * 2.0 * tube_radius;
        let spacing = bend_radius * 0.5;

        // --- Strands: decimate aggressively, clamp curvature, split, trim.
        let mut strands: Vec<(usize, Vec<Vec3>)> = Vec::new();
        for body in 0..3 {
            let stride = (steps / 20_000).max(1);
            let raw: Vec<Vec3> = ctx.positions[body]
                .iter()
                .step_by(stride)
                .filter(|p| p.x.is_finite() && p.y.is_finite())
                .map(|p| Vec3::new(p.x - center.0, p.y - center.1, 0.0))
                .collect();
            let simplified = rdp_simplify_3d(&raw, extent * 0.012);
            let mut resampled = resample(&simplified, spacing);
            clamp_curvature(&mut resampled, spacing, bend_radius, 16);
            for strand in split_strands(&resampled) {
                strands.push((body, strand));
            }
        }
        info!("   neon: {} strands, tube radius {:.4}", strands.len(), tube_radius);

        // --- Base capsules (glass runs, electrodes, standoffs).
        let wall_gap = WALL_GAP_FRACTION * extent;
        let strand_color = |body: usize| -> (f64, f64, f64) {
            let (l, a, b) = ctx.mean_color(body);
            let (_, chroma, hue) = oklab_to_oklch(l, a, b);
            let (bl, ba, bb) = oklch_to_oklab(0.78, (chroma * 1.25).clamp(0.09, 0.24), hue);
            let (r, g, blue) = oklab_to_linear_rec2020(bl, ba, bb);
            (r.max(0.0) * EMISSION_GAIN, g.max(0.0) * EMISSION_GAIN, blue.max(0.0) * EMISSION_GAIN)
        };
        let electrode_color = (0.02, 0.02, 0.022);
        let standoff_color = (0.008, 0.008, 0.009);

        // (strand index or usize::MAX for hardware, capsule) pairs.
        let mut tagged: Vec<(usize, Capsule)> = Vec::new();
        for (strand_index, (body, strand)) in strands.iter().enumerate() {
            let color = strand_color(*body);
            for pair in strand.windows(2) {
                tagged.push((
                    strand_index,
                    Capsule {
                        a: pair[0],
                        b: pair[1],
                        radius: tube_radius,
                        emission_a: color,
                        emission_b: color,
                        core_darkening: 0.25,
                    },
                ));
            }
            // Electrode cylinders extending past the strand ends.
            for (end, direction) in [
                (strand[0], (strand[0] - strand[1]).normalize()),
                (
                    strand[strand.len() - 1],
                    (strand[strand.len() - 1] - strand[strand.len() - 2]).normalize(),
                ),
            ] {
                tagged.push((
                    usize::MAX,
                    Capsule {
                        a: end,
                        b: end + direction * (4.0 * tube_radius),
                        radius: tube_radius * 1.3,
                        emission_a: electrode_color,
                        emission_b: electrode_color,
                        core_darkening: 0.0,
                    },
                ));
            }
            // Standoff pins to the wall roughly every 10% of the extent.
            let pin_every = (extent * 0.10 / spacing) as usize;
            for point in strand.iter().step_by(pin_every.max(2)).skip(1) {
                tagged.push((
                    usize::MAX,
                    Capsule {
                        a: *point,
                        b: Vec3::new(point.x, point.y, -wall_gap),
                        radius: tube_radius * 0.35,
                        emission_a: standoff_color,
                        emission_b: standoff_color,
                        core_darkening: 0.0,
                    },
                ));
            }
        }

        let wall = || Plane {
            point: Vec3::new(0.0, 0.0, -wall_gap),
            normal: Vec3::new(0.0, 0.0, 1.0),
            u_axis: Vec3::new(1.0, 0.0, 0.0),
            albedo: (0.145, 0.115, 0.10),
            gloss: 0.0,
            extent: None,
            finish: PlaneFinish::Brick {
                brick_w: extent * 0.115,
                brick_h: extent * 0.052,
                mortar: extent * 0.007,
                relief: 0.8,
                seed: brick_seed,
            },
        };
        let build_scene = |amplitudes: &dyn Fn(usize) -> f64| -> Scene {
            let capsules: Vec<Capsule> = tagged
                .iter()
                .map(|(strand, capsule)| {
                    let scale = if *strand == usize::MAX { 1.0 } else { amplitudes(*strand) };
                    Capsule {
                        emission_a: (
                            capsule.emission_a.0 * scale,
                            capsule.emission_a.1 * scale,
                            capsule.emission_a.2 * scale,
                        ),
                        emission_b: (
                            capsule.emission_b.0 * scale,
                            capsule.emission_b.1 * scale,
                            capsule.emission_b.2 * scale,
                        ),
                        ..capsule.clone()
                    }
                })
                .collect();
            let mut scene = Scene::new(capsules, 32);
            scene.planes.push(wall());
            scene.glass_rim = 0.9;
            scene
        };

        let clip_black = ctx.settings.resolved_config.clip_black;
        let clip_white = ctx.settings.resolved_config.clip_white;
        let camera = {
            let distance = fit_distance(extent * 0.62, 36.0);
            let mut camera = orbit_camera(Vec3::zeros(), distance, 0.12, 0.04, 36.0);
            camera.position += Vec3::new(extent * 0.02, extent * 0.01, 0.0);
            camera
        };

        // --- Still: full res, shadowed wall wash.
        let still_scene = build_scene(&|_| 1.0);
        let still_w = ctx.quality.scale_dim(ctx.width);
        let still_h = ctx.quality.scale_dim(ctx.height);
        let (still_spp, still_shadow) = match ctx.quality {
            crate::viz::context::VizQuality::Final => (4u32, 12usize),
            crate::viz::context::VizQuality::Draft => (1u32, 4usize),
        };
        let params = RenderParams {
            width: still_w,
            height: still_h,
            spp: still_spp,
            max_steps: 160,
            shadows: true,
            shadow_emitters: still_shadow,
            jitter_seed,
        };
        let started = std::time::Instant::now();
        let pixels = render(&still_scene, &camera, &params);
        info!(
            "   neon still: {still_w}x{still_h} spp {still_spp} in {:.1}s",
            started.elapsed().as_secs_f64()
        );
        let mut rgba: PixelBuffer = Vec::new();
        fill_rgba(&pixels, &mut rgba);
        let image = grade_auto_levels(&rgba, still_w, still_h, clip_black, clip_white, 1.0);
        sink.save_png16(&image, "neon.png")?;

        // --- Flicker loop: per-strand periodic amplitude + rare dropout.
        let strand_count = strands.len().max(1);
        let dropout_strand = (dropout_strand_roll * strand_count as f64) as usize;
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16);
        let frame_count = ctx.quality.scale_count(LOOP_SECONDS * 30);
        let video_params = RenderParams {
            width: video_w,
            height: video_h,
            spp: 1,
            max_steps: 128,
            shadows: false,
            shadow_emitters: 0,
            jitter_seed,
        };
        // Loop-periodic dropout schedule: two 2-frame events per loop.
        let dropout_frames: Vec<usize> = vec![
            frame_count / 3,
            frame_count / 3 + 1,
            5 * frame_count / 6,
            5 * frame_count / 6 + 1,
        ];
        let amplitude_at = |strand: usize, phase: f64, frame: usize| -> f64 {
            let cycles = 1.0 + (strand % 3) as f64;
            let hum =
                0.5 + 0.5 * (std::f64::consts::TAU * (phase * cycles + strand as f64 * 0.37)).sin();
            let mut amplitude = 0.97 + 0.03 * hum;
            if strand == dropout_strand && dropout_frames.contains(&frame) {
                amplitude *= 0.22;
            }
            amplitude
        };

        let frame0_scene = build_scene(&|strand| amplitude_at(strand, 0.0, 0));
        let frame0 = render(&frame0_scene, &camera, &video_params);
        let mut frame0_rgba: PixelBuffer = Vec::new();
        fill_rgba(&frame0, &mut frame0_rgba);
        let levels = auto_levels(&frame0_rgba, clip_black, clip_white, 1.0);
        let video_started = std::time::Instant::now();
        let mut logged = false;
        stream_video(
            video_w,
            video_h,
            30,
            &sink.path("neon_loop.mp4"),
            &sink.path("neon_loop_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            &levels,
            |frame, rgba| {
                let phase = frame as f64 / frame_count.max(1) as f64;
                let scene = build_scene(&|strand| amplitude_at(strand, phase, frame));
                let pixels = render(&scene, &camera, &video_params);
                fill_rgba(&pixels, rgba);
                if frame == 0 && !logged {
                    logged = true;
                    let per_frame = video_started.elapsed().as_secs_f64();
                    info!(
                        "   neon loop: {per_frame:.2}s/frame, projected {:.1} min",
                        per_frame * frame_count as f64 / 60.0
                    );
                }
            },
        )?;
        sink.record("neon_loop.mp4", "video");
        sink.record("neon_loop_hq.mp4", "video");

        let meta = serde_json::json!({
            "strands": strand_count,
            "tube_radius": tube_radius,
            "bend_radius": bend_radius,
            "trim_fraction": TRIM_FRACTION,
            "wall_gap": wall_gap,
            "dropout_strand": dropout_strand,
            "loop_seconds": LOOP_SECONDS,
            "note": "glass-length cap applied as strand-end trim; see Wave 6 addendum",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("neon_params.json", &json, "data")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_spaces_points_uniformly() {
        let line = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0)];
        let resampled = resample(&line, 1.0);
        assert!(resampled.len() >= 10);
        let gap = (resampled[1] - resampled[0]).norm();
        assert!((gap - 1.0).abs() < 1e-9);
    }

    #[test]
    fn curvature_clamp_relaxes_sharp_corners() {
        let mut corner: Vec<Vec3> = (0..20)
            .map(|i| {
                if i < 10 {
                    Vec3::new(f64::from(i), 0.0, 0.0)
                } else {
                    Vec3::new(9.0, f64::from(i - 9), 0.0)
                }
            })
            .collect();
        let before_turn = {
            let a = (corner[9] - corner[8]).normalize();
            let b = (corner[10] - corner[9]).normalize();
            a.dot(&b).acos()
        };
        clamp_curvature(&mut corner, 1.0, 4.0, 24);
        let after_turn = {
            let a = (corner[9] - corner[8]).normalize();
            let b = (corner[10] - corner[9]).normalize();
            a.dot(&b).clamp(-1.0, 1.0).acos()
        };
        assert!(after_turn < before_turn, "clamp must soften the corner");
    }

    #[test]
    fn strands_are_capped() {
        let zigzag: Vec<Vec3> = (0..200)
            .map(|i| {
                let x = f64::from(i);
                let y = if (i / 25) % 2 == 0 { 0.0 } else { 25.0 };
                Vec3::new(x, y, 0.0)
            })
            .collect();
        let strands = split_strands(&zigzag);
        assert!(!strands.is_empty() && strands.len() <= MAX_STRANDS);
    }
}
