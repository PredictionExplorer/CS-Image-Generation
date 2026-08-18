//! V68 `reliquary` -- The Monument to Almost.
//!
//! Nine worldtubes in one glass block: the true history plus V24's eight
//! siblings (the same RNG domain, the same 1e-9 perturbation -- re-derived
//! bit-exactly), identical at the base and shearing apart as they climb.
//! Sibling tubes run at 0.7x radius, tinted from the body palette toward
//! glass-gray by `1 - exp(-delta/delta_0)` where delta is the distance to
//! the true worldline; a single etched ring engraves the fray altitude
//! where the median delta crosses 1% of the bounding box -- the Lyapunov
//! time, as a measurement. The bundle sits inside a rounded glass block
//! with single-bounce refraction (IOR 1.5), polished-face glints, faint
//! interior fog, and a soft floor gradient for the implied pedestal.
//! Scored by the GW strain pitched two octaves down.

use crate::error::Result;
use crate::oklab::oklab_to_linear_rec2020;
use crate::render::context::PixelBuffer;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::stream_video;
use crate::viz::common::audio::{
    self, SAMPLE_RATE, fade_ends, mux, normalize_to_lufs, write_wav_stereo_24bit,
};
use crate::viz::common::display::{auto_levels, grade_auto_levels};
use crate::viz::common::resim::{
    perturb_ensemble, recover_orientation, rerun, rotate_trajectories,
};
use crate::viz::common::tube_render::{
    Capsule, Fog, GlassBlock, Plane, PlaneFinish, RenderParams, Scene, Vec3, fit_distance,
    orbit_camera, rdp_simplify_3d, render,
};
use crate::viz::context::VizContext;
use crate::viz::modes::gw_chirp::strain_series;
use crate::viz::modes::multiverse::{EPSILON, SIBLINGS};
use crate::viz::modes::worldtube::{HEIGHT_FACTOR, RADIUS_FRACTION, fill_rgba};
use crate::viz::sink::ArtifactSink;
use nalgebra::Vector3;
use tracing::{info, warn};

/// Sibling tube radius relative to the true history.
const SIBLING_RADIUS: f64 = 0.7;
/// Divergence tint scale as a fraction of the bbox diagonal.
const DELTA_0_FRACTION: f64 = 0.002;
/// Fray threshold as a fraction of the bbox diagonal (median delta).
const FRAY_FRACTION: f64 = 0.01;
/// Glass index of refraction (single entry bounce).
const IOR: f64 = 1.5;
/// Video length in seconds at 30 fps.
const VIDEO_SECONDS: usize = 30;
/// Orbit climb over the video as a fraction of the block height.
const CLIMB_FRACTION: f64 = 0.06;
/// Decimation target for the true history's tubes (per body).
const TRUE_POINTS: usize = 6_000;
/// Decimation target for sibling tubes (per body).
const SIBLING_POINTS: usize = 2_200;
/// Emission scale for the true history.
const EMISSION_SCALE: f64 = 3.0;
/// Glass-gray the siblings desaturate toward (`OkLab`).
const GLASS_GRAY: (f64, f64, f64) = (0.70, -0.004, -0.012);

/// The reliquary mode.
pub struct Reliquary;

/// Divergence tint: lerp a body color toward glass-gray in `OkLab`.
pub(crate) fn divergence_tint(color: (f64, f64, f64), delta: f64, delta_0: f64) -> (f64, f64, f64) {
    let g = 1.0 - (-delta / delta_0.max(1e-12)).exp();
    (
        color.0 + (GLASS_GRAY.0 - color.0) * g,
        color.1 + (GLASS_GRAY.1 - color.1) * g,
        color.2 + (GLASS_GRAY.2 - color.2) * g,
    )
}

/// First step where the median (over siblings) divergence crosses the
/// threshold; `None` when the braid never frays.
pub(crate) fn fray_step(divergences: &[Vec<f64>], threshold: f64) -> Option<usize> {
    let steps = divergences.first().map_or(0, Vec::len);
    let mut medians = vec![0.0f64; 0];
    medians.reserve(steps);
    let mut scratch: Vec<f64> = Vec::with_capacity(divergences.len());
    for step in 0..steps {
        scratch.clear();
        scratch.extend(divergences.iter().map(|series| series[step]));
        scratch.sort_by(f64::total_cmp);
        medians.push(scratch[scratch.len() / 2]);
    }
    medians.iter().position(|&median| median >= threshold)
}

impl VizMode for Reliquary {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("reliquary").expect("reliquary is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 1_000 || ctx.bodies.len() != 3 {
            warn!("reliquary skipped: trajectory too short or bodies unavailable");
            return Ok(());
        }

        // --- The nine universes: V24's exact ensemble, dressed to the
        // master frame by the recovered Kabsch rotation.
        let mut rng = ctx.fork_rng("multiverse");
        let sibling_bodies = perturb_ensemble(ctx.bodies, SIBLINGS, EPSILON, &mut rng);
        let sim_started = std::time::Instant::now();
        let true_raw = rerun(ctx.bodies, steps);
        let rotation = recover_orientation(&true_raw, ctx.positions);
        info!(
            "   reliquary: true replay + Kabsch in {:.1}s; simulating {} siblings",
            sim_started.elapsed().as_secs_f64(),
            SIBLINGS
        );

        // --- Sculpture space from the true history (shared frame).
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
            warn!("reliquary skipped: degenerate bounds");
            return Ok(());
        }
        let center_x = 0.5 * (min_x + max_x);
        let center_y = 0.5 * (min_y + max_y);
        let diag = ((max_x - min_x).powi(2) + (max_y - min_y).powi(2)).sqrt().max(1e-9);
        let height = HEIGHT_FACTOR * diag;
        let time_scale = height / (steps.max(2) - 1) as f64;
        let delta_0 = DELTA_0_FRACTION * diag;
        let fray_threshold = FRAY_FRACTION * diag;
        let to_sculpture = |point: Vector3<f64>, step: usize| -> Vec3 {
            Vec3::new(
                point.x - center_x,
                step as f64 * time_scale - 0.5 * height,
                point.y - center_y,
            )
        };

        // --- Capsule chains: the true history, then the eight ghosts.
        let quality_scale = |target: usize| ctx.quality.scale_count(target * 4) / 4;
        let mut capsules: Vec<Capsule> = Vec::new();
        let mut bound_radius_sq = 0.0f64;
        let build_chain = |positions: &[Vec<Vector3<f64>>],
                           capsules: &mut Vec<Capsule>,
                           bound_radius_sq: &mut f64,
                           radius_factor: f64,
                           target_points: usize,
                           divergence: Option<&Vec<f64>>,
                           emission_factor: f64| {
            #[allow(clippy::needless_range_loop)]
            for body in 0..3 {
                let stride = (steps / 40_000).max(1);
                let raw: Vec<Vec3> = positions[body]
                    .iter()
                    .enumerate()
                    .step_by(stride)
                    .map(|(step, point)| to_sculpture(*point, step))
                    .collect();
                let mut kept = rdp_simplify_3d(&raw, 5.0e-4 * height);
                if kept.len() > target_points {
                    let keep_stride = kept.len().div_ceil(target_points);
                    let last = *kept.last().expect("nonempty polyline");
                    kept = kept.into_iter().step_by(keep_stride).collect();
                    if kept.last() != Some(&last) {
                        kept.push(last);
                    }
                }
                let step_of = |point: &Vec3| -> usize {
                    (((point.y + 0.5 * height) / time_scale).round() as usize).min(steps - 1)
                };
                for pair in kept.windows(2) {
                    let (a, b) = (pair[0], pair[1]);
                    let color_at = |step: usize| -> (f64, f64, f64) {
                        let (l, ca, cb) = ctx.colors[body][step.min(ctx.colors[body].len() - 1)];
                        let tinted = match divergence {
                            Some(series) => divergence_tint(
                                (l, ca, cb),
                                series[step.min(series.len() - 1)],
                                delta_0,
                            ),
                            None => (l, ca, cb),
                        };
                        let (r, g, bl) = oklab_to_linear_rec2020(tinted.0, tinted.1, tinted.2);
                        (
                            r.max(0.0) * EMISSION_SCALE * emission_factor,
                            g.max(0.0) * EMISSION_SCALE * emission_factor,
                            bl.max(0.0) * EMISSION_SCALE * emission_factor,
                        )
                    };
                    *bound_radius_sq = bound_radius_sq.max(a.norm_squared()).max(b.norm_squared());
                    capsules.push(Capsule {
                        a,
                        b,
                        radius: RADIUS_FRACTION * height * radius_factor,
                        emission_a: color_at(step_of(&a)),
                        emission_b: color_at(step_of(&b)),
                        core_darkening: 0.35,
                    });
                }
            }
        };

        // True history at full radius, full palette.
        build_chain(
            ctx.positions,
            &mut capsules,
            &mut bound_radius_sq,
            1.0,
            quality_scale(TRUE_POINTS),
            None,
            1.0,
        );

        // Siblings, one at a time (memory), with their divergence series.
        let mut divergences: Vec<Vec<f64>> = Vec::with_capacity(SIBLINGS);
        for (index, bodies) in sibling_bodies.iter().enumerate() {
            let raw = rerun(bodies, steps);
            let dressed = rotate_trajectories(&raw, &rotation);
            let series: Vec<f64> = (0..steps)
                .map(|step| {
                    (0..3)
                        .map(|body| (dressed[body][step] - ctx.positions[body][step]).norm())
                        .sum::<f64>()
                        / 3.0
                })
                .collect();
            build_chain(
                &dressed,
                &mut capsules,
                &mut bound_radius_sq,
                SIBLING_RADIUS,
                quality_scale(SIBLING_POINTS),
                Some(&series),
                0.55,
            );
            divergences.push(series);
            if index == 0 {
                info!(
                    "   reliquary: sibling replays ~{:.1}s each",
                    sim_started.elapsed().as_secs_f64()
                );
            }
        }

        // --- The fray ring: the measurement, engraved.
        let fray = fray_step(&divergences, fray_threshold);
        let fray_altitude = fray.map(|step| step as f64 * time_scale - 0.5 * height);
        if let Some(step) = fray {
            info!("   reliquary: braid frays at step {step} (median delta > 1% bbox)");
            let ring_y = fray_altitude.expect("altitude follows step");
            let ring_radius = bound_radius_sq.sqrt() * 0.55;
            let etch = (0.16, 0.17, 0.20);
            for segment in 0..96u32 {
                let theta0 = std::f64::consts::TAU * f64::from(segment) / 96.0;
                let theta1 = std::f64::consts::TAU * f64::from(segment + 1) / 96.0;
                capsules.push(Capsule {
                    a: Vec3::new(ring_radius * theta0.cos(), ring_y, ring_radius * theta0.sin()),
                    b: Vec3::new(ring_radius * theta1.cos(), ring_y, ring_radius * theta1.sin()),
                    radius: 0.0010 * height,
                    emission_a: etch,
                    emission_b: etch,
                    core_darkening: 0.0,
                });
            }
        } else {
            warn!("reliquary: the braid never frays within the run (no ring)");
        }
        info!("   reliquary: {} capsules in the block", capsules.len());

        // --- The glass block and its dark gallery.
        let radial_extent =
            capsules.iter().map(|capsule| capsule.a.x.hypot(capsule.a.z)).fold(0.0f64, f64::max);
        let mut scene = Scene::new(capsules, 28);
        let half = Vec3::new(radial_extent * 1.18, height * 0.56, radial_extent * 1.18);
        scene.glass_block = Some(GlassBlock {
            center: Vec3::zeros(),
            half,
            round: 0.07 * half.x.min(half.y),
            ior: IOR,
            sheen: (0.012, 0.013, 0.018),
            glint: 0.6,
        });
        scene.fog = Some(Fog { sigma_s: 0.10 / height, sigma_t: 0.16 / height });
        scene.planes.push(Plane {
            point: Vec3::new(0.0, -height * 0.62, 0.0),
            normal: Vec3::new(0.0, 1.0, 0.0),
            u_axis: Vec3::new(1.0, 0.0, 0.0),
            albedo: (0.030, 0.031, 0.036),
            gloss: 0.25,
            extent: None,
            finish: PlaneFinish::Plain,
        });
        let mut jitter_rng = ctx.fork_rng("reliquary");
        let jitter_seed = jitter_rng.next_u64();

        let clip_black = ctx.settings.resolved_config.clip_black;
        let clip_white = ctx.settings.resolved_config.clip_white;
        let distance = fit_distance(half.norm(), 34.0);

        // --- Hero still: low quarter view, the fray ring at the upper third.
        let hero_w = ctx.quality.scale_dim(ctx.width);
        let hero_h = ctx.quality.scale_dim(ctx.height);
        let hero_spp = match ctx.quality {
            crate::viz::context::VizQuality::Final => 4,
            crate::viz::context::VizQuality::Draft => 1,
        };
        let mut hero_camera = orbit_camera(Vec3::zeros(), distance, 0.72, -0.24, 34.0);
        hero_camera.look_at.y = -0.10 * height;
        let hero_params = RenderParams {
            width: hero_w,
            height: hero_h,
            spp: hero_spp,
            max_steps: 220,
            shadows: false,
            shadow_emitters: 0,
            jitter_seed,
        };
        let hero_started = std::time::Instant::now();
        let hero = render(&scene, &hero_camera, &hero_params);
        info!(
            "   reliquary hero: {hero_w}x{hero_h} spp {hero_spp} in {:.1}s",
            hero_started.elapsed().as_secs_f64()
        );
        let mut rgba: PixelBuffer = Vec::new();
        fill_rgba(&hero, &mut rgba);
        let image = grade_auto_levels(&rgba, hero_w, hero_h, clip_black, clip_white, 1.0);
        sink.save_png16(&image, "reliquary.png")?;

        // --- 30 s orbit with a 6% climb.
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16);
        let frame_count = ctx.quality.scale_count(VIDEO_SECONDS * 30);
        let video_spp = match ctx.quality {
            crate::viz::context::VizQuality::Final => 2,
            crate::viz::context::VizQuality::Draft => 1,
        };
        let camera_at = |frame: usize| {
            let phase = frame as f64 / frame_count.max(1) as f64;
            let mut camera =
                orbit_camera(Vec3::zeros(), distance, phase * std::f64::consts::TAU, 0.10, 34.0);
            camera.position.y += (phase - 0.5) * CLIMB_FRACTION * height;
            camera
        };
        let video_params = RenderParams {
            width: video_w,
            height: video_h,
            spp: video_spp,
            max_steps: 180,
            shadows: false,
            shadow_emitters: 0,
            jitter_seed,
        };
        let frame0 = render(&scene, &camera_at(0), &video_params);
        let mut frame0_rgba: PixelBuffer = Vec::new();
        fill_rgba(&frame0, &mut frame0_rgba);
        let levels = auto_levels(&frame0_rgba, clip_black, clip_white, 1.0);

        let video_started = std::time::Instant::now();
        let mut logged = false;
        stream_video(
            video_w,
            video_h,
            30,
            &sink.path("reliquary_silent.mp4"),
            &sink.path("reliquary_silent_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            &levels,
            |frame, rgba| {
                let pixels = render(&scene, &camera_at(frame), &video_params);
                fill_rgba(&pixels, rgba);
                if frame == 0 && !logged {
                    logged = true;
                    let per_frame = video_started.elapsed().as_secs_f64();
                    info!(
                        "   reliquary video: {per_frame:.2}s/frame, projected {:.1} min",
                        per_frame * frame_count as f64 / 60.0
                    );
                }
            },
        )?;

        // --- Score: the strain pitched two octaves down (time-stretch x4
        // around the loudest window), felt more than heard.
        {
            let (h_plus, h_cross) = strain_series(ctx.positions, &ctx.kinematics().masses);
            let duration = frame_count as f64 / 30.0;
            let sample_rate = f64::from(SAMPLE_RATE);
            let total_samples = (duration * sample_rate) as usize;
            // Stretch the full strain x4 (down two octaves), then window
            // the loudest stretch.
            let stretched_left = audio::resample_linear(&h_plus, total_samples * 4);
            let stretched_right = audio::resample_linear(&h_cross, total_samples * 4);
            let window_start = {
                let mut best = (0usize, f64::MIN);
                let hop = total_samples / 4;
                let mut start = 0usize;
                while start + total_samples <= stretched_left.len() {
                    let power: f64 = stretched_left[start..start + total_samples]
                        .iter()
                        .step_by(64)
                        .map(|&v| v * v)
                        .sum();
                    if power > best.1 {
                        best = (start, power);
                    }
                    start += hop;
                }
                best.0
            };
            let mut left = stretched_left[window_start..window_start + total_samples].to_vec();
            let mut right = stretched_right[window_start..window_start + total_samples].to_vec();
            audio::high_pass(&mut left, 16.0);
            audio::high_pass(&mut right, 16.0);
            let mut low = audio::Biquad::default();
            low.set_lowpass(80.0, 0.7, sample_rate);
            for value in &mut left {
                *value = low.process(*value);
            }
            let mut low_r = audio::Biquad::default();
            low_r.set_lowpass(80.0, 0.7, sample_rate);
            for value in &mut right {
                *value = low_r.process(*value);
            }
            normalize_to_lufs(&mut left, &mut right, -20.0, sample_rate);
            fade_ends(&mut left, (sample_rate * 0.5) as usize);
            fade_ends(&mut right, (sample_rate * 0.5) as usize);
            let mix_path = sink.path("reliquary_sub.wav");
            write_wav_stereo_24bit(&mix_path, &left, &right)?;
            for (silent, scored) in [
                ("reliquary_silent.mp4", "reliquary.mp4"),
                ("reliquary_silent_hq.mp4", "reliquary_hq.mp4"),
            ] {
                let silent_path = sink.path(silent);
                match mux(&silent_path, &mix_path, &sink.path(scored)) {
                    Ok(()) => {
                        sink.record(scored, "video");
                        let _ = std::fs::remove_file(&silent_path);
                    }
                    Err(error) => {
                        warn!("reliquary mux failed for {scored}: {error}; keeping silent cut");
                        let _ = std::fs::rename(&silent_path, sink.path(scored));
                        sink.record(scored, "video");
                    }
                }
            }
            let _ = std::fs::remove_file(&mix_path);
        }

        // --- The divergence profile (the fray ring's provenance).
        let sample_stride = (steps / 600).max(1);
        let profiles: Vec<serde_json::Value> = divergences
            .iter()
            .enumerate()
            .map(|(sibling, series)| {
                let crossing = series.iter().position(|&delta| delta >= fray_threshold);
                serde_json::json!({
                    "sibling": sibling,
                    "crossing_step": crossing,
                    "samples": series
                        .iter()
                        .step_by(sample_stride)
                        .map(|&delta| (delta / diag * 1e4).round() / 1e4)
                        .collect::<Vec<_>>(),
                })
            })
            .collect();
        let meta = serde_json::json!({
            "siblings": SIBLINGS,
            "epsilon_relative": EPSILON,
            "delta_0_fraction": DELTA_0_FRACTION,
            "fray_fraction": FRAY_FRACTION,
            "fray_step": fray,
            "fray_altitude": fray_altitude,
            "sculpture_height": height,
            "sibling_radius_factor": SIBLING_RADIUS,
            "ior": IOR,
            "sample_stride": sample_stride,
            "divergence_relative_to_bbox": profiles,
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("divergence_profile.json", &json, "data")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn divergence_tint_interpolates_between_palette_and_glass() {
        let color = (0.55, 0.18, -0.09);
        // Zero divergence: the body's own color, untouched.
        let same = divergence_tint(color, 0.0, 0.01);
        assert!((same.0 - color.0).abs() < 1e-12 && (same.1 - color.1).abs() < 1e-12);
        // Far beyond delta_0: fully glass-gray.
        let ghost = divergence_tint(color, 1.0, 0.01);
        assert!((ghost.0 - GLASS_GRAY.0).abs() < 1e-6);
        assert!(ghost.1.abs() < 0.01, "ghost chroma must collapse, got {}", ghost.1);
        // Monotone in delta.
        let quarter = divergence_tint(color, 0.005, 0.01);
        let half = divergence_tint(color, 0.02, 0.01);
        assert!(quarter.1 > half.1, "chroma must fall as delta grows");
    }

    #[test]
    fn fray_step_is_the_median_crossing() {
        // Five siblings; three cross the threshold at step 60, two never do.
        let series = |cross_at: Option<usize>| -> Vec<f64> {
            (0..100)
                .map(|step| match cross_at {
                    Some(at) if step >= at => 1.0,
                    _ => 0.0,
                })
                .collect()
        };
        let divergences =
            vec![series(Some(40)), series(Some(50)), series(Some(60)), series(None), series(None)];
        assert_eq!(fray_step(&divergences, 0.5), Some(60));
        // All-quiet ensembles never fray.
        let calm = vec![series(None), series(None), series(None)];
        assert_eq!(fray_step(&calm, 0.5), None);
    }

    #[test]
    fn identical_series_below_threshold_report_no_fray() {
        let divergences = vec![vec![0.001f64; 500], vec![0.002f64; 500], vec![0.0015f64; 500]];
        assert_eq!(fray_step(&divergences, 0.01), None);
    }
}
