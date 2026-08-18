//! V43 `worldtube` -- The Spacetime Sculpture.
//!
//! The three trajectories extruded through time as braided glowing tubes:
//! worldtube space maps the projected xy plane onto xz and time onto the
//! vertical axis, scaled so the sculpture stands 1.6x its xy diagonal.
//! Sphere-traced with soft fog, ring gauges at time deciles, a full-res
//! hero still, and a 30 s helical orbit video. The capsule geometry is
//! shared with V45 `chandelier` and V50 `sculpture-export`.

use crate::error::Result;
use crate::oklab::oklab_to_linear_rec2020;
use crate::render::context::PixelBuffer;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::stream_video;
use crate::viz::common::display::{auto_levels, grade_auto_levels};
use crate::viz::common::tube_render::{
    Capsule, RenderParams, Scene, Vec3, fit_distance, orbit_camera, rdp_simplify_3d, render,
};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use tracing::{info, warn};

/// Sculpture height as a multiple of the xy bounding-box diagonal.
/// Exported for V68 `reliquary` (same worldtube space).
pub(crate) const HEIGHT_FACTOR: f64 = 1.6;
/// Tube radius as a fraction of the sculpture height.
/// Exported for V68 `reliquary`.
pub(crate) const RADIUS_FRACTION: f64 = 0.0035;
/// Decimation target per body after simplification.
const TARGET_POINTS: usize = 12_000;
/// RDP tolerance as a fraction of the sculpture height.
const RDP_TOLERANCE: f64 = 5.0e-4;
/// HDR emission scale applied to the body palette.
const EMISSION_SCALE: f64 = 3.0;
/// Interior core-line darkening strength.
const CORE_DARKENING: f64 = 0.45;
/// Number of time-decile ring gauges.
const GAUGE_COUNT: usize = 10;
/// Video length in seconds at 30 fps.
const VIDEO_SECONDS: usize = 30;
/// Hero still samples per pixel at final quality.
const HERO_SPP: u32 = 4;
/// Video samples per pixel at final quality (budget-bound; see addendum).
const VIDEO_SPP: u32 = 2;

/// Shared worldtube geometry: capsule chains in sculpture space.
pub(crate) struct WorldtubeGeometry {
    /// Capsule chains for all three bodies (trajectory xy on xz, time up).
    pub capsules: Vec<Capsule>,
    /// Sculpture height (time-axis span).
    pub height: f64,
    /// Bounding-sphere radius around the origin-centered sculpture.
    pub bound_radius: f64,
    /// Post-decimation point counts per body.
    pub point_counts: [usize; 3],
    /// Original trajectory-space center removed from xy (for mapping
    /// sculpture points back to master-frame coordinates).
    pub center_xy: (f64, f64),
}

/// Build the shared origin-centered worldtube capsule set.
pub(crate) fn worldtube_geometry(ctx: &VizContext<'_>, emission_scale: f64) -> WorldtubeGeometry {
    let steps = ctx.step_count();
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
        (min_x, max_x, min_y, max_y) = (-1.0, 1.0, -1.0, 1.0);
    }
    let center_x = 0.5 * (min_x + max_x);
    let center_y = 0.5 * (min_y + max_y);
    let diag = ((max_x - min_x).powi(2) + (max_y - min_y).powi(2)).sqrt().max(1e-9);
    let height = HEIGHT_FACTOR * diag;
    let time_scale = height / (steps.max(2) - 1) as f64;

    let speed_window = ctx.kinematics().speed_window();
    let mut capsules = Vec::new();
    let mut point_counts = [0usize; 3];
    let mut bound_radius_sq = 0.0f64;

    #[allow(clippy::needless_range_loop)]
    for body in 0..3 {
        // Pre-stride to a tractable RDP input, then simplify. The time
        // coordinate is strictly monotonic in the step index, so kept
        // points map back to steps exactly.
        let stride = (steps / 40_000).max(1);
        let raw: Vec<Vec3> = ctx.positions[body]
            .iter()
            .enumerate()
            .step_by(stride)
            .map(|(step, point)| {
                Vec3::new(
                    point.x - center_x,
                    step as f64 * time_scale - 0.5 * height,
                    point.y - center_y,
                )
            })
            .collect();
        let mut kept = rdp_simplify_3d(&raw, RDP_TOLERANCE * height);
        if kept.len() > TARGET_POINTS {
            let keep_stride = kept.len().div_ceil(TARGET_POINTS);
            let last = *kept.last().expect("nonempty polyline");
            kept = kept.into_iter().step_by(keep_stride).collect();
            if kept.last() != Some(&last) {
                kept.push(last);
            }
        }
        point_counts[body] = kept.len();

        let step_of = |point: &Vec3| -> usize {
            (((point.y + 0.5 * height) / time_scale).round() as usize).min(steps - 1)
        };
        for pair in kept.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let step_a = step_of(&a);
            let step_b = step_of(&b);
            let mid_step = usize::midpoint(step_a, step_b);
            let speed = ctx.kinematics().speeds[body][mid_step.min(steps - 1)];
            let normalized = ctx.kinematics().normalized_speed(speed_window, speed);
            let radius = RADIUS_FRACTION * height * (1.3 - 0.6 * normalized);
            let color_at = |step: usize| {
                let (l, ca, cb) = ctx.colors[body][step.min(ctx.colors[body].len() - 1)];
                let (r, g, bl) = oklab_to_linear_rec2020(l, ca, cb);
                (
                    r.max(0.0) * emission_scale,
                    g.max(0.0) * emission_scale,
                    bl.max(0.0) * emission_scale,
                )
            };
            bound_radius_sq = bound_radius_sq.max(a.norm_squared()).max(b.norm_squared());
            capsules.push(Capsule {
                a,
                b,
                radius,
                emission_a: color_at(step_a),
                emission_b: color_at(step_b),
                core_darkening: CORE_DARKENING,
            });
        }
    }

    WorldtubeGeometry {
        capsules,
        height,
        bound_radius: bound_radius_sq.sqrt().max(1e-9),
        point_counts,
        center_xy: (center_x, center_y),
    }
}

/// Ring gauges: faint tori around the bundle at time deciles.
fn ring_gauges(geometry: &WorldtubeGeometry) -> Vec<Capsule> {
    let ring_radius = geometry
        .capsules
        .iter()
        .map(|capsule| capsule.a.x.hypot(capsule.a.z))
        .fold(0.0f64, f64::max)
        * 1.05;
    let tube_radius = 0.0012 * geometry.height;
    let emission = (0.055, 0.06, 0.075);
    let mut gauges = Vec::with_capacity(GAUGE_COUNT * 48);
    for gauge in 1..=GAUGE_COUNT {
        let y = -0.5 * geometry.height + geometry.height * gauge as f64 / (GAUGE_COUNT + 1) as f64;
        for segment in 0..48u32 {
            let theta0 = std::f64::consts::TAU * f64::from(segment) / 48.0;
            let theta1 = std::f64::consts::TAU * f64::from(segment + 1) / 48.0;
            gauges.push(Capsule {
                a: Vec3::new(ring_radius * theta0.cos(), y, ring_radius * theta0.sin()),
                b: Vec3::new(ring_radius * theta1.cos(), y, ring_radius * theta1.sin()),
                radius: tube_radius,
                emission_a: emission,
                emission_b: emission,
                core_darkening: 0.0,
            });
        }
    }
    gauges
}

/// Convert a linear Rec.2020 frame into the RGBA buffer graders expect.
pub(crate) fn fill_rgba(pixels: &[(f64, f64, f64)], rgba: &mut PixelBuffer) {
    rgba.clear();
    rgba.extend(pixels.iter().map(|&(r, g, b)| (r, g, b, 1.0)));
}

/// The worldtube mode.
pub struct Worldtube;

impl VizMode for Worldtube {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("worldtube").expect("worldtube is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 2 {
            warn!("worldtube skipped: trajectory too short");
            return Ok(());
        }
        let geometry = worldtube_geometry(ctx, EMISSION_SCALE);
        let mut capsules = geometry.capsules.clone();
        capsules.extend(ring_gauges(&geometry));
        let mut scene = Scene::new(capsules, 24);
        scene.fog = Some(crate::viz::common::tube_render::Fog {
            sigma_s: 0.30 / geometry.height,
            sigma_t: 0.40 / geometry.height,
        });
        let mut rng = ctx.fork_rng("worldtube");
        let jitter_seed = rng.next_u64();

        let clip_black = ctx.settings.resolved_config.clip_black;
        let clip_white = ctx.settings.resolved_config.clip_white;
        let distance = fit_distance(geometry.bound_radius, 34.0);
        let center = Vec3::zeros();

        // --- Hero still: low three-quarter view looking up the time axis.
        let hero_w = ctx.quality.scale_dim(ctx.width);
        let hero_h = ctx.quality.scale_dim(ctx.height);
        let hero_spp = match ctx.quality {
            crate::viz::context::VizQuality::Final => HERO_SPP,
            crate::viz::context::VizQuality::Draft => 1,
        };
        let hero_camera = orbit_camera(center, distance, 0.65, -0.30, 34.0);
        let hero_params = RenderParams {
            width: hero_w,
            height: hero_h,
            spp: hero_spp,
            max_steps: 200,
            shadows: false,
            shadow_emitters: 0,
            jitter_seed,
        };
        let started = std::time::Instant::now();
        let hero = render(&scene, &hero_camera, &hero_params);
        info!(
            "   worldtube hero: {hero_w}x{hero_h} spp {hero_spp} in {:.1}s ({} capsules)",
            started.elapsed().as_secs_f64(),
            scene.set.capsules().len()
        );
        let mut rgba: PixelBuffer = Vec::new();
        fill_rgba(&hero, &mut rgba);
        let image = grade_auto_levels(&rgba, hero_w, hero_h, clip_black, clip_white, 1.0);
        sink.save_png16(&image, "worldtube.png")?;

        // --- Helical orbit video: one revolution plus an 8% climb.
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16);
        let frame_count = ctx.quality.scale_count(VIDEO_SECONDS * 30);
        let video_spp = match ctx.quality {
            crate::viz::context::VizQuality::Final => VIDEO_SPP,
            crate::viz::context::VizQuality::Draft => 1,
        };
        let camera_at = |frame: usize| {
            let phase = frame as f64 / frame_count.max(1) as f64;
            let mut camera =
                orbit_camera(center, distance, phase * std::f64::consts::TAU, 0.18, 34.0);
            camera.position.y += (phase - 0.5) * 0.08 * geometry.height;
            camera
        };
        let video_params = RenderParams {
            width: video_w,
            height: video_h,
            spp: video_spp,
            max_steps: 160,
            shadows: false,
            shadow_emitters: 0,
            jitter_seed,
        };

        // Fixed levels from frame 0 keep the exposure steady across the
        // sweep; the first frame is re-rendered inside the stream.
        let frame0 = render(&scene, &camera_at(0), &video_params);
        let mut frame0_rgba: PixelBuffer = Vec::new();
        fill_rgba(&frame0, &mut frame0_rgba);
        let levels = auto_levels(&frame0_rgba, clip_black, clip_white, 1.0);

        let video_started = std::time::Instant::now();
        let mut first_frame_logged = false;
        stream_video(
            video_w,
            video_h,
            30,
            &sink.path("worldtube.mp4"),
            &sink.path("worldtube_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            &levels,
            |frame, rgba| {
                let pixels = render(&scene, &camera_at(frame), &video_params);
                fill_rgba(&pixels, rgba);
                if frame == 0 && !first_frame_logged {
                    first_frame_logged = true;
                    let per_frame = video_started.elapsed().as_secs_f64();
                    info!(
                        "   worldtube video: {video_w}x{video_h} spp {video_spp}, \
                         {per_frame:.2}s/frame, projected {:.1} min for {frame_count} frames",
                        per_frame * frame_count as f64 / 60.0
                    );
                }
            },
        )?;
        sink.record("worldtube.mp4", "video");
        sink.record("worldtube_hq.mp4", "video");

        let meta = serde_json::json!({
            "height": geometry.height,
            "height_factor": HEIGHT_FACTOR,
            "radius_fraction": RADIUS_FRACTION,
            "point_counts": geometry.point_counts,
            "capsules": scene.set.capsules().len(),
            "gauges": GAUGE_COUNT,
            "video": { "frames": frame_count, "fps": 30, "spp": video_spp },
            "hero_spp": hero_spp,
            "note": "ring gauge labels deferred until common/text.rs (posters wave)",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("worldtube.json", &json, "data")?;
        Ok(())
    }
}
