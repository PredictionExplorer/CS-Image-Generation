//! V08 `shape-sphere` -- The Planet of Shapes.
//!
//! The triangle's shape (scale and rotation removed) mapped onto the
//! standard three-body shape sphere via the Hopf-style Jacobi map: binary
//! collisions sit on the equator 120 degrees apart, the two equilateral
//! (Lagrange) configurations at the poles. The orbit's path is splatted
//! onto a 2048x2048 equirectangular energy texture with dwell-weighted
//! brightness, decorated with graticule and landmark sigils, and rendered
//! as an emissive globe with a rim atmosphere; the video spins it about a
//! 23-degree tilted axis.

use crate::error::Result;
use crate::oklab::{oklab_to_linear_rec2020, oklab_to_oklch, oklch_to_oklab};
use crate::render::context::PixelBuffer;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::stream_video;
use crate::viz::common::display::{auto_levels, grade_auto_levels};
use crate::viz::common::raster::{Rgb64, draw_line, splat_line_additive};
use crate::viz::common::tube_render::{
    RenderParams, Scene, TexturedSphere, Vec3, fit_distance, orbit_camera, render,
};
use crate::viz::context::VizContext;
use crate::viz::modes::worldtube::fill_rgba;
use crate::viz::sink::ArtifactSink;
use nalgebra::{Matrix3, Vector3};
use tracing::{info, warn};

/// Equirectangular texture edge (square per the spec constants).
const TEXTURE_SIZE: usize = 2048;
/// Route decimation target.
const ROUTE_SAMPLES: usize = 40_000;
/// Axial tilt of the rotation video in degrees.
const AXIAL_TILT_DEG: f64 = 23.0;
/// Video length in seconds at 30 fps.
const VIDEO_SECONDS: usize = 30;
/// Graticule spacing in degrees.
const GRATICULE_DEG: f64 = 30.0;

/// Shape-sphere point for one triangle configuration (unit vector).
///
/// With Jacobi coordinates `rho = r2 - r1` and
/// `lambda = (2 r3 - r1 - r2) / sqrt(3)` (projected xy plane), the map is
/// `n = (2 rho.lambda, |lambda|^2 - |rho|^2, 2 rho x lambda) /
/// (|rho|^2 + |lambda|^2)`.
#[must_use]
pub fn shape_point(r1: (f64, f64), r2: (f64, f64), r3: (f64, f64)) -> Vector3<f64> {
    let rho = (r2.0 - r1.0, r2.1 - r1.1);
    let inv_sqrt3 = 1.0 / 3.0_f64.sqrt();
    let lambda = ((2.0 * r3.0 - r1.0 - r2.0) * inv_sqrt3, (2.0 * r3.1 - r1.1 - r2.1) * inv_sqrt3);
    let rho_sq = rho.0 * rho.0 + rho.1 * rho.1;
    let lambda_sq = lambda.0 * lambda.0 + lambda.1 * lambda.1;
    let scale = rho_sq + lambda_sq;
    if scale <= 1e-24 {
        return Vector3::new(0.0, 1.0, 0.0);
    }
    let dot = rho.0 * lambda.0 + rho.1 * lambda.1;
    let cross = rho.0 * lambda.1 - rho.1 * lambda.0;
    Vector3::new(2.0 * dot / scale, (lambda_sq - rho_sq) / scale, 2.0 * cross / scale)
}

/// Texture pixel of a shape point: longitude from `atan2(n2, n1)`, latitude
/// from `asin(n3)`, matching the tracer's equirect convention through the
/// orientation basis used below.
fn texture_uv(point: Vector3<f64>) -> (f64, f64) {
    let lon = point.y.atan2(point.x);
    let lat = point.z.clamp(-1.0, 1.0).asin();
    let u = (lon / std::f64::consts::TAU + 0.5).rem_euclid(1.0);
    let v = (0.5 - lat / std::f64::consts::PI).clamp(0.0, 1.0);
    (u * TEXTURE_SIZE as f64, v * TEXTURE_SIZE as f64)
}

/// Draw a segment in texture space, splitting at the longitude seam.
#[allow(clippy::too_many_arguments)]
fn splat_wrapped(
    weight: &mut [f32],
    value: &mut [f32],
    from: (f64, f64),
    to: (f64, f64),
    value_from: f32,
    value_to: f32,
    stroke: f64,
) {
    let size = TEXTURE_SIZE;
    let width = size as f64;
    let (mut x0, y0) = from;
    let (mut x1, y1) = to;
    if (x1 - x0).abs() > width * 0.5 {
        // Crossing the seam: shift one endpoint by a full wrap and draw the
        // segment twice, once for each side.
        if x1 > x0 {
            x1 -= width;
        } else {
            x0 -= width;
        }
        for shift in [0.0, width] {
            splat_line_additive(
                weight,
                value,
                size,
                size,
                ((x0 + shift) as f32, y0 as f32),
                ((x1 + shift) as f32, y1 as f32),
                value_from,
                value_to,
                stroke,
            );
        }
    } else {
        splat_line_additive(
            weight,
            value,
            size,
            size,
            (x0 as f32, y0 as f32),
            (x1 as f32, y1 as f32),
            value_from,
            value_to,
            stroke,
        );
    }
}

/// The shape-sphere mode.
pub struct ShapeSphere;

impl VizMode for ShapeSphere {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("shape-sphere").expect("shape-sphere is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 2 {
            warn!("shape-sphere skipped: trajectory too short");
            return Ok(());
        }

        // --- Route on the sphere, decimated and dwell-weighted.
        let stride = (steps / ROUTE_SAMPLES).max(1);
        let speed_window = ctx.kinematics().speed_window();
        let route: Vec<(usize, Vector3<f64>)> = (0..steps)
            .step_by(stride)
            .map(|step| {
                let p = |body: usize| (ctx.positions[body][step].x, ctx.positions[body][step].y);
                (step, shape_point(p(0), p(1), p(2)))
            })
            .collect();

        // --- Energy texture: additive dwell-weighted route splat.
        let size = TEXTURE_SIZE;
        let mut weight = vec![0.0f32; size * size];
        let mut value = vec![0.0f32; size * size];
        for pair in route.windows(2) {
            let (step, a) = pair[0];
            let (_, b) = pair[1];
            // Skip antipodal jumps across near-collinear passages: they
            // are artifacts of decimation, not real route segments.
            if a.dot(&b) < -0.2 {
                continue;
            }
            let speed = ctx.kinematics().speeds[0][step.min(steps - 1)];
            let normalized = ctx.kinematics().normalized_speed(speed_window, speed);
            let dwell = (1.0 - normalized).max(0.05);
            let stroke = 1.6 + 5.0 * dwell;
            splat_wrapped(
                &mut weight,
                &mut value,
                texture_uv(a),
                texture_uv(b),
                dwell as f32,
                dwell as f32,
                stroke,
            );
        }

        // --- Compose the emissive texture: graticule, route, landmarks.
        let ink = {
            let mut sum = (0.0, 0.0, 0.0);
            for body in 0..3 {
                let (l, a, b) = ctx.mean_color(body);
                sum = (sum.0 + l, sum.1 + a, sum.2 + b);
            }
            (sum.0 / 3.0, sum.1 / 3.0, sum.2 / 3.0)
        };
        let (_, ink_chroma, ink_hue) = oklab_to_oklch(ink.0, ink.1, ink.2);
        let route_lab = oklch_to_oklab(0.74, ink_chroma.max(0.06), ink_hue);
        let route_rgb = {
            let (r, g, b) = oklab_to_linear_rec2020(route_lab.0, route_lab.1, route_lab.2);
            (r.max(0.0), g.max(0.0), b.max(0.0))
        };
        let graticule: Rgb64 = (0.028, 0.032, 0.042);
        let sigil: Rgb64 = (0.30, 0.30, 0.34);

        let mut texture: Vec<Rgb64> = vec![(0.0015, 0.0018, 0.0028); size * size];
        // Graticule hairlines every 30 degrees.
        let lines = (360.0 / GRATICULE_DEG) as usize;
        for line in 0..lines {
            let x = line as f64 / lines as f64 * size as f64;
            draw_line(
                &mut texture,
                size,
                size,
                (x as f32, 0.0),
                (x as f32, size as f32),
                graticule,
                1.2,
                0.9,
            );
        }
        for line in 1..(180.0 / GRATICULE_DEG) as usize {
            let y = line as f64 * GRATICULE_DEG / 180.0 * size as f64;
            draw_line(
                &mut texture,
                size,
                size,
                (0.0, y as f32),
                (size as f32, y as f32),
                graticule,
                1.2,
                0.9,
            );
        }

        // Route energy over the base: normalize by the 99th percentile.
        let mut sorted: Vec<f32> = value.iter().copied().filter(|&v| v > 0.0).collect();
        sorted.sort_by(f32::total_cmp);
        let reference = sorted
            .get(((sorted.len().saturating_sub(1)) as f64 * 0.99) as usize)
            .copied()
            .unwrap_or(1.0)
            .max(1e-6);
        for (texel, &energy) in texture.iter_mut().zip(value.iter()) {
            let glow = (f64::from(energy / reference)).min(2.5).powf(0.62);
            texel.0 += route_rgb.0 * glow;
            texel.1 += route_rgb.1 * glow;
            texel.2 += route_rgb.2 * glow;
        }
        drop(weight);
        drop(value);

        // Landmarks: collision sigils on the equator (lon 90/210/330), star
        // caps at the Lagrange poles.
        let equator_y = size as f32 / 2.0;
        for lon_deg in [90.0f64, 210.0, 330.0] {
            let x = ((lon_deg / 360.0 + 0.5).rem_euclid(1.0) * size as f64) as f32;
            let radius = 26.0f32;
            let mut previous: Option<(f32, f32)> = None;
            for segment in 0..=24 {
                let theta = f64::from(segment) / 24.0 * std::f64::consts::TAU;
                let px = x + radius * theta.cos() as f32;
                let py = equator_y + radius * theta.sin() as f32;
                if let Some(prev) = previous {
                    draw_line(&mut texture, size, size, prev, (px, py), sigil, 2.2, 1.0);
                }
                previous = Some((px, py));
            }
            draw_line(
                &mut texture,
                size,
                size,
                (x - radius * 0.45, equator_y),
                (x + radius * 0.45, equator_y),
                sigil,
                2.2,
                1.0,
            );
        }
        for pole_v in [6.0f32, size as f32 - 6.0] {
            draw_line(
                &mut texture,
                size,
                size,
                (0.0, pole_v),
                (size as f32, pole_v),
                sigil,
                6.0,
                0.8,
            );
            let ray_len = 40.0f32;
            for ray in 0..6 {
                let x = (f64::from(ray) / 6.0 * size as f64) as f32;
                let (y0, y1) = if pole_v < size as f32 / 2.0 {
                    (pole_v, pole_v + ray_len)
                } else {
                    (pole_v - ray_len, pole_v)
                };
                draw_line(&mut texture, size, size, (x, y0), (x, y1), sigil, 2.0, 0.8);
            }
        }

        // --- Globe scene: analytic emissive sphere with rim atmosphere.
        let rim_lab = oklch_to_oklab(0.72, 0.055, ink_hue);
        let rim_rgb = {
            let (r, g, b) = oklab_to_linear_rec2020(rim_lab.0, rim_lab.1, rim_lab.2);
            (r.max(0.0) * 0.5, g.max(0.0) * 0.5, b.max(0.0) * 0.5)
        };
        let tilt = AXIAL_TILT_DEG.to_radians();
        let tilt_matrix =
            Matrix3::new(1.0, 0.0, 0.0, 0.0, tilt.cos(), -tilt.sin(), 0.0, tilt.sin(), tilt.cos());
        // Shape axes (n1, n2, n3) -> shader texture basis (x, z, y): the
        // polar n3 axis becomes texture y (latitude).
        let shape_to_tex = Matrix3::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0);
        let orientation_at = |spin: f64| -> Matrix3<f64> {
            let (s, c) = spin.sin_cos();
            let spin_matrix = Matrix3::new(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c);
            shape_to_tex * spin_matrix * tilt_matrix.transpose()
        };
        let mut scene = Scene::new(Vec::new(), 0);
        scene.spheres.push(TexturedSphere {
            center: Vec3::zeros(),
            radius: 1.0,
            texture,
            tex_w: size,
            tex_h: size,
            orientation: orientation_at(0.6),
            rim_color: rim_rgb,
            rim_strength: 0.9,
        });
        let mut rng = ctx.fork_rng("shape-sphere");
        let jitter_seed = rng.next_u64();
        let clip_black = ctx.settings.resolved_config.clip_black;
        let clip_white = ctx.settings.resolved_config.clip_white;
        let distance = fit_distance(1.0, 30.0) * 1.12;

        // Still: three-quarter view.
        let still_w = ctx.quality.scale_dim(ctx.width);
        let still_h = ctx.quality.scale_dim(ctx.height);
        let camera = orbit_camera(Vec3::zeros(), distance, 0.55, 0.30, 30.0);
        let params = RenderParams {
            width: still_w,
            height: still_h,
            spp: 2,
            max_steps: 32,
            shadows: false,
            shadow_emitters: 0,
            jitter_seed,
        };
        let pixels = render(&scene, &camera, &params);
        let mut rgba: PixelBuffer = Vec::new();
        fill_rgba(&pixels, &mut rgba);
        let image = grade_auto_levels(&rgba, still_w, still_h, clip_black, clip_white, 1.0);
        sink.save_png16(&image, "sphere.png")?;

        // Video: slow rotation about the tilted axis.
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16);
        let frame_count = ctx.quality.scale_count(VIDEO_SECONDS * 30);
        let video_camera = orbit_camera(Vec3::zeros(), distance, 0.55, 0.22, 30.0);
        let video_params = RenderParams {
            width: video_w,
            height: video_h,
            spp: 1,
            max_steps: 32,
            shadows: false,
            shadow_emitters: 0,
            jitter_seed,
        };
        scene.spheres[0].orientation = orientation_at(0.0);
        let frame0 = render(&scene, &video_camera, &video_params);
        let mut frame0_rgba: PixelBuffer = Vec::new();
        fill_rgba(&frame0, &mut frame0_rgba);
        let levels = auto_levels(&frame0_rgba, clip_black, clip_white, 1.0);
        let started = std::time::Instant::now();
        let mut logged = false;
        stream_video(
            video_w,
            video_h,
            30,
            &sink.path("sphere.mp4"),
            &sink.path("sphere_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            &levels,
            |frame, rgba| {
                let spin = frame as f64 / frame_count.max(1) as f64 * std::f64::consts::TAU;
                scene.spheres[0].orientation = orientation_at(spin);
                let pixels = render(&scene, &video_camera, &video_params);
                fill_rgba(&pixels, rgba);
                if frame == 0 && !logged {
                    logged = true;
                    let per_frame = started.elapsed().as_secs_f64();
                    info!(
                        "   shape-sphere video: {per_frame:.2}s/frame, projected {:.1} min",
                        per_frame * frame_count as f64 / 60.0
                    );
                }
            },
        )?;
        sink.record("sphere.mp4", "video");
        sink.record("sphere_hq.mp4", "video");

        // Route sidecar (decimated spherical coordinates).
        let route_stride = (route.len() / 4000).max(1);
        let series: Vec<serde_json::Value> = route
            .iter()
            .step_by(route_stride)
            .map(|&(step, n)| {
                serde_json::json!({
                    "step": step,
                    "lon_deg": n.y.atan2(n.x).to_degrees(),
                    "lat_deg": n.z.clamp(-1.0, 1.0).asin().to_degrees(),
                })
            })
            .collect();
        let meta = serde_json::json!({
            "texture": TEXTURE_SIZE,
            "route_samples": route.len(),
            "axial_tilt_deg": AXIAL_TILT_DEG,
            "graticule_deg": GRATICULE_DEG,
            "collision_longitudes_deg": [90.0, 210.0, 330.0],
            "route": series,
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("route.json", &json, "data")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equilateral_configurations_map_to_the_poles() {
        let sqrt3_half = 3.0_f64.sqrt() / 2.0;
        let counterclockwise = shape_point((0.0, 0.0), (1.0, 0.0), (0.5, sqrt3_half));
        assert!(counterclockwise.x.abs() < 1e-12);
        assert!(counterclockwise.y.abs() < 1e-12);
        assert!((counterclockwise.z - 1.0).abs() < 1e-12, "L4 must sit at the north pole");
        let clockwise = shape_point((0.0, 0.0), (1.0, 0.0), (0.5, -sqrt3_half));
        assert!((clockwise.z + 1.0).abs() < 1e-12, "L5 must sit at the south pole");
    }

    #[test]
    fn binary_collisions_sit_120_degrees_apart_on_the_equator() {
        // r1 = r2 collision.
        let c12 = shape_point((0.3, 0.4), (0.3, 0.4), (1.0, -0.2));
        assert!(c12.z.abs() < 1e-12, "collisions are collinear -> equator");
        let lon12 = c12.y.atan2(c12.x).to_degrees();
        assert!((lon12 - 90.0).abs() < 1e-9, "r1=r2 longitude {lon12}");
        // r1 = r3 collision.
        let c13 = shape_point((0.1, 0.9), (-0.7, 0.2), (0.1, 0.9));
        let lon13 = (c13.y.atan2(c13.x).to_degrees() + 360.0) % 360.0;
        assert!(c13.z.abs() < 1e-12);
        assert!((lon13 - 210.0).abs() < 1e-9, "r1=r3 longitude {lon13}");
        // r2 = r3 collision.
        let c23 = shape_point((0.5, -0.5), (-0.2, 0.8), (-0.2, 0.8));
        let lon23 = (c23.y.atan2(c23.x).to_degrees() + 360.0) % 360.0;
        assert!(c23.z.abs() < 1e-12);
        assert!((lon23 - 330.0).abs() < 1e-9, "r2=r3 longitude {lon23}");
    }

    #[test]
    fn shape_points_are_unit_vectors() {
        let n = shape_point((0.2, -1.1), (0.9, 0.3), (-0.4, 0.6));
        assert!((n.norm() - 1.0).abs() < 1e-12);
    }
}
