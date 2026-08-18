//! V40 `aurora` -- Curtains Over the Void.
//!
//! The three trails extruded vertically into translucent curtains with
//! altitude-graded color (body hue low, drifting toward its complementary
//! high, with the green-oxygen skirt band boosted), vertical ray striations
//! from frozen 1D value noise, and a slow lateral dolly beneath them. The
//! curtains are baked into a shared emissive voxel grid and ray-marched
//! with front-to-back compositing; the void has no floor.

use crate::error::Result;
use crate::oklab::{oklab_to_linear_rec2020, oklab_to_oklch, oklch_to_oklab};
use crate::render::context::PixelBuffer;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::stream_video;
use crate::viz::common::display::{auto_levels, grade_auto_levels};
use crate::viz::common::tube_render::{DollyPath, Vec3, sample_jitter};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use rayon::prelude::*;
use tracing::{info, warn};

/// Curtain height as a fraction of the xy bounding-box diagonal.
const HEIGHT_FRACTION: f64 = 0.22;
/// Voxel grid resolution (x, y, z).
const GRID: (usize, usize, usize) = (256, 96, 256);
/// Vertical density profile falloff exponent factor.
const VERTICAL_FALLOFF: f64 = 3.0;
/// Green skirt band center as a fraction of local height.
const SKIRT_CENTER: f64 = 0.25;
/// Skirt boost factor.
const SKIRT_BOOST: f64 = 1.4;
/// Trail samples per body baked into the grid.
const BAKE_SAMPLES: usize = 24_000;
/// Video seconds at 30 fps.
const VIDEO_SECONDS: usize = 30;
/// Look-up pitch of the dolly camera in degrees.
const LOOK_UP_DEG: f64 = 12.0;
/// Altitude color LUT resolution.
const RAMP_STEPS: usize = 64;
/// Emission gain feeding the graders.
const EMISSION_GAIN: f64 = 3.2;
/// Absorption per unit density.
const SIGMA: f64 = 0.9;

/// Symmetric-difference dolly tangent (a one-sided look-ahead collapses to
/// a zero vector, hence NaN, at the clamped path ends).
fn path_tangent(path: &DollyPath, t: f64) -> Vec3 {
    (path.sample((t + 0.05).min(1.0)) - path.sample((t - 0.05).max(0.0))).normalize()
}

/// Deterministic 1D value noise (smoothstep-interpolated lattice hash).
fn value_noise(x: f64, seed: u64) -> f64 {
    let lattice = |i: i64| -> f64 {
        let mut state = (i as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(seed.wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
        state ^= state >> 33;
        state = state.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        state ^= state >> 33;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let i = x.floor() as i64;
    let f = x - x.floor();
    let smooth = f * f * (3.0 - 2.0 * f);
    lattice(i) * (1.0 - smooth) + lattice(i + 1) * smooth
}

/// Shared emissive voxel volume (density + premultiplied color).
struct CurtainVolume {
    density: Vec<f32>,
    color: Vec<(f32, f32, f32)>,
    /// World-space origin of voxel (0, 0, 0).
    origin: Vec3,
    /// World units per voxel along each axis.
    scale: Vec3,
}

impl CurtainVolume {
    #[inline]
    fn index(x: usize, y: usize, z: usize) -> usize {
        (z * GRID.1 + y) * GRID.0 + x
    }

    /// World bounds of the volume.
    fn bounds(&self) -> (Vec3, Vec3) {
        (
            self.origin,
            self.origin
                + Vec3::new(
                    self.scale.x * GRID.0 as f64,
                    self.scale.y * GRID.1 as f64,
                    self.scale.z * GRID.2 as f64,
                ),
        )
    }

    /// Trilinear (density, color) sample at a world position.
    #[inline]
    fn sample(&self, world: Vec3) -> (f64, (f64, f64, f64)) {
        let gx = ((world.x - self.origin.x) / self.scale.x - 0.5).clamp(0.0, (GRID.0 - 1) as f64);
        let gy = ((world.y - self.origin.y) / self.scale.y - 0.5).clamp(0.0, (GRID.1 - 1) as f64);
        let gz = ((world.z - self.origin.z) / self.scale.z - 0.5).clamp(0.0, (GRID.2 - 1) as f64);
        let (x0, y0, z0) = (gx.floor() as usize, gy.floor() as usize, gz.floor() as usize);
        let (x1, y1, z1) =
            ((x0 + 1).min(GRID.0 - 1), (y0 + 1).min(GRID.1 - 1), (z0 + 1).min(GRID.2 - 1));
        let (fx, fy, fz) = (gx - x0 as f64, gy - y0 as f64, gz - z0 as f64);

        let mut density = 0.0f64;
        let mut color = (0.0f64, 0.0f64, 0.0f64);
        for (zi, wz) in [(z0, 1.0 - fz), (z1, fz)] {
            for (yi, wy) in [(y0, 1.0 - fy), (y1, fy)] {
                for (xi, wx) in [(x0, 1.0 - fx), (x1, fx)] {
                    let weight = wz * wy * wx;
                    if weight <= 0.0 {
                        continue;
                    }
                    let index = Self::index(xi, yi, zi);
                    density += f64::from(self.density[index]) * weight;
                    let c = self.color[index];
                    color.0 += f64::from(c.0) * weight;
                    color.1 += f64::from(c.1) * weight;
                    color.2 += f64::from(c.2) * weight;
                }
            }
        }
        (density, color)
    }
}

/// Altitude color ramp for one body: base hue low, complementary high,
/// green-oxygen skirt boosted around 0.25 h.
fn altitude_ramp(mean_color: (f64, f64, f64)) -> Vec<(f64, f64, f64)> {
    let (_, chroma, hue) = oklab_to_oklch(mean_color.0, mean_color.1, mean_color.2);
    let chroma = chroma.max(0.08);
    (0..RAMP_STEPS)
        .map(|step| {
            let altitude = step as f64 / (RAMP_STEPS - 1) as f64;
            let ramp_hue = (hue + 180.0 * 0.85 * altitude).rem_euclid(360.0);
            let (l, a, b) = oklch_to_oklab(0.62, chroma, ramp_hue);
            let (mut r, mut g, mut bl) = oklab_to_linear_rec2020(l, a, b);
            let skirt = (-((altitude - SKIRT_CENTER) / 0.08).powi(2)).exp();
            if skirt > 1e-3 {
                let (gl, ga, gb) = oklch_to_oklab(0.66, 0.13, 145.0);
                let (gr, gg, gbl) = oklab_to_linear_rec2020(gl, ga, gb);
                let mix = skirt * (SKIRT_BOOST - 1.0);
                r = (r + gr * mix) * (1.0 + mix * 0.4);
                g = (g + gg * mix) * (1.0 + mix * 0.4);
                bl = (bl + gbl * mix) * (1.0 + mix * 0.4);
            }
            (r.max(0.0), g.max(0.0), bl.max(0.0))
        })
        .collect()
}

/// The aurora mode.
pub struct Aurora;

impl VizMode for Aurora {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("aurora").expect("aurora is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 2 {
            warn!("aurora skipped: trajectory too short");
            return Ok(());
        }

        // --- World frame: trajectory xy on the xz ground plane, y up.
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
            warn!("aurora skipped: degenerate trajectory bounds");
            return Ok(());
        }
        let diag = ((max_x - min_x).powi(2) + (max_y - min_y).powi(2)).sqrt().max(1e-9);
        let height0 = HEIGHT_FRACTION * diag;
        let pad_x = (max_x - min_x) * 0.08 + 1e-9;
        let pad_z = (max_y - min_y) * 0.08 + 1e-9;
        let origin = Vec3::new(min_x - pad_x, 0.0, min_y - pad_z);
        let extent =
            Vec3::new((max_x - min_x) + 2.0 * pad_x, height0 * 1.08, (max_y - min_y) + 2.0 * pad_z);
        let scale =
            Vec3::new(extent.x / GRID.0 as f64, extent.y / GRID.1 as f64, extent.z / GRID.2 as f64);

        // --- Bake the curtains.
        let mut rng = ctx.fork_rng("aurora");
        let noise_seed = rng.next_u64();
        let jitter_seed = rng.next_u64();
        let speed_window = ctx.kinematics().speed_window();
        let mut volume = CurtainVolume {
            density: vec![0.0; GRID.0 * GRID.1 * GRID.2],
            color: vec![(0.0, 0.0, 0.0); GRID.0 * GRID.1 * GRID.2],
            origin,
            scale,
        };
        let stride = (steps / BAKE_SAMPLES).max(1);
        for body in 0..3 {
            let ramp = altitude_ramp(ctx.mean_color(body));
            let mut arc = 0.0f64;
            let mut previous: Option<Vec3> = None;
            let striation_frequency = 42.0 / diag;
            for step in (0..steps).step_by(stride) {
                let position = ctx.positions[body][step];
                let world = Vec3::new(position.x, 0.0, position.y);
                if let Some(prev) = previous {
                    arc += (world - prev).norm();
                }
                previous = Some(world);

                let speed = ctx.kinematics().speeds[body][step];
                let normalized = ctx.kinematics().normalized_speed(speed_window, speed);
                let local_height = height0 * (0.6 + 0.4 * normalized);
                let noise = value_noise(
                    arc * striation_frequency + f64::from(u32::from(body as u8)) * 977.0,
                    noise_seed,
                );
                let striation = 0.45 + 1.1 * noise.powf(1.5);

                let gx = (world.x - origin.x) / scale.x;
                let gz = (world.z - origin.z) / scale.z;
                let ix = gx.floor() as i64;
                let iz = gz.floor() as i64;
                let top_voxel = ((local_height / scale.y).ceil() as usize).min(GRID.1 - 1);
                for dz in -1i64..=1 {
                    let z = iz + dz;
                    if z < 0 || z >= GRID.2 as i64 {
                        continue;
                    }
                    for dx in -1i64..=1 {
                        let x = ix + dx;
                        if x < 0 || x >= GRID.0 as i64 {
                            continue;
                        }
                        let lateral = (-((f64::from((dx * dx + dz * dz) as u32)) * 0.9)).exp();
                        for y in 0..=top_voxel {
                            let altitude_world = (y as f64 + 0.5) * scale.y;
                            let relative = altitude_world / local_height.max(1e-9);
                            if relative > 1.05 {
                                break;
                            }
                            let vertical = (-VERTICAL_FALLOFF * relative * relative).exp();
                            let deposit = (striation * lateral * vertical) as f32;
                            if deposit <= 1e-5 {
                                continue;
                            }
                            let ramp_index =
                                ((relative * (RAMP_STEPS - 1) as f64) as usize).min(RAMP_STEPS - 1);
                            let tint = ramp[ramp_index];
                            let index = CurtainVolume::index(x as usize, y, z as usize);
                            volume.density[index] += deposit;
                            let slot = &mut volume.color[index];
                            slot.0 += deposit * tint.0 as f32;
                            slot.1 += deposit * tint.1 as f32;
                            slot.2 += deposit * tint.2 as f32;
                        }
                    }
                }
            }
        }
        // Normalize density so sigma operates on a stable scale.
        let mut sorted: Vec<f32> =
            volume.density.iter().copied().filter(|&d| d > 0.0).step_by(7).collect();
        sorted.sort_by(f32::total_cmp);
        let reference = sorted
            .get(((sorted.len().saturating_sub(1)) as f64 * 0.98) as usize)
            .copied()
            .unwrap_or(1.0)
            .max(1e-6);
        let inv_reference = 1.0 / f64::from(reference);
        volume.density.par_iter_mut().for_each(|d| *d = (f64::from(*d) * inv_reference) as f32);
        volume.color.par_iter_mut().for_each(|c| {
            *c = (
                (f64::from(c.0) * inv_reference) as f32,
                (f64::from(c.1) * inv_reference) as f32,
                (f64::from(c.2) * inv_reference) as f32,
            );
        });

        // --- Dolly path beneath the curtains (full crossing over the video).
        let center = origin + extent * 0.5;
        let path = DollyPath {
            points: vec![
                Vec3::new(origin.x - extent.x * 0.15, height0 * 0.055, center.z + extent.z * 0.34),
                Vec3::new(center.x - extent.x * 0.18, height0 * 0.075, center.z + extent.z * 0.20),
                Vec3::new(center.x + extent.x * 0.18, height0 * 0.065, center.z + extent.z * 0.26),
                Vec3::new(origin.x + extent.x * 1.15, height0 * 0.08, center.z + extent.z * 0.32),
            ],
        };
        let march_step = scale.y.min(scale.x.min(scale.z)) * 0.55;
        let (vol_lo, vol_hi) = volume.bounds();
        let look_up = LOOK_UP_DEG.to_radians();

        let render_view = |position: Vec3,
                           forward: Vec3,
                           width: u32,
                           height: u32,
                           spp: u32,
                           rgba: &mut PixelBuffer| {
            let world_up = Vec3::new(0.0, 1.0, 0.0);
            let pitched = (forward * look_up.cos() + world_up * look_up.sin()).normalize();
            let right = pitched.cross(&world_up).normalize();
            let up = right.cross(&pitched);
            let tan_half = (52.0_f64.to_radians() * 0.5).tan();
            let aspect = f64::from(width) / f64::from(height);
            rgba.par_iter_mut().enumerate().for_each(|(index, pixel)| {
                let px = (index % width as usize) as f64;
                let py = (index / width as usize) as f64;
                let mut sum = (0.0f64, 0.0f64, 0.0f64);
                for sample in 0..spp {
                    let jitter = sample_jitter(index as u64, sample, jitter_seed);
                    let ndc_x =
                        ((px + jitter.0) / f64::from(width) * 2.0 - 1.0) * tan_half * aspect;
                    let ndc_y = (1.0 - (py + jitter.1) / f64::from(height) * 2.0) * tan_half;
                    let dir = (pitched + right * ndc_x + up * ndc_y).normalize();

                    let mut t_enter = 0.0f64;
                    let mut t_exit = f64::INFINITY;
                    for axis in 0..3 {
                        let inv = 1.0 / dir[axis];
                        let mut near = (vol_lo[axis] - position[axis]) * inv;
                        let mut far = (vol_hi[axis] - position[axis]) * inv;
                        if near > far {
                            std::mem::swap(&mut near, &mut far);
                        }
                        t_enter = t_enter.max(near);
                        t_exit = t_exit.min(far);
                    }
                    let interval_valid =
                        t_enter.is_finite() && t_exit.is_finite() && t_enter < t_exit;
                    if !interval_valid {
                        continue;
                    }
                    let mut transmittance = 1.0f64;
                    let mut accum = (0.0f64, 0.0f64, 0.0f64);
                    let mut ray_t = t_enter + march_step * jitter.1;
                    let mut marched = 0u32;
                    while ray_t < t_exit && marched < 640 {
                        let world = position + dir * ray_t;
                        let (density, color) = volume.sample(world);
                        if density > 1e-4 {
                            let tau = SIGMA * density * (march_step / scale.y);
                            let alpha = 1.0 - (-tau).exp();
                            let weight = transmittance * alpha * EMISSION_GAIN;
                            let inv_density = 1.0 / density;
                            accum.0 += weight * color.0 * inv_density;
                            accum.1 += weight * color.1 * inv_density;
                            accum.2 += weight * color.2 * inv_density;
                            transmittance *= 1.0 - alpha;
                            if transmittance < 1e-3 {
                                break;
                            }
                        }
                        ray_t += march_step;
                        marched += 1;
                    }
                    sum.0 += accum.0;
                    sum.1 += accum.1;
                    sum.2 += accum.2;
                }
                let inv = 1.0 / f64::from(spp.max(1));
                *pixel = (sum.0 * inv, sum.1 * inv, sum.2 * inv, 1.0);
            });
        };

        let clip_black = ctx.settings.resolved_config.clip_black;
        let clip_white = ctx.settings.resolved_config.clip_white;

        // --- Hero still: widest curtain overlap along the path.
        let mut best = (0.0f64, 0.30f64);
        for candidate in 0..24 {
            let t = f64::from(candidate) / 23.0 * 0.8 + 0.1;
            let position = path.sample(t);
            let ahead = path_tangent(&path, t);
            // Coarse coverage proxy: total density in a short cone ahead.
            let mut coverage = 0.0f64;
            for probe in 1..24 {
                let world = position
                    + ahead * (f64::from(probe) * extent.x / 30.0)
                    + Vec3::new(0.0, height0 * 0.35, 0.0);
                coverage += volume.sample(world).0;
            }
            if coverage > best.0 {
                best = (coverage, t);
            }
        }
        let hero_w = ctx.quality.scale_dim(ctx.width);
        let hero_h = ctx.quality.scale_dim(ctx.height);
        let hero_spp = match ctx.quality {
            crate::viz::context::VizQuality::Final => 4,
            crate::viz::context::VizQuality::Draft => 1,
        };
        let hero_position = path.sample(best.1);
        let hero_forward = path_tangent(&path, best.1);
        let started = std::time::Instant::now();
        let mut rgba: PixelBuffer = vec![(0.0, 0.0, 0.0, 0.0); hero_w as usize * hero_h as usize];
        render_view(hero_position, hero_forward, hero_w, hero_h, hero_spp, &mut rgba);
        info!(
            "   aurora hero: {hero_w}x{hero_h} spp {hero_spp} at path t={:.2} in {:.1}s",
            best.1,
            started.elapsed().as_secs_f64()
        );
        let image = grade_auto_levels(&rgba, hero_w, hero_h, clip_black, clip_white, 1.0);
        sink.save_png16(&image, "aurora.png")?;

        // --- Dolly video.
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16);
        let frame_count = ctx.quality.scale_count(VIDEO_SECONDS * 30);
        let mut frame0: PixelBuffer =
            vec![(0.0, 0.0, 0.0, 0.0); video_w as usize * video_h as usize];
        let start_pos = path.sample(0.0);
        let start_fwd = path_tangent(&path, 0.0);
        render_view(start_pos, start_fwd, video_w, video_h, 1, &mut frame0);
        let mut levels = auto_levels(&frame0, clip_black, clip_white, 1.0);
        levels.exposure_scale *= 1.12;

        let video_started = std::time::Instant::now();
        let mut logged = false;
        stream_video(
            video_w,
            video_h,
            30,
            &sink.path("aurora.mp4"),
            &sink.path("aurora_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            &levels,
            |frame, rgba| {
                let t = frame as f64 / (frame_count - 1).max(1) as f64;
                let position = path.sample(t);
                let forward = path_tangent(&path, t);
                render_view(position, forward, video_w, video_h, 1, rgba);
                if frame == 0 && !logged {
                    logged = true;
                    let per_frame = video_started.elapsed().as_secs_f64();
                    info!(
                        "   aurora video: {per_frame:.2}s/frame, projected {:.1} min",
                        per_frame * frame_count as f64 / 60.0
                    );
                }
            },
        )?;
        sink.record("aurora.mp4", "video");
        sink.record("aurora_hq.mp4", "video");

        let meta = serde_json::json!({
            "height0": height0,
            "grid": [GRID.0, GRID.1, GRID.2],
            "skirt": { "center": SKIRT_CENTER, "boost": SKIRT_BOOST },
            "hero_path_t": best.1,
            "frames": frame_count,
            "fps": 30,
            "note": "video 1 spp jittered; still 4 spp (see Wave 6 addendum)",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("aurora_params.json", &json, "data")?;
        Ok(())
    }
}
