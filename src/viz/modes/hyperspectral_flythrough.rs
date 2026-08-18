//! V02 `hyperspectral-flythrough` -- Flying Through the Spectrum.
//!
//! The accumulated SPD treated as a translucent (x, y, lambda) volume:
//! wavelength becomes depth, and a spline camera dives from the violet face
//! to the red face with a slow barrel roll. Emissive-absorptive front-to-
//! back compositing over a downsampled energy grid; per-slab colors come
//! from `wavelength_to_rgb`. SPD phase (needs the live buffer).

use crate::error::Result;
use crate::render::context::PixelBuffer;
use crate::spectrum::{NUM_BINS, wavelength_nm_for_bin, wavelength_to_rgb};
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::stream_video;
use crate::viz::common::tube_render::{DollyPath, Vec3, sample_jitter};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use crate::viz::{VizMode, VizPhase};
use rayon::prelude::*;
use tracing::{info, warn};

/// Output video frames at final quality (24 s at 60 fps).
const VIDEO_FRAMES: usize = 1440;
/// Slab spacing as a fraction of the shorter volume axis (1/96).
const SLAB_SPACING_FACTOR: f64 = 1.0 / 96.0;
/// Calibration target: mean transmittance through half the volume.
const MID_TRANSMITTANCE: f64 = 0.35;
/// Maximum barrel roll in degrees.
const MAX_ROLL_DEG: f64 = 20.0;
/// Early-out transmittance floor.
const TRANSMITTANCE_FLOOR: f64 = 1e-3;
/// Volume downsample factor relative to the main render.
const VOLUME_DOWNSAMPLE: u32 = 4;
/// Overall emission gain feeding the run's tonemap levels.
const EMISSION_GAIN: f64 = 2.4;

/// The hyperspectral flythrough mode.
pub struct HyperspectralFlythrough;

/// Downsampled emissive volume: 64 slabs of scalar energy.
struct Volume {
    energy: Vec<f32>,
    width: usize,
    height: usize,
    /// Slab spacing in volume pixel units.
    slab_dz: f64,
    /// Per-slab linear Rec.2020 color.
    slab_color: [(f64, f64, f64); NUM_BINS],
}

impl Volume {
    /// Depth extent in volume pixel units.
    fn depth(&self) -> f64 {
        NUM_BINS as f64 * self.slab_dz
    }

    /// Trilinear energy sample; `z` in volume pixels.
    #[inline]
    fn sample(&self, x: f64, y: f64, z: f64) -> (f32, usize, f64) {
        let slab_pos = (z / self.slab_dz - 0.5).clamp(0.0, (NUM_BINS - 1) as f64);
        let slab0 = slab_pos.floor() as usize;
        let slab1 = (slab0 + 1).min(NUM_BINS - 1);
        let fz = slab_pos - slab0 as f64;

        let xc = x.clamp(0.0, (self.width - 1) as f64);
        let yc = y.clamp(0.0, (self.height - 1) as f64);
        let x0 = xc.floor() as usize;
        let y0 = yc.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = xc - x0 as f64;
        let fy = yc - y0 as f64;

        let plane = self.width * self.height;
        let bilinear = |slab: usize| -> f64 {
            let base = slab * plane;
            let sample = |px: usize, py: usize| f64::from(self.energy[base + py * self.width + px]);
            let top = sample(x0, y0) * (1.0 - fx) + sample(x1, y0) * fx;
            let bottom = sample(x0, y1) * (1.0 - fx) + sample(x1, y1) * fx;
            top * (1.0 - fy) + bottom * fy
        };
        let value = bilinear(slab0) * (1.0 - fz) + bilinear(slab1) * fz;
        (value as f32, slab0, fz)
    }
}

/// Build the downsampled volume from the live SPD buffer.
fn build_volume(spd: &[[f64; NUM_BINS]], width: usize, height: usize, factor: usize) -> Volume {
    let vw = (width / factor).max(8);
    let vh = (height / factor).max(8);
    let plane = vw * vh;
    let mut energy = vec![0.0f32; plane * NUM_BINS];

    energy.par_chunks_mut(plane).enumerate().for_each(|(bin, slab)| {
        for vy in 0..vh {
            for vx in 0..vw {
                let mut sum = 0.0f64;
                let mut count = 0.0f64;
                for sy in 0..factor {
                    let y = vy * factor + sy;
                    if y >= height {
                        continue;
                    }
                    for sx in 0..factor {
                        let x = vx * factor + sx;
                        if x >= width {
                            continue;
                        }
                        sum += spd[y * width + x][bin];
                        count += 1.0;
                    }
                }
                slab[vy * vw + vx] = (sum / count.max(1.0)) as f32;
            }
        }
    });

    let slab_dz = vw.min(vh) as f64 * SLAB_SPACING_FACTOR;
    let mut slab_color = [(0.0, 0.0, 0.0); NUM_BINS];
    for (bin, slot) in slab_color.iter_mut().enumerate() {
        let (r, g, b) = wavelength_to_rgb(wavelength_nm_for_bin(bin));
        *slot = (r.max(0.0), g.max(0.0), b.max(0.0));
    }
    Volume { energy, width: vw, height: vh, slab_dz, slab_color }
}

/// Sigma such that mean transmittance through half the volume is the target.
fn calibrate_sigma(volume: &Volume) -> f64 {
    let plane = volume.width * volume.height;
    let half_depth: f64 = (0..plane)
        .into_par_iter()
        .map(|pixel| {
            (0..NUM_BINS / 2).map(|bin| f64::from(volume.energy[bin * plane + pixel])).sum::<f64>()
        })
        .sum::<f64>()
        / plane as f64;
    if half_depth <= 1e-12 {
        return 1.0;
    }
    // Optical depth accumulates as sigma * energy * (step / slab_dz), and
    // the mean half-volume line integral in those units is `half_depth`.
    -MID_TRANSMITTANCE.ln() / half_depth
}

/// Smoothstep easing.
#[inline]
fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl VizMode for HyperspectralFlythrough {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("hyperspectral-flythrough")
            .expect("hyperspectral-flythrough is in the catalog")
    }

    fn phase(&self) -> VizPhase {
        VizPhase::Spd
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let Some(spd) = ctx.accum_spd else {
            warn!("hyperspectral-flythrough skipped: no SPD buffer (image-only run)");
            return Ok(());
        };
        let volume =
            build_volume(spd, ctx.width as usize, ctx.height as usize, VOLUME_DOWNSAMPLE as usize);
        let sigma = calibrate_sigma(&volume);
        let depth = volume.depth();
        let vw = volume.width as f64;
        let vh = volume.height as f64;
        let center = Vec3::new(vw * 0.5, vh * 0.5, depth * 0.5);

        // Spline: wide violet-side three-quarter dive through to the red
        // face. Coordinates are volume pixels; z is the lambda axis.
        let path = DollyPath {
            points: vec![
                center + Vec3::new(-0.62 * vw, 0.34 * vh, -1.05 * depth - 0.35 * vw),
                center + Vec3::new(-0.22 * vw, 0.12 * vh, -0.55 * depth),
                center + Vec3::new(0.02 * vw, -0.02 * vh, 0.05 * depth),
                center + Vec3::new(0.16 * vw, -0.10 * vh, 0.62 * depth),
                center + Vec3::new(0.30 * vw, -0.18 * vh, 1.05 * depth + 0.30 * vw),
            ],
        };

        let out_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16);
        let out_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16);
        let frame_count = ctx.quality.scale_count(VIDEO_FRAMES);
        let supersample = match ctx.quality {
            crate::viz::context::VizQuality::Final => 2u32,
            crate::viz::context::VizQuality::Draft => 1u32,
        };
        let mut rng = ctx.fork_rng("hyperspectral-flythrough");
        let jitter_seed = rng.next_u64();
        let step = volume.slab_dz * 0.5;
        let vfov: f64 = 46.0;
        let tan_half = (vfov.to_radians() * 0.5).tan();
        let aspect = f64::from(out_w) / f64::from(out_h);

        info!(
            "   hyperspectral-flythrough: volume {}x{}x{NUM_BINS}, sigma {sigma:.3e}, \
             {out_w}x{out_h} x{frame_count} frames (supersample {supersample})",
            volume.width, volume.height
        );

        let render_frame = |frame: usize, rgba: &mut PixelBuffer| {
            let t = smoothstep(frame as f64 / (frame_count - 1).max(1) as f64);
            let position = path.sample(t);
            // Symmetric-difference tangent: a one-sided look-ahead collapses
            // to a zero vector at the clamped final frames, and normalizing
            // it poisons the ray/box clip with NaNs.
            let ahead = path.sample((t + 0.06).min(1.0));
            let behind = path.sample((t - 0.06).max(0.0));
            let forward = (ahead - behind).normalize();
            let roll = MAX_ROLL_DEG.to_radians() * (std::f64::consts::PI * t).sin();
            let world_up = Vec3::new(0.0, 1.0, 0.0);
            let right0 = forward.cross(&world_up).normalize();
            let up0 = right0.cross(&forward);
            let (sin_roll, cos_roll) = roll.sin_cos();
            let right = right0 * cos_roll + up0 * sin_roll;
            let up = up0 * cos_roll - right0 * sin_roll;

            rgba.par_iter_mut().enumerate().for_each(|(index, pixel)| {
                let px = (index % out_w as usize) as f64;
                let py = (index / out_w as usize) as f64;
                let mut color = (0.0f64, 0.0f64, 0.0f64);
                for sample in 0..supersample {
                    let jitter = sample_jitter(index as u64, sample, jitter_seed);
                    let ndc_x =
                        ((px + jitter.0) / f64::from(out_w) * 2.0 - 1.0) * tan_half * aspect;
                    let ndc_y = (1.0 - (py + jitter.1) / f64::from(out_h) * 2.0) * tan_half;
                    let dir = (forward + right * ndc_x + up * ndc_y).normalize();

                    // Clip to the volume box.
                    let lo = Vec3::new(0.0, 0.0, 0.0);
                    let hi = Vec3::new(vw - 1.0, vh - 1.0, depth);
                    let mut t_enter = 0.0f64;
                    let mut t_exit = f64::INFINITY;
                    for axis in 0..3 {
                        let inv = 1.0 / dir[axis];
                        let mut near = (lo[axis] - position[axis]) * inv;
                        let mut far = (hi[axis] - position[axis]) * inv;
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
                    let mut ray_t = t_enter + step * jitter.1;
                    let mut marched = 0u32;
                    while ray_t < t_exit && marched < 640 {
                        marched += 1;
                        let p = position + dir * ray_t;
                        let (energy, slab, fz) = volume.sample(p.x, p.y, p.z);
                        if energy > 0.0 {
                            let tau = sigma * f64::from(energy) * (step / volume.slab_dz);
                            let alpha = 1.0 - (-tau).exp();
                            let c0 = volume.slab_color[slab];
                            let c1 = volume.slab_color[(slab + 1).min(NUM_BINS - 1)];
                            let tint = (
                                c0.0 + (c1.0 - c0.0) * fz,
                                c0.1 + (c1.1 - c0.1) * fz,
                                c0.2 + (c1.2 - c0.2) * fz,
                            );
                            let weight = transmittance * alpha * EMISSION_GAIN;
                            accum.0 += weight * tint.0;
                            accum.1 += weight * tint.1;
                            accum.2 += weight * tint.2;
                            transmittance *= 1.0 - alpha;
                            if transmittance < TRANSMITTANCE_FLOOR {
                                break;
                            }
                        }
                        ray_t += step;
                    }
                    color.0 += accum.0;
                    color.1 += accum.1;
                    color.2 += accum.2;
                }
                let inv = 1.0 / f64::from(supersample);
                *pixel = (color.0 * inv, color.1 * inv, color.2 * inv, 1.0);
            });
        };

        let started = std::time::Instant::now();
        let mut logged = false;
        stream_video(
            out_w,
            out_h,
            60,
            &sink.path("flythrough.mp4"),
            &sink.path("flythrough_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            ctx.levels,
            |frame, rgba| {
                render_frame(frame, rgba);
                if frame == 0 && !logged {
                    logged = true;
                    let per_frame = started.elapsed().as_secs_f64();
                    info!(
                        "   flythrough: {per_frame:.2}s/frame, projected {:.1} min total",
                        per_frame * frame_count as f64 / 60.0
                    );
                }
            },
        )?;
        sink.record("flythrough.mp4", "video");
        sink.record("flythrough_hq.mp4", "video");

        let meta = serde_json::json!({
            "volume": [volume.width, volume.height, NUM_BINS],
            "slab_dz": volume.slab_dz,
            "sigma": sigma,
            "mid_transmittance_target": MID_TRANSMITTANCE,
            "frames": frame_count,
            "fps": 60,
            "supersample": supersample,
            "max_roll_deg": MAX_ROLL_DEG,
            "note": "output at half resolution (upscale dropped; see Wave 6 addendum)",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("flythrough_params.json", &json, "data")?;
        Ok(())
    }
}
