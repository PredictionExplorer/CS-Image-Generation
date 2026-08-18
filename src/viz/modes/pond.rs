//! V69 `pond` -- The Surface of a Dark Pond.
//!
//! The full water-world reading, one integrated surface: unseen bodies drag
//! marbling dye wakes (V36's fluid) while emitting Doppler-compressed
//! ripples (V32's wave field); beneath everything the GW strain drives the
//! box's (2,3)+(3,2) standing-wave eigenmodes, so cymatic roses bloom
//! framewide exactly when the strain spikes. The height field shades as
//! 2.5D water under a single palette-anchored moonlight -- specular glints
//! on crests, dye as a sub-surface glaze -- and the deepest periapsis
//! swells the cymatics x4 for three seconds; the still is captured at the
//! crescendo's maximum-coherence frame (peak projection onto the driven
//! eigenmode). Audio: a hydrophone-treated bed with the strain felt as sub.

use crate::error::Result;
use crate::oklab::{oklab_to_linear_rec2020, oklab_to_oklch, oklch_to_oklab};
use crate::render::context::RenderContext;
use crate::render::{VideoEncodingOptions, VideoOutputSpec, create_videos_from_frames_singlepass};
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::audio::{
    self, SAMPLE_RATE, fade_ends, mux, normalize_to_lufs, saw_bandlimited, soft_limit,
    write_wav_stereo_24bit,
};
use crate::viz::common::display::encode_linear_rec2020_to_u16;
use crate::viz::common::fluid::Fluid;
use crate::viz::common::raster::Rgb64;
use crate::viz::common::wave::{Oscillator, WaveField};
use crate::viz::context::VizContext;
use crate::viz::modes::gw_chirp::strain_series;
use crate::viz::sink::ArtifactSink;
use rayon::prelude::*;
use tracing::{info, warn};

/// Film length in seconds at 30 fps.
const POND_SECONDS: f64 = 45.0;
/// Frame rate.
const FPS: u32 = 30;
/// Driven box eigenmodes (m, n) pairs.
const EIGENMODES: [(f64, f64); 2] = [(2.0, 3.0), (3.0, 2.0)];
/// Crescendo amplification at the deepest periapsis.
const CRESCENDO_GAIN: f64 = 4.0;
/// Crescendo length in seconds.
const CRESCENDO_SECONDS: f64 = 3.0;
/// Target maximum body Mach number (V32's subsonic convention).
const MAX_MACH: f32 = 0.7;
/// Field damping (~7 s visual persistence).
const DAMPING: f32 = 2.0 / 7.0;
/// Wave-to-dye coupling strength (px/s per unit surface slope).
const COUPLING: f32 = 26.0;
/// Deep-water base (`OKLab`).
const BASE_WATER: (f64, f64, f64) = (0.13, -0.004, -0.018);

/// The pond mode.
pub struct Pond;

/// The driven standing-wave pattern: the (2,3)+(3,2) box eigenmodes
/// (zero on all four walls).
pub(crate) fn eigenmode(x: f64, y: f64, width: f64, height: f64) -> f64 {
    let mut value = 0.0;
    for &(m, n) in &EIGENMODES {
        value += (m * std::f64::consts::PI * x / width).sin()
            * (n * std::f64::consts::PI * y / height).sin();
    }
    value
}

/// Coherence of a height field with the driven pattern: the normalized
/// projection onto the eigenmode (the crescendo-still selector).
pub(crate) fn eigenmode_coherence(field: &[f32], width: usize, height: usize) -> f64 {
    let mut projection = 0.0f64;
    let mut energy = 1e-12f64;
    for row in 0..height {
        for col in 0..width {
            let value = f64::from(field[row * width + col]);
            let mode = eigenmode(col as f64 + 0.5, row as f64 + 0.5, width as f64, height as f64);
            projection += value * mode;
            energy += value * value;
        }
    }
    projection.abs() / energy.sqrt()
}

impl VizMode for Pond {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("pond").expect("pond is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 1_000 {
            warn!("pond skipped: trajectory too short");
            return Ok(());
        }
        let grid_w = (((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16)) as usize;
        let grid_h = (((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16)) as usize;
        let frame_count = ctx.quality.scale_count((POND_SECONDS * f64::from(FPS)) as usize);
        let relative = grid_w.min(grid_h) as f32 / 1117.0;
        let render_ctx = RenderContext::new(
            grid_w as u32,
            grid_h as u32,
            ctx.positions,
            ctx.settings.aspect_correction,
        );

        // --- Per-frame body tracks (pixels) and speeds (px per second).
        let body_px: Vec<[(f32, f32); 3]> = (0..frame_count)
            .map(|frame| {
                let step = ((frame + 1) * steps / frame_count).min(steps - 1);
                std::array::from_fn(|body| {
                    let position = ctx.positions[body][step];
                    render_ctx.to_pixel(position.x, position.y)
                })
            })
            .collect();
        let mut speeds_px: Vec<[f32; 3]> = vec![[0.0; 3]; frame_count];
        for frame in 1..frame_count {
            for body in 0..3 {
                let (x0, y0) = body_px[frame - 1][body];
                let (x1, y1) = body_px[frame][body];
                speeds_px[frame][body] = (x1 - x0).hypot(y1 - y0) * f64::from(FPS) as f32;
            }
        }
        if frame_count > 1 {
            speeds_px[0] = speeds_px[1];
        }
        let mut pooled: Vec<f32> =
            speeds_px.iter().flat_map(|frame| frame.iter().copied()).collect();
        pooled.sort_by(f32::total_cmp);
        let p98 = pooled[((pooled.len() - 1) as f64 * 0.98) as usize];
        let wave_speed = (p98 / MAX_MACH).clamp(60.0 * relative, 1500.0 * relative);

        // --- The strain drive, resampled to frames and normalized.
        let (h_plus, _) = strain_series(ctx.positions, &ctx.kinematics().masses);
        let mut strain = audio::resample_linear(&h_plus, frame_count.max(2));
        {
            let mut sorted: Vec<f64> = strain.iter().map(|v| v.abs()).collect();
            sorted.sort_by(f64::total_cmp);
            let p99 = sorted[((sorted.len() - 1) as f64 * 0.99) as usize].max(1e-12);
            for value in &mut strain {
                *value = (*value / p99).clamp(-1.5, 1.5);
            }
        }
        // Crescendo window around the deepest periapsis.
        let crescendo_frame = ctx.events().closest_triple * frame_count / steps.max(1);
        let crescendo_half = (CRESCENDO_SECONDS * f64::from(FPS) / 2.0) as usize;
        let crescendo_gain = |frame: usize| -> f64 {
            let distance = frame.abs_diff(crescendo_frame) as f64;
            if distance > crescendo_half as f64 {
                1.0
            } else {
                // Raised-cosine swell to x4 at the center.
                let window =
                    0.5 + 0.5 * (std::f64::consts::PI * distance / crescendo_half as f64).cos();
                1.0 + (CRESCENDO_GAIN - 1.0) * window
            }
        };

        // Hue-locked oscillator frequencies (V32's convention).
        let mut frequencies = [0.0f32; 3];
        for (body, frequency) in frequencies.iter_mut().enumerate() {
            let (l, a, b) = ctx.mean_color(body);
            let (_, _, hue) = oklab_to_oklch(l, a, b);
            *frequency = 2.2 + 2.6 * (hue / 360.0) as f32;
        }
        let anchor_hue = {
            let (l, a, b) = ctx.mean_color(0);
            oklab_to_oklch(l, a, b).2
        };
        let moon = {
            let (l, a, b) = oklch_to_oklab(0.90, 0.03, anchor_hue);
            let rgb = oklab_to_linear_rec2020(l, a, b);
            (rgb.0.max(0.0), rgb.1.max(0.0), rgb.2.max(0.0))
        };
        let body_dye: [Rgb64; 3] = std::array::from_fn(|body| {
            let (l, a, b) = ctx.mean_color(body);
            let rgb = oklab_to_linear_rec2020((l + 0.12).min(0.9), a * 1.15, b * 1.15);
            (rgb.0.max(0.0), rgb.1.max(0.0), rgb.2.max(0.0))
        });
        info!(
            "   pond: {grid_w}x{grid_h} surface, c {wave_speed:.0} px/s, crescendo at frame \
             {crescendo_frame}"
        );

        // --- Simulation state.
        let mut waves = WaveField::new(grid_w, grid_h, wave_speed, DAMPING);
        let mut fluid = Fluid::new(grid_w, grid_h);
        let speed_window = ctx.kinematics().speed_window();
        let cells = grid_w * grid_h;
        let mut height_field = vec![0.0f32; cells];
        let mut force_u = vec![0.0f32; cells];
        let mut force_v = vec![0.0f32; cells];
        // Precompute the eigenmode pattern once.
        let pattern: Vec<f32> = (0..cells)
            .map(|index| {
                let col = index % grid_w;
                let row = index / grid_w;
                eigenmode(col as f64 + 0.5, row as f64 + 0.5, grid_w as f64, grid_h as f64) as f32
            })
            .collect();

        let dt_frame = 1.0f32 / f64::from(FPS) as f32;
        let mut best_still: (f64, usize, Vec<f32>, [Vec<f32>; 3]) =
            (f64::MIN, 0, Vec::new(), [Vec::new(), Vec::new(), Vec::new()]);
        let mut rms_track = 0.02f64;
        let started = std::time::Instant::now();
        let mut logged = false;

        let outputs = [
            VideoOutputSpec {
                output_file: sink.path("pond_silent.mp4"),
                options: VideoEncodingOptions::web_compatible(),
            },
            VideoOutputSpec {
                output_file: sink.path("pond_silent_hq.mp4"),
                options: if ctx.fast_encode {
                    VideoEncodingOptions::fast_encode()
                } else {
                    VideoEncodingOptions::high_quality()
                },
            },
        ];
        create_videos_from_frames_singlepass(
            grid_w as u32,
            grid_h as u32,
            FPS,
            |out| {
                let mut rgb: Vec<Rgb64> = vec![(0.0, 0.0, 0.0); cells];
                let mut bytes: Vec<u16> = Vec::new();
                for frame in 0..frame_count {
                    let step = ((frame + 1) * steps / frame_count).min(steps - 1);
                    // Ripples: moving oscillators, amplitude by speed.
                    let oscillators: Vec<Oscillator> = (0..3)
                        .map(|body| {
                            let (x, y) = body_px[frame][body];
                            let speed = ctx.kinematics().speeds[body][step];
                            let normalized = ctx.kinematics().normalized_speed(speed_window, speed);
                            Oscillator {
                                x,
                                y,
                                amplitude: (0.35 + 1.0 * normalized as f32).min(1.5),
                                frequency: frequencies[body],
                            }
                        })
                        .collect();
                    waves.advance_frame(dt_frame, &oscillators, frame as f32 * dt_frame);

                    // The integrated height: ripples + strain-driven roses.
                    let wave_field = waves.field();
                    let rms =
                        (wave_field.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>()
                            / cells as f64)
                            .sqrt();
                    rms_track = rms_track * 0.95 + rms * 0.05;
                    let cymatic = (rms_track
                        * 1.6
                        * strain[frame.min(strain.len() - 1)]
                        * crescendo_gain(frame)) as f32;
                    height_field
                        .par_iter_mut()
                        .zip(wave_field.par_iter())
                        .zip(pattern.par_iter())
                        .for_each(|((height, &wave), &mode)| {
                            *height = wave + cymatic * mode;
                        });

                    // One-way coupling: the surface gradient pushes the dye.
                    force_u.par_chunks_mut(grid_w).enumerate().for_each(|(row, line)| {
                        for (col, slot) in line.iter_mut().enumerate() {
                            let left = col.saturating_sub(1);
                            let right = (col + 1).min(grid_w - 1);
                            let gradient = (height_field[row * grid_w + right]
                                - height_field[row * grid_w + left])
                                * 0.5;
                            *slot = (-COUPLING * gradient * dt_frame).clamp(-3.0, 3.0);
                        }
                    });
                    force_v.par_chunks_mut(grid_w).enumerate().for_each(|(row, line)| {
                        let up = row.saturating_sub(1);
                        let down = (row + 1).min(grid_h - 1);
                        for (col, slot) in line.iter_mut().enumerate() {
                            let gradient = (height_field[down * grid_w + col]
                                - height_field[up * grid_w + col])
                                * 0.5;
                            *slot = (-COUPLING * gradient * dt_frame).clamp(-3.0, 3.0);
                        }
                    });
                    fluid.add_velocity_field(&force_u, &force_v);
                    // Wakes: dye + a gentle drag force at each body.
                    #[allow(clippy::needless_range_loop)]
                    for body in 0..3 {
                        let (x, y) = body_px[frame][body];
                        let (px, py) = body_px[frame.saturating_sub(1)][body];
                        let speed = ctx.kinematics().speeds[body][step];
                        let normalized = ctx.kinematics().normalized_speed(speed_window, speed);
                        fluid.inject_dye(
                            body,
                            x,
                            y,
                            2.4 * relative.max(0.6) + 2.0,
                            0.10 + 0.22 * normalized as f32,
                        );
                        fluid.add_force(
                            x,
                            y,
                            3.0 * relative.max(0.6) + 2.0,
                            (x - px) * 1.6,
                            (y - py) * 1.6,
                        );
                    }
                    fluid.step(dt_frame);

                    // Crescendo still: the maximum-coherence frame.
                    if frame.abs_diff(crescendo_frame) <= crescendo_half * 2 {
                        let coherence = eigenmode_coherence(&height_field, grid_w, grid_h);
                        if coherence > best_still.0 {
                            best_still = (
                                coherence,
                                frame,
                                height_field.clone(),
                                std::array::from_fn(|body| fluid.dye[body].clone()),
                            );
                        }
                    }

                    // --- Shade the surface (2.5D, one moon).
                    shade_surface(
                        &height_field,
                        &fluid.dye,
                        grid_w,
                        grid_h,
                        moon,
                        &body_dye,
                        &mut rgb,
                    );
                    encode_linear_rec2020_to_u16(&rgb, &mut bytes);
                    out.write_all(bytemuck::cast_slice(&bytes))
                        .map_err(crate::render::error::RenderError::VideoEncoding)?;
                    if frame == 0 && !logged {
                        logged = true;
                        let per_frame = started.elapsed().as_secs_f64();
                        info!(
                            "   pond: {per_frame:.2}s first frame, projected {:.1} min",
                            per_frame * frame_count as f64 / 60.0
                        );
                    }
                }
                Ok(())
            },
            &outputs,
        )?;

        // --- The crescendo still at full resolution (bicubic upsample of
        // the coherence-peak surface, re-shaded at full res).
        if !best_still.2.is_empty() {
            let full_w = ctx.quality.scale_dim(ctx.width) as usize;
            let full_h = ctx.quality.scale_dim(ctx.height) as usize;
            let up = |field: &[f32]| -> Vec<f32> {
                crate::viz::common::raster::upsample_bicubic(field, grid_w, grid_h, full_w, full_h)
            };
            let height_full = up(&best_still.2);
            let dye_full: [Vec<f32>; 3] = std::array::from_fn(|body| up(&best_still.3[body]));
            let mut rgb: Vec<Rgb64> = vec![(0.0, 0.0, 0.0); full_w * full_h];
            shade_surface(&height_full, &dye_full, full_w, full_h, moon, &body_dye, &mut rgb);
            let image = crate::viz::common::display::encode_linear_rec2020_png16(
                &rgb,
                full_w as u32,
                full_h as u32,
            );
            sink.save_png16(&image, "pond_still.png")?;
            info!(
                "   pond: crescendo still at frame {} (coherence {:.4})",
                best_still.1, best_still.0
            );
        }

        // --- Audio: hydrophone bed + the strain as felt sub.
        {
            let duration = frame_count as f64 / f64::from(FPS);
            let sample_rate = f64::from(SAMPLE_RATE);
            let total_samples = (duration * sample_rate) as usize;
            let mut left = vec![0.0f64; total_samples];
            let mut right = vec![0.0f64; total_samples];
            let sub = audio::resample_linear(&h_plus, total_samples);
            let sub_peak = sub.iter().fold(1e-12f64, |acc, &v| acc.max(v.abs()));
            let root = 55.0;
            let mut phase = (0.0f64, 0.0f64);
            let mut lowpass = audio::Biquad::default();
            lowpass.set_lowpass(320.0, 0.7, sample_rate);
            let mut sub_low = audio::Biquad::default();
            sub_low.set_lowpass(60.0, 0.7, sample_rate);
            for sample in 0..total_samples {
                let t = sample as f64 / sample_rate;
                phase.0 = (phase.0 + root / sample_rate) % 1.0;
                phase.1 = (phase.1 + root * 1.498 / sample_rate) % 1.0;
                // Hydrophone: slow drone through a dark lowpass, breathing.
                let breathe = 0.75 + 0.25 * (t * 0.21 * std::f64::consts::TAU).sin();
                let drone = lowpass.process(
                    saw_bandlimited(phase.0, root / sample_rate) * 0.6
                        + saw_bandlimited(phase.1, root * 1.498 / sample_rate) * 0.3,
                ) * 0.16
                    * breathe;
                let felt = sub_low.process(sub[sample] / sub_peak) * 0.22;
                left[sample] = drone + felt;
                right[sample] = drone * 0.96 + felt;
            }
            soft_limit(&mut left, &mut right, 1.2);
            normalize_to_lufs(&mut left, &mut right, -18.0, sample_rate);
            fade_ends(&mut left, (sample_rate * 0.5) as usize);
            fade_ends(&mut right, (sample_rate * 0.5) as usize);
            let mix_path = sink.path("pond_mix.wav");
            write_wav_stereo_24bit(&mix_path, &left, &right)?;
            for (silent, scored) in
                [("pond_silent.mp4", "pond.mp4"), ("pond_silent_hq.mp4", "pond_hq.mp4")]
            {
                let silent_path = sink.path(silent);
                match mux(&silent_path, &mix_path, &sink.path(scored)) {
                    Ok(()) => {
                        sink.record(scored, "video");
                        let _ = std::fs::remove_file(&silent_path);
                    }
                    Err(error) => {
                        warn!("pond mux failed for {scored}: {error}; keeping silent cut");
                        let _ = std::fs::rename(&silent_path, sink.path(scored));
                        sink.record(scored, "video");
                    }
                }
            }
            let _ = std::fs::remove_file(&mix_path);
        }

        let meta = serde_json::json!({
            "eigenmodes": [[2, 3], [3, 2]],
            "crescendo_frame": crescendo_frame,
            "crescendo_gain": CRESCENDO_GAIN,
            "crescendo_seconds": CRESCENDO_SECONDS,
            "still_frame": best_still.1,
            "still_coherence": best_still.0,
            "wave_speed_px_s": wave_speed,
            "coupling": COUPLING,
            "coherence_metric": "normalized projection onto the driven eigenmode",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("pond_params.json", &json, "data")?;
        Ok(())
    }
}

/// Normal-from-height water shading with one moonlight, crest speculars,
/// and the dye as a sub-surface glaze under the shading.
fn shade_surface(
    height: &[f32],
    dye: &[Vec<f32>; 3],
    width: usize,
    height_px: usize,
    moon: Rgb64,
    body_dye: &[Rgb64; 3],
    rgb: &mut [Rgb64],
) {
    // The moon sits high and slightly off-axis; normals from central
    // differences with a relief gain that keeps slopes readable.
    let light = {
        let (x, y, z) = (-0.32f64, -0.44, 0.84);
        let norm = (x * x + y * y + z * z).sqrt();
        (x / norm, y / norm, z / norm)
    };
    let relief = 9.0f64;
    let base = oklab_to_linear_rec2020(BASE_WATER.0, BASE_WATER.1, BASE_WATER.2);
    rgb.par_chunks_mut(width).enumerate().for_each(|(row, line)| {
        let up = row.saturating_sub(1);
        let down = (row + 1).min(height_px - 1);
        for (col, pixel) in line.iter_mut().enumerate() {
            let left = col.saturating_sub(1);
            let right = (col + 1).min(width - 1);
            let dhdx = f64::from(height[row * width + right] - height[row * width + left]) * 0.5;
            let dhdy = f64::from(height[down * width + col] - height[up * width + col]) * 0.5;
            let normal = {
                let (x, y, z) = (-dhdx * relief, -dhdy * relief, 1.0);
                let norm = (x * x + y * y + z * z).sqrt();
                (x / norm, y / norm, z / norm)
            };
            let diffuse = (normal.0 * light.0 + normal.1 * light.1 + normal.2 * light.2).max(0.0);
            // Blinn-style glint toward the viewer (0, 0, 1).
            let half = {
                let (x, y, z) = (light.0, light.1, light.2 + 1.0);
                let norm = (x * x + y * y + z * z).sqrt();
                (x / norm, y / norm, z / norm)
            };
            let glint =
                (normal.0 * half.0 + normal.1 * half.1 + normal.2 * half.2).max(0.0).powi(180);
            // Dye glaze: body inks summed, softly saturated.
            let index = row * width + col;
            let mut glaze = (0.0f64, 0.0f64, 0.0f64);
            for body in 0..3 {
                let amount = f64::from(dye[body][index]).min(2.2);
                let knee = amount / (1.0 + 0.55 * amount);
                glaze.0 += body_dye[body].0 * knee;
                glaze.1 += body_dye[body].1 * knee;
                glaze.2 += body_dye[body].2 * knee;
            }
            let shade = 0.28 + 0.72 * diffuse;
            *pixel = (
                (base.0.max(0.0) + glaze.0 * 0.32) * shade + moon.0 * glint * 0.9,
                (base.1.max(0.0) + glaze.1 * 0.32) * shade + moon.1 * glint * 0.9,
                (base.2.max(0.0) + glaze.2 * 0.32) * shade + moon.2 * glint * 0.9,
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigenmode_vanishes_on_the_walls() {
        let (width, height) = (200.0f64, 130.0f64);
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            assert!(eigenmode(0.0, t * height, width, height).abs() < 1e-9);
            assert!(eigenmode(width, t * height, width, height).abs() < 1e-9);
            assert!(eigenmode(t * width, 0.0, width, height).abs() < 1e-9);
            assert!(eigenmode(t * width, height, width, height).abs() < 1e-9);
        }
        // Interior structure exists.
        assert!(eigenmode(width * 0.23, height * 0.19, width, height).abs() > 1e-3);
    }

    #[test]
    fn coherence_selects_the_driven_pattern() {
        let (width, height) = (96usize, 64usize);
        // A field that IS the eigenmode scores far higher than noise.
        let rose: Vec<f32> = (0..width * height)
            .map(|index| {
                let col = index % width;
                let row = index / width;
                eigenmode(col as f64 + 0.5, row as f64 + 0.5, width as f64, height as f64) as f32
            })
            .collect();
        let noise: Vec<f32> = (0..width * height)
            .map(|index| ((index as f32 * 12.9898).sin() * 43758.5) % 1.0)
            .collect();
        let rose_score = eigenmode_coherence(&rose, width, height);
        let noise_score = eigenmode_coherence(&noise, width, height);
        assert!(
            rose_score > noise_score * 10.0,
            "rose {rose_score:.4} must dominate noise {noise_score:.4}"
        );
    }

    #[test]
    fn cymatic_amplitude_follows_the_strain_gate() {
        // The surface term is A * h(t) * pattern: zero strain, zero roses.
        let (width, height) = (32usize, 24usize);
        let pattern: Vec<f32> = (0..width * height)
            .map(|index| {
                let col = index % width;
                let row = index / width;
                eigenmode(col as f64 + 0.5, row as f64 + 0.5, width as f64, height as f64) as f32
            })
            .collect();
        let quiet: Vec<f32> = pattern.iter().map(|&mode| 0.0 * mode).collect();
        let loud: Vec<f32> = pattern.iter().map(|&mode| 1.0 * mode).collect();
        assert!(quiet.iter().all(|&v| v == 0.0));
        let correlation: f64 =
            loud.iter().zip(pattern.iter()).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum();
        assert!(correlation > 0.0, "roses must rise exactly with the strain");
    }
}
