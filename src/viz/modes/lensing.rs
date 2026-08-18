//! V30 `lensing` -- Gravity Bends the Gallery.
//!
//! Re-observes the finished artwork through its own gravity: each body is a
//! point lens with an Einstein radius scaled from its mass; rays are
//! inverse-mapped through the lens equation so arcs curve around the masses
//! and multiple imaging emerges naturally. The still lenses the master at
//! the closest-approach configuration; the video lenses the re-accumulating
//! artwork with the lens configuration evolving per frame.

use crate::error::Result;
use crate::render::context::{PixelBuffer, RenderContext};
use crate::render::{ImageBuffer, SpectralRenderSettings, SpectralScene};
use crate::sim::Sha3RandomByteStream;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::{Accumulator, resized_config, scene_levels, stream_video};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use rayon::prelude::*;
use tracing::{info, warn};

/// Maximum Einstein radius as a fraction of the short edge.
const MAX_THETA_E_FRACTION: f64 = 0.045;
/// Supersampling factor per axis for the still.
const SUPERSAMPLE: usize = 2;
/// Jitter tile size (stratified jitter pattern, seeded per run).
const JITTER_TILE: usize = 64;
/// Video frames at 60 fps (30 s).
const VIDEO_FRAMES: usize = 1800;
/// Display gamma for decoding/encoding the master still.
const DISPLAY_GAMMA: f64 = 2.2;

/// The gravitational lensing mode.
pub struct Lensing;

/// One point lens: image-plane position and squared Einstein radius, in
/// pixels of the working canvas.
#[derive(Clone, Copy)]
struct Deflector {
    x: f64,
    y: f64,
    theta_e_sq: f64,
}

/// Point-lens deflection: `alpha = sum theta_Ei^2 (theta - theta_i) / |theta - theta_i|^2`.
fn deflection(deflectors: &[Deflector], x: f64, y: f64) -> (f64, f64) {
    let mut ax = 0.0;
    let mut ay = 0.0;
    for lens in deflectors {
        let dx = x - lens.x;
        let dy = y - lens.y;
        let dist_sq = (dx * dx + dy * dy).max(1e-9);
        ax += lens.theta_e_sq * dx / dist_sq;
        ay += lens.theta_e_sq * dy / dist_sq;
    }
    (ax, ay)
}

/// Einstein radii for the three bodies on a canvas with the given short
/// edge: `theta_E = k sqrt(m_i / sum m)`, k set so the largest is
/// `MAX_THETA_E_FRACTION` of the short edge.
fn einstein_radii(masses: [f64; 3], short_edge: f64) -> [f64; 3] {
    let total: f64 = masses.iter().sum::<f64>().max(1e-12);
    let largest = masses.iter().copied().fold(0.0f64, f64::max);
    let k = MAX_THETA_E_FRACTION * short_edge / (largest / total).sqrt().max(1e-12);
    [k * (masses[0] / total).sqrt(), k * (masses[1] / total).sqrt(), k * (masses[2] / total).sqrt()]
}

/// Stratified jitter tile in [-0.5, 0.5)^2, seeded from the mode RNG.
fn jitter_tile(rng: &mut Sha3RandomByteStream) -> Vec<(f32, f32)> {
    (0..JITTER_TILE * JITTER_TILE)
        .map(|_| ((rng.next_f64() - 0.5) as f32, (rng.next_f64() - 0.5) as f32))
        .collect()
}

/// Bilinear sample of an RGB f64 buffer; returns black outside the frame.
fn sample_rgb(
    source: &[(f64, f64, f64)],
    width: usize,
    height: usize,
    x: f64,
    y: f64,
) -> (f64, f64, f64) {
    if x < 0.0 || y < 0.0 || x > (width - 1) as f64 || y > (height - 1) as f64 {
        return (0.0, 0.0, 0.0);
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = x - x0 as f64;
    let ty = y - y0 as f64;
    let at = |px: usize, py: usize| source[py * width + px];
    let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
    let top =
        (a.0 * (1.0 - tx) + b.0 * tx, a.1 * (1.0 - tx) + b.1 * tx, a.2 * (1.0 - tx) + b.2 * tx);
    let bottom =
        (c.0 * (1.0 - tx) + d.0 * tx, c.1 * (1.0 - tx) + d.1 * tx, c.2 * (1.0 - tx) + d.2 * tx);
    (
        top.0 * (1.0 - ty) + bottom.0 * ty,
        top.1 * (1.0 - ty) + bottom.1 * ty,
        top.2 * (1.0 - ty) + bottom.2 * ty,
    )
}

/// Inverse-map lensing of a linear RGB buffer (supersampled + jittered).
fn lens_rgb(
    source: &[(f64, f64, f64)],
    width: usize,
    height: usize,
    deflectors: &[Deflector],
    jitter: &[(f32, f32)],
    supersample: usize,
) -> Vec<(f64, f64, f64)> {
    let mut out = vec![(0.0, 0.0, 0.0); width * height];
    let samples = (supersample * supersample) as f64;
    out.par_chunks_mut(width).enumerate().for_each(|(row, line)| {
        for (col, pixel) in line.iter_mut().enumerate() {
            let mut sum = (0.0, 0.0, 0.0);
            for sub_y in 0..supersample {
                for sub_x in 0..supersample {
                    let tile_index = (row % JITTER_TILE) * JITTER_TILE + col % JITTER_TILE;
                    let (jx, jy) = jitter[(tile_index + sub_y * 7 + sub_x * 13) % jitter.len()];
                    let x = col as f64 + (sub_x as f64 + 0.5 + f64::from(jx)) / supersample as f64;
                    let y = row as f64 + (sub_y as f64 + 0.5 + f64::from(jy)) / supersample as f64;
                    let (ax, ay) = deflection(deflectors, x, y);
                    let sampled = sample_rgb(source, width, height, x - ax - 0.5, y - ay - 0.5);
                    sum.0 += sampled.0;
                    sum.1 += sampled.1;
                    sum.2 += sampled.2;
                }
            }
            *pixel = (sum.0 / samples, sum.1 / samples, sum.2 / samples);
        }
    });
    out
}

impl VizMode for Lensing {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("lensing").expect("lensing is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps == 0 {
            warn!("lensing skipped: empty trajectory");
            return Ok(());
        }
        let masses = ctx.kinematics().masses;
        let closest = ctx.events().closest_triple;
        let mut rng = ctx.fork_rng(self.entry().flag);
        let jitter = jitter_tile(&mut rng);

        // --- Still: lens the finished master at the closest approach.
        let mut flux_audit: Option<(f64, f64)> = None;
        let master_path = format!("{}/images/source/master.png", ctx.seed_dir);
        match image::ImageReader::open(&master_path).map(image::ImageReader::decode) {
            Ok(Ok(decoded)) => {
                let master = decoded.into_rgb16();
                let width = master.width() as usize;
                let height = master.height() as usize;
                let source: Vec<(f64, f64, f64)> = master
                    .pixels()
                    .map(|pixel| {
                        let decode = |value: u16| (f64::from(value) / 65535.0).powf(DISPLAY_GAMMA);
                        (decode(pixel.0[0]), decode(pixel.0[1]), decode(pixel.0[2]))
                    })
                    .collect();

                let full_ctx = RenderContext::new(
                    master.width(),
                    master.height(),
                    ctx.positions,
                    ctx.settings.aspect_correction,
                );
                let radii = einstein_radii(masses, f64::from(master.width().min(master.height())));
                let deflectors: Vec<Deflector> = (0..3)
                    .map(|body| {
                        let position = ctx.positions[body][closest];
                        let (px, py) = full_ctx.to_pixel(position.x, position.y);
                        Deflector {
                            x: f64::from(px),
                            y: f64::from(py),
                            theta_e_sq: radii[body] * radii[body],
                        }
                    })
                    .collect();

                let lensed = lens_rgb(&source, width, height, &deflectors, &jitter, SUPERSAMPLE);
                let flux_in: f64 = source.iter().map(|&(r, g, b)| r + g + b).sum();
                let flux_out: f64 = lensed.iter().map(|&(r, g, b)| r + g + b).sum();
                flux_audit = Some((flux_in, flux_out));
                let ratio = flux_out / flux_in.max(1e-12);
                info!("   lensing: flux ratio {ratio:.4} (magnification-conserving target 1.0)");
                if !(0.98..=1.02).contains(&ratio) {
                    warn!("lensing flux conservation outside 2%: ratio {ratio:.4}");
                }

                let mut quantized = vec![0u16; width * height * 3];
                for (chunk, &(r, g, b)) in quantized.chunks_mut(3).zip(lensed.iter()) {
                    let encode = |value: f64| {
                        (value.clamp(0.0, 1.0).powf(1.0 / DISPLAY_GAMMA) * 65535.0).round() as u16
                    };
                    chunk[0] = encode(r);
                    chunk[1] = encode(g);
                    chunk[2] = encode(b);
                }
                let image = ImageBuffer::from_raw(master.width(), master.height(), quantized)
                    .expect("lensed buffer has width*height*3 samples");
                sink.save_png16(&image, "lensed.png")?;
            }
            Ok(Err(error)) => warn!("lensing: master decode failed ({error}); no still"),
            Err(error) => warn!("lensing: master missing ({error}); no still"),
        }

        // --- Video: lens the re-accumulating artwork per frame.
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16);
        let frame_count = ctx.quality.scale_count(VIDEO_FRAMES);

        let mut source_acc = Accumulator::new(
            ctx.positions.to_vec(),
            ctx.colors.to_vec(),
            ctx.body_alphas.to_vec(),
            video_w,
            video_h,
            ctx.settings.aspect_correction,
            ctx.settings.traits,
            ctx.settings.render_config.hdr_scale,
        );
        let resized = resized_config(ctx.settings.resolved_config, video_w, video_h);
        let render_config = *ctx.settings.render_config;
        let video_settings =
            SpectralRenderSettings::new(&resized, &render_config, ctx.settings.aspect_correction)
                .with_traits(ctx.settings.traits);
        let scene = SpectralScene::new(ctx.positions, ctx.colors, ctx.body_alphas);
        let levels = scene_levels(scene, video_settings, 1.0);
        let video_radii = einstein_radii(masses, f64::from(video_w.min(video_h)));

        // Half-resolution deflection grid, bilinearly upsampled per pixel.
        let grid_w = (video_w as usize / 2).max(8);
        let grid_h = (video_h as usize / 2).max(8);
        let grid_scale_x = grid_w as f64 / f64::from(video_w);
        let grid_scale_y = grid_h as f64 / f64::from(video_h);
        let mut alpha_x = vec![0.0f32; grid_w * grid_h];
        let mut alpha_y = vec![0.0f32; grid_w * grid_h];

        let mut source_rgba: PixelBuffer = Vec::new();
        let mut cursor = 0usize;
        stream_video(
            video_w,
            video_h,
            60,
            &sink.path("lensing.mp4"),
            &sink.path("lensing_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            &levels,
            |frame, rgba| {
                let target = ((frame + 1) * steps / frame_count).min(steps);
                source_acc.accumulate(cursor..target);
                cursor = target;
                source_acc.convert_into(&mut source_rgba);

                // Lens configuration at this frame's cursor step.
                let step = cursor.min(steps - 1);
                let deflectors: Vec<Deflector> = (0..3)
                    .map(|body| {
                        let position = ctx.positions[body][step];
                        let (px, py) = source_acc.render_ctx().to_pixel(position.x, position.y);
                        Deflector {
                            x: f64::from(px),
                            y: f64::from(py),
                            theta_e_sq: video_radii[body] * video_radii[body],
                        }
                    })
                    .collect();

                // Fill the half-res deflection grid (smooth field).
                alpha_x
                    .par_chunks_mut(grid_w)
                    .zip(alpha_y.par_chunks_mut(grid_w))
                    .enumerate()
                    .for_each(|(row, (ax_row, ay_row))| {
                        let y = (row as f64 + 0.5) / grid_scale_y;
                        for col in 0..grid_w {
                            let x = (col as f64 + 0.5) / grid_scale_x;
                            let (ax, ay) = deflection(&deflectors, x, y);
                            ax_row[col] = ax as f32;
                            ay_row[col] = ay as f32;
                        }
                    });

                let width = video_w as usize;
                let height = video_h as usize;
                rgba.resize(width * height, (0.0, 0.0, 0.0, 0.0));
                let source_ref = &source_rgba;
                let alpha_x_ref = &alpha_x;
                let alpha_y_ref = &alpha_y;
                let jitter_ref = &jitter;
                rgba.par_chunks_mut(width).enumerate().for_each(|(row, line)| {
                    let sample_grid = |field: &[f32], x: f64, y: f64| -> f64 {
                        let gx = (x * grid_scale_x - 0.5).clamp(0.0, (grid_w - 1) as f64);
                        let gy = (y * grid_scale_y - 0.5).clamp(0.0, (grid_h - 1) as f64);
                        let x0 = gx.floor() as usize;
                        let y0 = gy.floor() as usize;
                        let x1 = (x0 + 1).min(grid_w - 1);
                        let y1 = (y0 + 1).min(grid_h - 1);
                        let tx = gx - x0 as f64;
                        let ty = gy - y0 as f64;
                        let top = f64::from(field[y0 * grid_w + x0]) * (1.0 - tx)
                            + f64::from(field[y0 * grid_w + x1]) * tx;
                        let bottom = f64::from(field[y1 * grid_w + x0]) * (1.0 - tx)
                            + f64::from(field[y1 * grid_w + x1]) * tx;
                        top * (1.0 - ty) + bottom * ty
                    };
                    for (col, pixel) in line.iter_mut().enumerate() {
                        let tile_index = (row % JITTER_TILE) * JITTER_TILE + col % JITTER_TILE;
                        let (jx, jy) = jitter_ref[tile_index];
                        let x = col as f64 + 0.5 + f64::from(jx);
                        let y = row as f64 + 0.5 + f64::from(jy);
                        let ax = sample_grid(alpha_x_ref, x, y);
                        let ay = sample_grid(alpha_y_ref, x, y);
                        let sx = x - ax - 0.5;
                        let sy = y - ay - 0.5;
                        *pixel = if sx < 0.0
                            || sy < 0.0
                            || sx > (width - 1) as f64
                            || sy > (height - 1) as f64
                        {
                            (0.0, 0.0, 0.0, 0.0)
                        } else {
                            let x0 = sx.floor() as usize;
                            let y0 = sy.floor() as usize;
                            let x1 = (x0 + 1).min(width - 1);
                            let y1 = (y0 + 1).min(height - 1);
                            let tx = sx - x0 as f64;
                            let ty = sy - y0 as f64;
                            let at = |px: usize, py: usize| source_ref[py * width + px];
                            let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
                            let mix = |u: f64, v: f64, w: f64, z: f64| {
                                (u * (1.0 - tx) + v * tx) * (1.0 - ty)
                                    + (w * (1.0 - tx) + z * tx) * ty
                            };
                            (
                                mix(a.0, b.0, c.0, d.0),
                                mix(a.1, b.1, c.1, d.1),
                                mix(a.2, b.2, c.2, d.2),
                                mix(a.3, b.3, c.3, d.3),
                            )
                        };
                    }
                });
            },
        )?;
        sink.record("lensing.mp4", "video");
        sink.record("lensing_hq.mp4", "video");

        let radii_full = einstein_radii(masses, f64::from(ctx.width.min(ctx.height)));
        let meta = serde_json::json!({
            "closest_triple_step": closest,
            "theta_e_px_full_res": radii_full,
            "max_theta_e_fraction": MAX_THETA_E_FRACTION,
            "supersample": SUPERSAMPLE,
            "flux_in": flux_audit.map(|(input, _)| input),
            "flux_out": flux_audit.map(|(_, output)| output),
            "flux_ratio": flux_audit.map(|(input, output)| output / input.max(1e-12)),
            "jitter": "stratified 64x64 tile, seeded viz/lensing/v1",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("lens_params.json", &json, "data")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// QA math gate: total flux through the lens map stays within 2% for a
    /// compact source a few Einstein radii from the lens (point-lens total
    /// magnification approaches 1 away from the caustic).
    #[test]
    fn lens_flux_is_conserved_within_two_percent() {
        let size = 128usize;
        let mut source = vec![(0.0, 0.0, 0.0); size * size];
        let (cx, cy, sigma) = (56.0f64, 64.0f64, 4.0f64);
        for row in 0..size {
            for col in 0..size {
                let dist_sq =
                    ((col as f64 - cx).powi(2) + (row as f64 - cy).powi(2)) / (2.0 * sigma * sigma);
                let value = (-dist_sq).exp();
                source[row * size + col] = (value, value * 0.7, value * 0.4);
            }
        }
        // Lens 20 px from the blob center with theta_E = 3 px (u ~ 6.7):
        // far enough from the caustic that total magnification ~ 1.
        let deflectors = [Deflector { x: cx + 20.0, y: cy, theta_e_sq: 9.0 }];
        let jitter = vec![(0.0f32, 0.0f32); JITTER_TILE * JITTER_TILE];
        let lensed = lens_rgb(&source, size, size, &deflectors, &jitter, 2);

        let flux_in: f64 = source.iter().map(|&(r, g, b)| r + g + b).sum();
        let flux_out: f64 = lensed.iter().map(|&(r, g, b)| r + g + b).sum();
        let ratio = flux_out / flux_in;
        assert!(
            (ratio - 1.0).abs() < 0.02,
            "lens flux ratio {ratio:.4} should stay within 2% of unity"
        );
    }

    #[test]
    fn einstein_radii_scale_with_mass_and_cap_at_fraction() {
        let radii = einstein_radii([100.0, 200.0, 300.0], 1000.0);
        assert!(radii[0] < radii[1] && radii[1] < radii[2]);
        assert!((radii[2] - 45.0).abs() < 1e-9, "largest radius caps at 4.5% of short edge");
    }
}
