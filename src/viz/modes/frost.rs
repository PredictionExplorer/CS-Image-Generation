//! V34 `frost` -- Winter Claims the Window.
//!
//! Diffusion-limited aggregation: frost ferns nucleate on the brightest
//! trail pixels and grow outward, denser where energy is higher, with a
//! six-fold anisotropy combed along the local stroke tangent. The artwork
//! crystallizes over 30 seconds into a winter-glass version of itself; the
//! still reads as macro photography of ice on the finished master.

use crate::error::Result;
use crate::oklab::{
    linear_rec2020_to_oklab, oklab_to_linear_rec2020, oklab_to_oklch, oklch_to_oklab,
};
use crate::render::context::RenderContext;
use crate::render::{
    ImageBuffer, VideoEncodingOptions, VideoOutputSpec, create_videos_from_frames_singlepass,
};
use crate::spectrum::linear_rec2020_to_display_p3;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::agents::{DlaGrid, DlaParams};
use crate::viz::common::raster::{blur_field, splat_line_additive, upsample_bicubic};
use crate::viz::context::VizContext;
use crate::viz::modes::physarum::energy_density_grid;
use crate::viz::sink::ArtifactSink;
use rayon::prelude::*;
use tracing::{info, warn};

/// Total DLA walker budget at final quality (draft scales to a quarter).
const WALKER_BUDGET: usize = 1_200_000;
/// Nucleation percentile (top 0.5% of energy).
const NUCLEATION_QUANTILE: f64 = 0.995;
/// Minimum nucleation spacing at the reference half-res short edge (1117).
const NUCLEATION_SPACING_REF: f64 = 24.0;
/// Base sticking probability.
const STICK_BASE: f32 = 0.35;
/// Energy-hungry sticking span.
const STICK_SPAN: f32 = 0.65;
/// Six-fold comb bias amplitude.
const BIAS_STRENGTH: f32 = 0.8;
/// Refraction offset in pixels at full resolution.
const REFRACTION_PX: f64 = 1.5;
/// Cold hue shift under ice (degrees).
const HUE_SHIFT_DEG: f64 = -8.0;
/// Video frames (30 s at 30 fps).
const VIDEO_FRAMES: usize = 900;
/// Display gamma for master decode / frame encode.
const DISPLAY_GAMMA: f64 = 2.2;

/// The frost mode.
pub struct Frost;

/// Trajectory tangent orientation per grid cell (doubled-angle average, so
/// opposite directions reinforce rather than cancel).
fn tangent_field(ctx: &VizContext<'_>, grid_w: usize, grid_h: usize) -> Vec<f32> {
    let render_ctx = RenderContext::new(
        grid_w as u32,
        grid_h as u32,
        ctx.positions,
        ctx.settings.aspect_correction,
    );
    let mut weight = vec![0.0f32; grid_w * grid_h];
    let mut sin2 = vec![0.0f32; grid_w * grid_h];
    let mut cos_weight = vec![0.0f32; grid_w * grid_h];
    let mut cos2 = vec![0.0f32; grid_w * grid_h];
    let steps = ctx.step_count();
    for step in (0..steps.saturating_sub(1)).step_by(4) {
        for body in 0..3 {
            let a = ctx.positions[body][step];
            let b = ctx.positions[body][step + 1];
            let (ax, ay) = render_ctx.to_pixel(a.x, a.y);
            let (bx, by) = render_ctx.to_pixel(b.x, b.y);
            let angle = f64::from(by - ay).atan2(f64::from(bx - ax));
            let (s2, c2) = (2.0 * angle).sin_cos();
            splat_line_additive(
                &mut weight,
                &mut sin2,
                grid_w,
                grid_h,
                (ax, ay),
                (bx, by),
                s2 as f32,
                s2 as f32,
                2.0,
            );
            splat_line_additive(
                &mut cos_weight,
                &mut cos2,
                grid_w,
                grid_h,
                (ax, ay),
                (bx, by),
                c2 as f32,
                c2 as f32,
                2.0,
            );
        }
    }
    let mut orientation = vec![0.0f32; grid_w * grid_h];
    orientation
        .par_iter_mut()
        .zip(weight.par_iter().zip(sin2.par_iter().zip(cos2.par_iter())))
        .for_each(|(out, (&w, (&s, &c)))| {
            *out = if w > 1e-4 { 0.5 * f64::from(s).atan2(f64::from(c)) as f32 } else { 0.0 };
        });
    orientation
}

impl VizMode for Frost {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("frost").expect("frost is in the catalog")
    }

    fn needs_energy_field(&self) -> bool {
        true
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        if ctx.step_count() == 0 {
            warn!("frost skipped: empty trajectory");
            return Ok(());
        }
        let grid_w = (((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16)) as usize;
        let grid_h = (((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16)) as usize;
        let relative = f64::from(grid_w.min(grid_h) as u32) / 1117.0;
        let budget = ctx.quality.scale_count(WALKER_BUDGET);

        let energy = energy_density_grid(ctx, grid_w, grid_h);

        // Nucleation: brightest cells, Poisson-thinned.
        let spacing = (NUCLEATION_SPACING_REF * relative).max(8.0);
        let mut candidates: Vec<(usize, f32)> =
            energy.iter().copied().enumerate().filter(|&(_, value)| value > 0.0).collect();
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        let keep = ((candidates.len() as f64) * (1.0 - NUCLEATION_QUANTILE)) as usize;
        let mut nuclei: Vec<(usize, usize)> = Vec::new();
        for &(index, _) in candidates.iter().take(keep.max(24)) {
            let col = index % grid_w;
            let row = index / grid_w;
            let far_enough = nuclei.iter().all(|&(nc, nr)| {
                let dx = col as f64 - nc as f64;
                let dy = row as f64 - nr as f64;
                dx * dx + dy * dy >= spacing * spacing
            });
            if far_enough {
                nuclei.push((col, row));
            }
        }
        if nuclei.is_empty() {
            nuclei.push((grid_w / 2, grid_h / 2));
        }
        info!("   frost: {} nucleation sites, {budget} walkers", nuclei.len());

        // Grow the aggregate.
        let orientation = tangent_field(ctx, grid_w, grid_h);
        let mut rng = ctx.fork_rng(self.entry().flag);
        let mut grid = DlaGrid::new(grid_w, grid_h, &nuclei);
        grid.grow(
            budget,
            &energy,
            &orientation,
            &DlaParams {
                stick_base: STICK_BASE,
                stick_span: STICK_SPAN,
                bias_strength: BIAS_STRENGTH,
                batch_steps: 96,
                concurrent: 4096,
            },
            &mut rng,
        );
        let stuck = grid.stuck_count();
        info!("   frost: {stuck} crystals grown");

        // Crystal thickness by age: early growth is thick, tips taper.
        let max_age = f64::from(stuck.max(1));
        let thickness: Vec<f32> = grid
            .age
            .par_iter()
            .map(|&age| {
                if age == 0 {
                    0.0
                } else {
                    (1.0 - f64::from(age) / max_age).powf(0.7) as f32 * 0.85 + 0.15
                }
            })
            .collect();

        // Reveal schedule: each crystal appears at its age fraction.
        let frame_count = ctx.quality.scale_count(VIDEO_FRAMES);
        let reveal: Vec<u32> = grid
            .age
            .par_iter()
            .map(|&age| {
                if age == 0 {
                    u32::MAX
                } else {
                    ((f64::from(age) / max_age) * frame_count as f64) as u32
                }
            })
            .collect();

        // Master ghost at grid and still resolutions (display-linear).
        let load_master = |width: usize, height: usize| -> Vec<(f64, f64, f64)> {
            let path = format!("{}/images/source/master.png", ctx.seed_dir);
            let mut base = vec![(0.0, 0.0, 0.0); width * height];
            match image::ImageReader::open(&path).map(image::ImageReader::decode) {
                Ok(Ok(decoded)) => {
                    let master = decoded.into_rgb16();
                    let source_w = master.width() as usize;
                    let source_h = master.height() as usize;
                    let raw = master.as_raw();
                    for (row, line) in base.chunks_mut(width).enumerate() {
                        let sy = ((row as f64 + 0.5) * source_h as f64 / height as f64) as usize;
                        let sy = sy.min(source_h - 1);
                        for (col, slot) in line.iter_mut().enumerate() {
                            let sx = ((col as f64 + 0.5) * source_w as f64 / width as f64) as usize;
                            let sx = sx.min(source_w - 1);
                            let at = (sy * source_w + sx) * 3;
                            let decode = |v: u16| (f64::from(v) / 65535.0).powf(DISPLAY_GAMMA);
                            *slot = (decode(raw[at]), decode(raw[at + 1]), decode(raw[at + 2]));
                        }
                    }
                }
                Ok(Err(error)) => warn!("frost: master decode failed ({error}); black glass"),
                Err(error) => warn!("frost: master missing ({error}); black glass"),
            }
            base
        };

        // Shared ice compositor: height field + master -> display u16 frame.
        let compose = |height_field: &[f32],
                       master: &[(f64, f64, f64)],
                       width: usize,
                       height: usize,
                       refraction_px: f64,
                       out: &mut Vec<u16>| {
            out.resize(width * height * 3, 0);
            out.par_chunks_mut(width * 3).enumerate().for_each(|(row, line)| {
                for col in 0..width {
                    let index = row * width + col;
                    let at = |c: i64, r: i64| -> f64 {
                        let c = c.clamp(0, width as i64 - 1) as usize;
                        let r = r.clamp(0, height as i64 - 1) as usize;
                        f64::from(height_field[r * width + c])
                    };
                    let gx =
                        (at(col as i64 + 1, row as i64) - at(col as i64 - 1, row as i64)) * 0.5;
                    let gy =
                        (at(col as i64, row as i64 + 1) - at(col as i64, row as i64 - 1)) * 0.5;
                    let ice = f64::from(height_field[index]);

                    // Refracted, cold-shifted master beneath the ice.
                    let coverage = (ice * 3.0).clamp(0.0, 1.0);
                    let offset = refraction_px * coverage;
                    let sample_col =
                        ((col as f64 - gx * offset * 24.0).clamp(0.0, width as f64 - 1.0)) as usize;
                    let sample_row = ((row as f64 - gy * offset * 24.0)
                        .clamp(0.0, height as f64 - 1.0))
                        as usize;
                    let mut base = master[sample_row * width + sample_col];
                    if coverage > 0.0 {
                        let (l, a, b) = linear_rec2020_to_oklab(base.0, base.1, base.2);
                        let (lightness, chroma, hue) = oklab_to_oklch(l, a, b);
                        let (l2, a2, b2) =
                            oklch_to_oklab(lightness, chroma, hue + HUE_SHIFT_DEG * coverage);
                        let shifted = oklab_to_linear_rec2020(l2, a2, b2);
                        base = (
                            base.0 * (1.0 - coverage) + shifted.0.max(0.0) * coverage,
                            base.1 * (1.0 - coverage) + shifted.1.max(0.0) * coverage,
                            base.2 * (1.0 - coverage) + shifted.2.max(0.0) * coverage,
                        );
                    }

                    // Rim-lit crystal shading (light from upper left).
                    let normal_z = 1.0 / (1.0 + (gx * gx + gy * gy) * 900.0).sqrt();
                    let rim = (1.0 - normal_z).powf(0.8);
                    let light = (-gx * 0.6 - gy * 0.75).max(0.0);
                    let sparkle = ice * (0.05 + 0.9 * rim + 0.55 * light);
                    let icy = (0.82, 0.90, 1.0);
                    let color = (
                        base.0 + icy.0 * sparkle,
                        base.1 + icy.1 * sparkle,
                        base.2 + icy.2 * sparkle,
                    );
                    let (p3_r, p3_g, p3_b) =
                        linear_rec2020_to_display_p3(color.0, color.1, color.2);
                    let encode = |v: f64| {
                        (v.clamp(0.0, 1.0).powf(1.0 / DISPLAY_GAMMA) * 65535.0).round() as u16
                    };
                    let out_index = col * 3;
                    line[out_index] = encode(p3_r);
                    line[out_index + 1] = encode(p3_g);
                    line[out_index + 2] = encode(p3_b);
                }
            });
        };

        // --- Video: age-threshold reveal at grid resolution.
        let master_grid = load_master(grid_w, grid_h);
        let mut masked = vec![0.0f32; grid_w * grid_h];
        let mut frame_bytes: Vec<u16> = Vec::new();
        let outputs = [
            VideoOutputSpec {
                output_file: sink.path("frost.mp4"),
                options: VideoEncodingOptions::web_compatible(),
            },
            VideoOutputSpec {
                output_file: sink.path("frost_hq.mp4"),
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
            30,
            |out| {
                for frame in 0..frame_count {
                    masked
                        .par_iter_mut()
                        .zip(thickness.par_iter().zip(reveal.par_iter()))
                        .for_each(|(slot, (&thick, &when))| {
                            *slot = if when <= frame as u32 { thick } else { 0.0 };
                        });
                    blur_field(&mut masked, grid_w, grid_h, 1.2);
                    compose(
                        &masked,
                        &master_grid,
                        grid_w,
                        grid_h,
                        REFRACTION_PX * relative,
                        &mut frame_bytes,
                    );
                    out.write_all(bytemuck::cast_slice(&frame_bytes))
                        .map_err(crate::render::error::RenderError::VideoEncoding)?;
                }
                Ok(())
            },
            &outputs,
        )?;
        sink.record("frost.mp4", "video");
        sink.record("frost_hq.mp4", "video");

        // --- Still: fully grown ice at full resolution.
        let still_w = ctx.quality.scale_dim(ctx.width) as usize;
        let still_h = ctx.quality.scale_dim(ctx.height) as usize;
        let mut grown: Vec<f32> = thickness.clone();
        blur_field(&mut grown, grid_w, grid_h, 1.2);
        let full_height = upsample_bicubic(&grown, grid_w, grid_h, still_w, still_h);
        let master_full = load_master(still_w, still_h);
        let mut still_bytes: Vec<u16> = Vec::new();
        compose(
            &full_height,
            &master_full,
            still_w,
            still_h,
            REFRACTION_PX * (still_w.min(still_h) as f64 / 2234.0),
            &mut still_bytes,
        );
        let image = ImageBuffer::from_raw(still_w as u32, still_h as u32, still_bytes)
            .expect("frost still buffer has width*height*3 samples");
        sink.save_png16(&image, "frost.png")?;

        let meta = serde_json::json!({
            "walkers": budget,
            "crystals": stuck,
            "nuclei": nuclei.len(),
            "nucleation_spacing_px": spacing,
            "stick_base": STICK_BASE,
            "stick_span": STICK_SPAN,
            "bias_strength": BIAS_STRENGTH,
            "hue_shift_deg": HUE_SHIFT_DEG,
            "grid": [grid_w, grid_h],
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("frost_params.json", &json, "data")?;
        Ok(())
    }
}
