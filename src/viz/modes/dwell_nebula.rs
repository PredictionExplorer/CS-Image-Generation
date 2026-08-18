//! V60 `dwell-nebula` -- The Ergodic Ghost.
//!
//! Where the bodies spent time, rendered as a soft nebula: velocity-weighted
//! occupancy densities (slow dwell = dense), blurred through a multi-scale
//! stack for astrophotographic depth, colored in the bodies' dimmest
//! registers. Emitted solo and as an energy-linear underlay composited
//! beneath the master conversion.

use crate::error::Result;
use crate::oklab::{oklab_to_linear_rec2020, oklab_to_oklch, oklch_to_oklab};
use crate::render::context::{PixelBuffer, RenderContext};
use crate::render::effects::convert_spd_buffer_to_rgba;
use crate::viz::VizMode;
use crate::viz::VizPhase;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::display::{encode_linear_rec2020_png16, grade_with_levels};
use crate::viz::common::raster::{Rgb64, blur_field};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use rayon::prelude::*;
use tracing::warn;

/// Blur stack sigmas at half resolution (full-res 9 / 34 / 110).
const BLUR_SIGMAS: [f64; 3] = [4.5, 17.0, 55.0];
/// Blend weights of the blur stack layers.
const BLUR_WEIGHTS: [f64; 3] = [1.0, 0.4, 0.15];
/// Chroma multiplier for nebula colors.
const CHROMA_SCALE: f64 = 0.35;
/// Lightness ceiling for nebula colors.
const LIGHTNESS_CAP: f64 = 0.42;
/// Underlay strength in the composite (energy-linear).
const COMPOSITE_STRENGTH: f64 = 0.14;

/// The dwell-nebula mode.
pub struct DwellNebula;

/// Velocity-weighted occupancy field for one body at half resolution.
fn occupancy_field(
    ctx: &VizContext<'_>,
    render_ctx: &RenderContext,
    body: usize,
    half_w: usize,
    half_h: usize,
) -> Vec<f32> {
    let mut field = vec![0.0f32; half_w * half_h];
    let speeds = &ctx.kinematics().speeds[body];
    let window = ctx.kinematics().speed_window();
    let epsilon = (window.0 * 0.5).max(1e-9);
    for (step, point) in ctx.positions[body].iter().enumerate() {
        let (px, py) = render_ctx.to_pixel(point.x, point.y);
        let x = f64::from(px) * 0.5;
        let y = f64::from(py) * 0.5;
        if x < 0.0 || y < 0.0 || x >= (half_w - 1) as f64 || y >= (half_h - 1) as f64 {
            continue;
        }
        let weight = (1.0 / speeds[step].max(epsilon)) as f32;
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let fx = (x - x0 as f64) as f32;
        let fy = (y - y0 as f64) as f32;
        field[y0 * half_w + x0] += weight * (1.0 - fx) * (1.0 - fy);
        field[y0 * half_w + x0 + 1] += weight * fx * (1.0 - fy);
        field[(y0 + 1) * half_w + x0] += weight * (1.0 - fx) * fy;
        field[(y0 + 1) * half_w + x0 + 1] += weight * fx * fy;
    }
    field
}

/// Multi-scale blur stack: sum of Gaussian-blurred copies, asinh-toned to 1.
fn nebula_tone(field: &[f32], half_w: usize, half_h: usize) -> Vec<f32> {
    let mut stack = vec![0.0f64; field.len()];
    for (sigma, weight) in BLUR_SIGMAS.iter().zip(BLUR_WEIGHTS.iter()) {
        let mut layer = field.to_vec();
        blur_field(&mut layer, half_w, half_h, *sigma);
        for (slot, &value) in stack.iter_mut().zip(layer.iter()) {
            *slot += f64::from(value) * weight;
        }
    }
    let max = stack.iter().copied().fold(1e-12f64, f64::max);
    let scale = max * 0.02;
    let norm = (max / scale).asinh();
    stack.iter().map(|&value| ((value / scale).asinh() / norm) as f32).collect()
}

impl VizMode for DwellNebula {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("dwell-nebula").expect("dwell-nebula is in the catalog")
    }

    fn phase(&self) -> VizPhase {
        VizPhase::Spd
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let Some(buffer) = ctx.accum_spd else {
            warn!("dwell-nebula skipped: SPD buffer unavailable");
            return Ok(());
        };
        let width = ctx.width as usize;
        let height = ctx.height as usize;
        let half_w = (width / 2).max(2);
        let half_h = (height / 2).max(2);
        let render_ctx = RenderContext::new(
            ctx.width,
            ctx.height,
            ctx.positions,
            ctx.settings.aspect_correction,
        );

        // Per-body toned fields and nebula colors.
        let toned: Vec<Vec<f32>> = (0..3)
            .map(|body| {
                let field = occupancy_field(ctx, &render_ctx, body, half_w, half_h);
                nebula_tone(&field, half_w, half_h)
            })
            .collect();
        let colors: Vec<(f64, f64)> = (0..3)
            .map(|body| {
                let (l, a, b) = ctx.mean_color(body);
                let (_, chroma, hue) = oklab_to_oklch(l, a, b);
                (chroma * CHROMA_SCALE, hue)
            })
            .collect();

        // Compose the half-res nebula in linear Rec.2020.
        let nebula_half: Vec<Rgb64> = (0..half_w * half_h)
            .into_par_iter()
            .map(|index| {
                let mut out = (0.0f64, 0.0f64, 0.0f64);
                for body in 0..3 {
                    let value = f64::from(toned[body][index]);
                    if value <= 1e-6 {
                        continue;
                    }
                    let lightness = (LIGHTNESS_CAP * value.powf(0.85)).min(LIGHTNESS_CAP);
                    let (l, a, b) = oklch_to_oklab(lightness, colors[body].0, colors[body].1);
                    let (r, g, bl) = oklab_to_linear_rec2020(l, a, b);
                    out.0 += r.max(0.0) * value;
                    out.1 += g.max(0.0) * value;
                    out.2 += bl.max(0.0) * value;
                }
                out
            })
            .collect();

        // Bilinear upsample to full resolution.
        let nebula_full: Vec<Rgb64> = (0..width * height)
            .into_par_iter()
            .map(|index| {
                let x = (index % width) as f64 * 0.5;
                let y = (index / width) as f64 * 0.5;
                let x0 = (x.floor() as usize).min(half_w - 2);
                let y0 = (y.floor() as usize).min(half_h - 2);
                let fx = x - x0 as f64;
                let fy = y - y0 as f64;
                let sample = |sx: usize, sy: usize| nebula_half[sy * half_w + sx];
                let mix = |a: Rgb64, b: Rgb64, t: f64| {
                    (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t, a.2 + (b.2 - a.2) * t)
                };
                let top = mix(sample(x0, y0), sample(x0 + 1, y0), fx);
                let bottom = mix(sample(x0, y0 + 1), sample(x0 + 1, y0 + 1), fx);
                mix(top, bottom, fy)
            })
            .collect();

        sink.save_png16(
            &encode_linear_rec2020_png16(&nebula_full, ctx.width, ctx.height),
            "nebula_solo.png",
        )?;

        // Composite: nebula added energy-linear beneath the master conversion.
        let mut master: PixelBuffer = vec![(0.0, 0.0, 0.0, 0.0); width * height];
        convert_spd_buffer_to_rgba(buffer, &mut master, width, height);
        master.par_iter_mut().zip(nebula_full.par_iter()).for_each(|(pixel, &(r, g, b))| {
            let luminance = (r + g + b) / 3.0;
            let alpha_boost = (COMPOSITE_STRENGTH * luminance.min(1.0)).clamp(0.0, 1.0);
            pixel.0 += r * COMPOSITE_STRENGTH;
            pixel.1 += g * COMPOSITE_STRENGTH;
            pixel.2 += b * COMPOSITE_STRENGTH;
            pixel.3 = 1.0 - (1.0 - pixel.3) * (1.0 - alpha_boost);
        });
        let composite = grade_with_levels(&master, ctx.width, ctx.height, ctx.levels);
        sink.save_png16(&composite, "nebula_composite.png")?;

        let meta = serde_json::json!({
            "blur_sigmas_half_res": BLUR_SIGMAS,
            "blur_weights": BLUR_WEIGHTS,
            "chroma_scale": CHROMA_SCALE,
            "lightness_cap": LIGHTNESS_CAP,
            "composite_strength": COMPOSITE_STRENGTH,
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("nebula.json", &json, "data")?;
        Ok(())
    }
}
