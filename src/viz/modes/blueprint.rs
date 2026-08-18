//! V59 `blueprint` -- Two Archival Restylings.
//!
//! The trajectory's energy field re-inked twice: a cyanotype blueprint
//! (ridge-filtered drafting lines on Prussian blue, with real measured
//! dimension callouts) and a glass-plate negative (inverted tone through an
//! H&D-style curve on cream, with seeded plate scratches, a corner emulsion
//! chip, and an italic catalog number). Restraint is the style.

use crate::error::Result;
use crate::render::context::RenderContext;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::display::encode_linear_rec2020_png16;
use crate::viz::common::raster::{blur_field, draw_line};
use crate::viz::common::style::{Paper, draw_footer, ink_color, margin, paper_color, type_px};
use crate::viz::common::text::{Face, TextStyle, draw_text};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use tracing::warn;

/// Ridge filter sigma in pixels (at field resolution).
const RIDGE_SIGMA: f64 = 1.8;
/// Dimension callouts placed at the deepest approaches.
const CALLOUTS: usize = 3;

/// H&D-style silver-gelatin tone curve (toe and shoulder).
pub(crate) fn hd_curve(density: f64) -> f64 {
    let x = density.clamp(0.0, 1.0);
    // Smootherstep-flavored S-curve: gentle toe, creamy shoulder.
    let s = x * x * (3.0 - 2.0 * x);
    0.08 + 0.9 * s.powf(1.15)
}

/// The blueprint mode.
pub struct Blueprint;

impl VizMode for Blueprint {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("blueprint").expect("blueprint is in the catalog")
    }

    fn needs_energy_field(&self) -> bool {
        true
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let Some(field) = ctx.energy_field() else {
            warn!("blueprint skipped: no energy field (image-only run)");
            return Ok(());
        };
        let source_w = ctx.width as usize;
        let source_h = ctx.height as usize;
        let out_w = ctx.quality.scale_dim(ctx.width) as usize;
        let out_h = ctx.quality.scale_dim(ctx.height) as usize;
        let mut rng = ctx.fork_rng("blueprint");

        // --- Shared monochrome source, p99-normalized.
        let mut sample: Vec<f32> = field.iter().copied().filter(|&e| e > 0.0).step_by(13).collect();
        sample.sort_by(f32::total_cmp);
        let reference = sample
            .get(((sample.len().saturating_sub(1)) as f64 * 0.99) as usize)
            .copied()
            .unwrap_or(1.0)
            .max(1e-9);
        let luminance: Vec<f32> = field.iter().map(|&e| (e / reference).min(2.5)).collect();

        // Ridge filter: energy minus its blur, positive part (thins glows
        // into drafting lines).
        let mut blurred = luminance.clone();
        blur_field(&mut blurred, source_w, source_h, RIDGE_SIGMA);
        let ridge: Vec<f32> = luminance
            .iter()
            .zip(blurred.iter())
            .map(|(&raw, &soft)| (raw - soft * 0.85).max(0.0))
            .collect();
        let ridge_max = ridge.iter().copied().fold(1e-9f32, f32::max);

        let sample_at = |data: &[f32], x: usize, y: usize| -> f64 {
            let sx = (x * source_w / out_w).min(source_w - 1);
            let sy = (y * source_h / out_h).min(source_h - 1);
            f64::from(data[sy * source_w + sx])
        };

        // --- (a) Cyanotype blueprint.
        let paper = Paper::Cyanotype;
        let stock = paper_color(paper);
        let white_ink = ink_color(paper);
        let mut blueprint = vec![stock; out_w * out_h];
        for y in 0..out_h {
            for x in 0..out_w {
                let line = (sample_at(&ridge, x, y) / f64::from(ridge_max)).powf(0.65).min(1.0);
                if line > 0.01 {
                    let pixel = &mut blueprint[y * out_w + x];
                    pixel.0 = pixel.0 * (1.0 - line) + white_ink.0 * line;
                    pixel.1 = pixel.1 * (1.0 - line) + white_ink.1 * line;
                    pixel.2 = pixel.2 * (1.0 - line) + white_ink.2 * line;
                }
            }
        }
        // Dimension callouts: bbox width/height plus the deepest approaches.
        let render_ctx = RenderContext::new(
            out_w as u32,
            out_h as u32,
            ctx.positions,
            ctx.settings.aspect_correction,
        );
        let bounds = render_ctx.bounds();
        let page_margin = margin(out_w, out_h);
        let callout_px = type_px(out_w.min(out_h), -1);
        let callout_style =
            TextStyle { tabular: true, ..TextStyle::new(Face::Mono, callout_px, white_ink) };
        // Width dimension line along the bottom margin.
        let dim_y = (out_h - page_margin / 2) as f32;
        draw_line(
            &mut blueprint,
            out_w,
            out_h,
            (page_margin as f32, dim_y),
            ((out_w - page_margin) as f32, dim_y),
            white_ink,
            0.9,
            0.8,
        );
        draw_text(
            &mut blueprint,
            out_w,
            out_h,
            out_w as f64 / 2.0 - callout_px * 4.0,
            f64::from(dim_y) - callout_px * 0.4,
            &callout_style,
            &format!("W {:.3}", bounds.width),
        );
        let dim_x = (page_margin / 2) as f32;
        draw_line(
            &mut blueprint,
            out_w,
            out_h,
            (dim_x, page_margin as f32),
            (dim_x, (out_h - page_margin) as f32),
            white_ink,
            0.9,
            0.8,
        );
        draw_text(
            &mut blueprint,
            out_w,
            out_h,
            f64::from(dim_x) + callout_px * 0.4,
            out_h as f64 / 2.0,
            &callout_style,
            &format!("H {:.3}", bounds.height),
        );
        // Periapsis radius callouts with greedy least-ink leader placement.
        let mut deepest = ctx.events().periapses.clone();
        deepest.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        for approach in deepest.iter().take(CALLOUTS) {
            let midpoint = (ctx.positions[approach.pair.0][approach.step]
                + ctx.positions[approach.pair.1][approach.step])
                * 0.5;
            let (px, py) = render_ctx.to_pixel(midpoint.x, midpoint.y);
            let candidates = [(1.0f64, -1.0f64), (-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
            let leader = callout_px * 5.0;
            let mut best = (f64::INFINITY, candidates[0]);
            for candidate in candidates {
                let tx = (f64::from(px) + candidate.0 * leader) as usize;
                let ty = (f64::from(py) + candidate.1 * leader) as usize;
                if tx + 40 >= out_w || ty >= out_h || tx < 40 {
                    continue;
                }
                // Ink density around the label target: pick the emptiest.
                let mut ink_sum = 0.0;
                for dy in 0..8usize {
                    for dx in 0..24usize {
                        let index = (ty + dy).min(out_h - 1) * out_w + (tx + dx).min(out_w - 1);
                        ink_sum += blueprint[index].0 + blueprint[index].1;
                    }
                }
                if ink_sum < best.0 {
                    best = (ink_sum, candidate);
                }
            }
            let target = (f64::from(px) + best.1.0 * leader, f64::from(py) + best.1.1 * leader);
            draw_line(
                &mut blueprint,
                out_w,
                out_h,
                (px, py),
                (target.0 as f32, target.1 as f32),
                white_ink,
                0.8,
                0.75,
            );
            draw_text(
                &mut blueprint,
                out_w,
                out_h,
                target.0 + callout_px * 0.3,
                target.1,
                &callout_style,
                &format!("R {:.4}", approach.distance),
            );
        }
        draw_footer(&mut blueprint, out_w, out_h, ctx, "V59 BLUEPRINT", paper);
        let image = encode_linear_rec2020_png16(&blueprint, out_w as u32, out_h as u32);
        sink.save_png16(&image, "blueprint.png")?;

        // --- (b) Glass-plate negative.
        let plate_paper = Paper::ArchivalCream;
        let cream = paper_color(plate_paper);
        let plate_ink = ink_color(plate_paper);
        let mut plate = vec![cream; out_w * out_h];
        for y in 0..out_h {
            for x in 0..out_w {
                let density = hd_curve(sample_at(&luminance, x, y).powf(0.8).min(1.0));
                let pixel = &mut plate[y * out_w + x];
                pixel.0 = cream.0 * (1.0 - density) + plate_ink.0 * density;
                pixel.1 = cream.1 * (1.0 - density) + plate_ink.1 * density;
                pixel.2 = cream.2 * (1.0 - density) + plate_ink.2 * density;
            }
        }
        // Seeded plate artifacts: hairline scratches and a corner chip.
        let scratches = 2 + (rng.next_f64() * 1.99) as usize;
        for _ in 0..scratches {
            let x0 = rng.next_f64() * out_w as f64;
            let drift = (rng.next_f64() - 0.5) * out_w as f64 * 0.15;
            let segments = 24;
            let mut previous = (x0 as f32, 0.0f32);
            for segment in 1..=segments {
                let t = f64::from(segment) / f64::from(segments);
                let wobble = (rng.next_f64() - 0.5) * out_w as f64 * 0.004;
                let point = ((x0 + drift * t + wobble) as f32, (t * out_h as f64) as f32);
                draw_line(
                    &mut plate,
                    out_w,
                    out_h,
                    previous,
                    point,
                    (plate_ink.0 * 0.7, plate_ink.1 * 0.7, plate_ink.2 * 0.7),
                    0.7,
                    0.18,
                );
                previous = point;
            }
        }
        // Corner emulsion chip: a lighter wedge at a seeded corner.
        let corner = (rng.next_f64() * 3.99) as usize;
        let chip = out_w.min(out_h) / 22;
        for dy in 0..chip {
            for dx in 0..(chip - dy) {
                let (x, y) = match corner {
                    0 => (dx, dy),
                    1 => (out_w - 1 - dx, dy),
                    2 => (dx, out_h - 1 - dy),
                    _ => (out_w - 1 - dx, out_h - 1 - dy),
                };
                let pixel = &mut plate[y * out_w + x];
                pixel.0 = pixel.0 * 0.35 + 0.85 * 0.65;
                pixel.1 = pixel.1 * 0.35 + 0.86 * 0.65;
                pixel.2 = pixel.2 * 0.35 + 0.84 * 0.65;
            }
        }
        // Catalog number in the italic cut.
        let catalog_px = type_px(out_w.min(out_h), 1);
        let catalog_style =
            TextStyle { opacity: 0.9, ..TextStyle::new(Face::SansItalic, catalog_px, plate_ink) };
        draw_text(
            &mut plate,
            out_w,
            out_h,
            (out_w - page_margin * 4) as f64,
            (out_h - page_margin) as f64,
            &catalog_style,
            &format!("C.S. {}", ctx.seed_hex.to_uppercase()),
        );
        draw_footer(&mut plate, out_w, out_h, ctx, "V59 GLASS PLATE", plate_paper);
        let plate_image = encode_linear_rec2020_png16(&plate, out_w as u32, out_h as u32);
        sink.save_png16(&plate_image, "glass_plate.png")?;

        let meta = serde_json::json!({
            "ridge_sigma": RIDGE_SIGMA,
            "callouts": CALLOUTS,
            "scratches": scratches,
            "bbox_world": { "width": bounds.width, "height": bounds.height },
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("blueprint_params.json", &json, "data")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hd_curve_has_toe_and_shoulder() {
        assert!(hd_curve(0.0) > 0.0 && hd_curve(0.0) < 0.15);
        assert!(hd_curve(1.0) > 0.9);
        let mid = hd_curve(0.5);
        assert!(mid > hd_curve(0.25) && mid < hd_curve(0.75), "monotonic");
    }
}
