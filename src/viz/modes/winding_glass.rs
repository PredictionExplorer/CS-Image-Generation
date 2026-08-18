//! V41 `winding-glass` -- Topological Stained Glass.
//!
//! For every pixel, the integer winding number of each body's closed path is
//! computed exactly via per-scanline crossing sweeps (O((W+H) * steps), no
//! sampling noise). The integer triple is mapped to flat glass panes in a
//! palette-anchored `OKLCh` family; winding-change boundaries are stroked as
//! dark lead cames.

use crate::error::Result;
use crate::oklab::{max_display_p3_chroma_for_lh, oklab_to_linear_rec2020, oklch_to_oklab};
use crate::render::context::RenderContext;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::display::encode_linear_rec2020_png16;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use std::collections::HashMap;

/// Stride over simulation steps when building path segments.
const PATH_STRIDE: usize = 2;
/// Lead came lightness.
const LEAD_LIGHTNESS: f64 = 0.08;
/// Paper (zero-winding) lightness.
const PAPER_LIGHTNESS: f64 = 0.045;

/// The topological stained-glass mode.
pub struct WindingGlass;

/// Per-row crossing: x position and winding delta for one body.
#[derive(Clone, Copy)]
struct RowCrossing {
    x: f32,
    body: u8,
    delta: i8,
}

/// Map a winding triple to a linear Rec.2020 pane color.
fn pane_color(triple: (i32, i32, i32), anchor_hue: f64) -> (f64, f64, f64) {
    let (w1, w2, w3) = triple;
    if w1 == 0 && w2 == 0 && w3 == 0 {
        let (l, a, b) = oklch_to_oklab(PAPER_LIGHTNESS, 0.0, 0.0);
        return oklab_to_linear_rec2020(l, a, b);
    }
    let u = f64::from(w1 - w3);
    let v = f64::from(w2 - w3);
    let angle = v.atan2(u).to_degrees();
    let hue = (anchor_hue + angle).rem_euclid(360.0);
    let total = f64::from(w1.abs() + w2.abs() + w3.abs());
    let lightness = (0.22 + 0.14 * (1.0 + total).log2()).clamp(0.22, 0.78);
    let chroma = 0.6 * max_display_p3_chroma_for_lh(lightness, hue);
    let (l, a, b) = oklch_to_oklab(lightness, chroma, hue);
    oklab_to_linear_rec2020(l, a, b)
}

impl VizMode for WindingGlass {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("winding-glass").expect("winding-glass is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let width = ctx.width as usize;
        let height = ctx.height as usize;
        let render_ctx = RenderContext::new(
            ctx.width,
            ctx.height,
            ctx.positions,
            ctx.settings.aspect_correction,
        );

        // Per-row crossing buckets across all bodies.
        let mut rows: Vec<Vec<RowCrossing>> = vec![Vec::new(); height];
        for body in 0..3 {
            let path = &ctx.positions[body];
            let mut points: Vec<(f32, f32)> = path
                .iter()
                .step_by(PATH_STRIDE)
                .map(|point| render_ctx.to_pixel(point.x, point.y))
                .collect();
            if let Some(&first) = points.first() {
                points.push(first); // close the loop: winding is then well-defined
            }
            for pair in points.windows(2) {
                let (x0, y0) = pair[0];
                let (x1, y1) = pair[1];
                if (y1 - y0).abs() < f32::EPSILON {
                    continue;
                }
                let delta: i8 = if y1 > y0 { 1 } else { -1 };
                let (top, bottom) = if y0 < y1 { (y0, y1) } else { (y1, y0) };
                let row_start = (f64::from(top) - 0.5).ceil().max(0.0) as usize;
                let row_end_f = (f64::from(bottom) - 0.5).floor().min((height - 1) as f64);
                if row_end_f < 0.0 {
                    continue;
                }
                let row_end = row_end_f as usize;
                for (row, bucket) in rows.iter_mut().enumerate().take(row_end + 1).skip(row_start) {
                    let scan_y = row as f32 + 0.5;
                    let t = (scan_y - y0) / (y1 - y0);
                    let x = x0 + t * (x1 - x0);
                    bucket.push(RowCrossing { x, body: body as u8, delta });
                }
            }
        }

        // Sweep each row left-to-right, filling winding-triple spans.
        let anchor = {
            let mut sum = (0.0, 0.0);
            for body in 0..3 {
                let (_, a, b) = ctx.mean_color(body);
                sum.0 += a;
                sum.1 += b;
            }
            sum.1.atan2(sum.0).to_degrees().rem_euclid(360.0)
        };

        let mut color_cache: HashMap<(i32, i32, i32), (f64, f64, f64)> = HashMap::new();
        let mut cached_color = |triple: (i32, i32, i32)| -> (f64, f64, f64) {
            *color_cache.entry(triple).or_insert_with(|| pane_color(triple, anchor))
        };

        let mut pixels = vec![(0.0, 0.0, 0.0); width * height];
        let mut triples = vec![(0i32, 0i32, 0i32); width * height];
        let mut row_events: Vec<RowCrossing> = Vec::new();
        for (row, bucket) in rows.iter().enumerate() {
            row_events.clear();
            row_events.extend_from_slice(bucket);
            row_events.sort_by(|lhs, rhs| lhs.x.total_cmp(&rhs.x));

            let mut winding = [0i32; 3];
            let mut span_start = 0.0_f32;
            let row_base = row * width;
            let mut event_index = 0;
            while span_start < width as f32 {
                // Advance through events at or before span_start.
                while event_index < row_events.len() && row_events[event_index].x <= span_start {
                    let event = row_events[event_index];
                    winding[event.body as usize] += i32::from(event.delta);
                    event_index += 1;
                }
                let span_end = row_events
                    .get(event_index)
                    .map_or(width as f32, |event| event.x.min(width as f32));
                let triple = (winding[0], winding[1], winding[2]);
                let color = cached_color(triple);

                // Fill whole pixels; blend the fractional boundary pixel.
                let first_pixel = span_start.max(0.0) as usize;
                let last_pixel = (span_end.ceil() as usize).min(width);
                for px in first_pixel..last_pixel {
                    let pixel_start = px as f32;
                    let pixel_end = pixel_start + 1.0;
                    let coverage =
                        (span_end.min(pixel_end) - span_start.max(pixel_start)).clamp(0.0, 1.0);
                    let index = row_base + px;
                    let weight = f64::from(coverage);
                    pixels[index].0 += color.0 * weight;
                    pixels[index].1 += color.1 * weight;
                    pixels[index].2 += color.2 * weight;
                    if coverage > 0.5 {
                        triples[index] = triple;
                    }
                }
                if span_end <= span_start {
                    break; // safety against degenerate float spans
                }
                span_start = span_end;
            }
        }

        // Lead cames along winding-change boundaries.
        let lead = oklab_to_linear_rec2020(LEAD_LIGHTNESS, 0.0, 0.0);
        for row in 0..height {
            for column in 0..width {
                let index = row * width + column;
                let differs_left = column > 0 && triples[index] != triples[index - 1];
                let differs_up = row > 0 && triples[index] != triples[index - width];
                if differs_left || differs_up {
                    pixels[index] = lead;
                    if differs_left {
                        pixels[index - 1] = lead;
                    }
                    if differs_up {
                        pixels[index - width] = lead;
                    }
                }
            }
        }

        let image = encode_linear_rec2020_png16(&pixels, ctx.width, ctx.height);
        sink.save_png16(&image, "winding.png")?;

        // Histogram of winding triples for the key / audits.
        let mut histogram: HashMap<(i32, i32, i32), u64> = HashMap::new();
        for &triple in &triples {
            *histogram.entry(triple).or_insert(0) += 1;
        }
        let mut entries: Vec<_> = histogram
            .into_iter()
            .map(|((w1, w2, w3), count)| {
                serde_json::json!({ "triple": [w1, w2, w3], "pixels": count })
            })
            .collect();
        entries.sort_by_key(|entry| std::cmp::Reverse(entry["pixels"].as_u64().unwrap_or(0)));
        let json = serde_json::to_string_pretty(&entries).map_err(std::io::Error::other)?;
        sink.write_text("winding_histogram.json", &json, "data")?;
        Ok(())
    }
}
