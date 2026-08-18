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

/// Per-row crossing: x position, winding delta, body, and the simulation
/// step at which the crossing segment completes (time-resolved winding).
/// Exported for V66 `rose-window` (incremental fractures).
#[derive(Clone, Copy)]
pub(crate) struct RowCrossing {
    pub(crate) x: f32,
    pub(crate) body: u8,
    pub(crate) delta: i8,
    /// Step tag; the loop-closure segment carries `u32::MAX` so finite
    /// `max_step` sweeps exclude it (open paths close dynamically instead).
    pub(crate) step: u32,
}

/// Push the scanline crossings of one segment into per-row buckets.
fn push_segment_crossings(
    rows: &mut [Vec<RowCrossing>],
    from: (f32, f32),
    to: (f32, f32),
    body: u8,
    step: u32,
) {
    let height = rows.len();
    let (x0, y0) = from;
    let (x1, y1) = to;
    if (y1 - y0).abs() < f32::EPSILON {
        return;
    }
    let delta: i8 = if y1 > y0 { 1 } else { -1 };
    let (top, bottom) = if y0 < y1 { (y0, y1) } else { (y1, y0) };
    let row_start = (f64::from(top) - 0.5).ceil().max(0.0) as usize;
    let row_end_f = (f64::from(bottom) - 0.5).floor().min((height - 1) as f64);
    if row_end_f < 0.0 {
        return;
    }
    let row_end = row_end_f as usize;
    for (row, bucket) in rows.iter_mut().enumerate().take(row_end + 1).skip(row_start) {
        let scan_y = row as f32 + 0.5;
        let t = (scan_y - y0) / (y1 - y0);
        let x = x0 + t * (x1 - x0);
        bucket.push(RowCrossing { x, body, delta, step });
    }
}

/// Build the per-row crossing buckets for all three bodies (x-sorted per
/// row), including the loop-closure segments tagged `u32::MAX`. Shared by
/// V41's still sweep and V66's incremental winding.
pub(crate) fn row_crossings(
    positions: &[Vec<nalgebra::Vector3<f64>>],
    render_ctx: &RenderContext,
    height: usize,
) -> Vec<Vec<RowCrossing>> {
    let mut rows: Vec<Vec<RowCrossing>> = vec![Vec::new(); height];
    for (body, path) in positions.iter().enumerate().take(3) {
        let points: Vec<(f32, f32)> = path
            .iter()
            .step_by(PATH_STRIDE)
            .map(|point| render_ctx.to_pixel(point.x, point.y))
            .collect();
        for (segment, pair) in points.windows(2).enumerate() {
            // The crossing "completes" when the segment's end step is reached.
            let step = ((segment + 1) * PATH_STRIDE) as u32;
            push_segment_crossings(&mut rows, pair[0], pair[1], body as u8, step);
        }
        // Close the loop: winding is then well-defined for the still.
        if let (Some(&last), Some(&first)) = (points.last(), points.first()) {
            push_segment_crossings(&mut rows, last, first, body as u8, u32::MAX);
        }
    }
    for bucket in &mut rows {
        bucket.sort_by(|lhs, rhs| lhs.x.total_cmp(&rhs.x));
    }
    rows
}

/// The three dynamic loop-closure segments (from, to) in pixel space.
pub(crate) type ClosureSegments = [((f32, f32), (f32, f32)); 3];

/// Sweep the buckets into a pixel-center winding-triple grid, counting only
/// crossings with `step <= max_step`, with optional dynamic loop-closure
/// segments (current position back to start, per body) for open paths.
/// Exported for V66 `rose-window`.
pub(crate) fn winding_triples(
    rows: &[Vec<RowCrossing>],
    width: usize,
    max_step: u32,
    closure: Option<&ClosureSegments>,
) -> Vec<(i32, i32, i32)> {
    let height = rows.len();
    // Dynamic closure crossings, bucketed then merged during each sweep.
    let mut closure_rows: Vec<Vec<RowCrossing>> = vec![Vec::new(); height];
    if let Some(segments) = closure {
        for (body, &(from, to)) in segments.iter().enumerate() {
            push_segment_crossings(&mut closure_rows, from, to, body as u8, 0);
        }
        for bucket in &mut closure_rows {
            bucket.sort_by(|lhs, rhs| lhs.x.total_cmp(&rhs.x));
        }
    }

    let mut triples = vec![(0i32, 0i32, 0i32); width * height];
    let mut merged: Vec<RowCrossing> = Vec::new();
    for (row, bucket) in rows.iter().enumerate() {
        merged.clear();
        merged.extend(bucket.iter().filter(|event| event.step <= max_step).copied());
        merged.extend_from_slice(&closure_rows[row]);
        merged.sort_by(|lhs, rhs| lhs.x.total_cmp(&rhs.x));

        let mut winding = [0i32; 3];
        let mut event_index = 0usize;
        let row_base = row * width;
        for px in 0..width {
            let center = px as f32 + 0.5;
            while event_index < merged.len() && merged[event_index].x <= center {
                let event = merged[event_index];
                winding[event.body as usize] += i32::from(event.delta);
                event_index += 1;
            }
            triples[row_base + px] = (winding[0], winding[1], winding[2]);
        }
    }
    triples
}

/// The palette anchor hue shared by V41's panes and V66's window.
pub(crate) fn anchor_hue(ctx: &VizContext<'_>) -> f64 {
    let mut sum = (0.0, 0.0);
    for body in 0..3 {
        let (_, a, b) = ctx.mean_color(body);
        sum.0 += a;
        sum.1 += b;
    }
    sum.1.atan2(sum.0).to_degrees().rem_euclid(360.0)
}

/// Map a winding triple to an `OkLab` pane color.
/// Exported for V66 `rose-window` (its pane easing runs in `OkLab`).
pub(crate) fn pane_oklab(triple: (i32, i32, i32), anchor_hue: f64) -> (f64, f64, f64) {
    let (w1, w2, w3) = triple;
    if w1 == 0 && w2 == 0 && w3 == 0 {
        return oklch_to_oklab(PAPER_LIGHTNESS, 0.0, 0.0);
    }
    let u = f64::from(w1 - w3);
    let v = f64::from(w2 - w3);
    let angle = v.atan2(u).to_degrees();
    let hue = (anchor_hue + angle).rem_euclid(360.0);
    let total = f64::from(w1.abs() + w2.abs() + w3.abs());
    let lightness = (0.22 + 0.14 * (1.0 + total).log2()).clamp(0.22, 0.78);
    let chroma = 0.6 * max_display_p3_chroma_for_lh(lightness, hue);
    oklch_to_oklab(lightness, chroma, hue)
}

/// Map a winding triple to a linear Rec.2020 pane color.
pub(crate) fn pane_color(triple: (i32, i32, i32), anchor_hue: f64) -> (f64, f64, f64) {
    let (l, a, b) = pane_oklab(triple, anchor_hue);
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

        // Per-row crossing buckets across all bodies (shared with V66).
        let rows = row_crossings(ctx.positions, &render_ctx, height);

        // Sweep each row left-to-right, filling winding-triple spans.
        let anchor = anchor_hue(ctx);

        let mut color_cache: HashMap<(i32, i32, i32), (f64, f64, f64)> = HashMap::new();
        let mut cached_color = |triple: (i32, i32, i32)| -> (f64, f64, f64) {
            *color_cache.entry(triple).or_insert_with(|| pane_color(triple, anchor))
        };

        let mut pixels = vec![(0.0, 0.0, 0.0); width * height];
        let mut triples = vec![(0i32, 0i32, 0i32); width * height];
        for (row, row_events) in rows.iter().enumerate() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    /// Three concentric circular loops (one per body), plus enough padding
    /// that auto-bounds keep the loops in-frame.
    fn circle_positions(steps: usize) -> Vec<Vec<Vector3<f64>>> {
        (0..3)
            .map(|body| {
                let radius = 1.0 + 0.3 * f64::from(body);
                (0..steps)
                    .map(|step| {
                        let t = step as f64 / steps as f64 * std::f64::consts::TAU;
                        Vector3::new(radius * t.cos(), radius * t.sin(), 0.0)
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn full_sweep_winds_once_inside_and_zero_outside() {
        let positions = circle_positions(4_000);
        let (width, height) = (96usize, 96usize);
        let render_ctx = RenderContext::new(width as u32, height as u32, &positions, false);
        let rows = row_crossings(&positions, &render_ctx, height);
        let triples = winding_triples(&rows, width, u32::MAX, None);

        let center = triples[(height / 2) * width + width / 2];
        assert_eq!(
            (center.0.abs(), center.1.abs(), center.2.abs()),
            (1, 1, 1),
            "center must be wound once by every body, got {center:?}"
        );
        assert_eq!(triples[0], (0, 0, 0), "corner must be unwound");
        assert_eq!(triples[triples.len() - 1], (0, 0, 0));
    }

    #[test]
    fn final_frame_with_dynamic_closure_matches_the_closed_still() {
        let positions = circle_positions(2_000);
        let (width, height) = (64usize, 64usize);
        let render_ctx = RenderContext::new(width as u32, height as u32, &positions, false);
        let rows = row_crossings(&positions, &render_ctx, height);

        let closed = winding_triples(&rows, width, u32::MAX, None);
        let last_step = (positions[0].len() - 1) as u32;
        let closure: [((f32, f32), (f32, f32)); 3] = std::array::from_fn(|body| {
            let path = &positions[body];
            let last = path.last().expect("nonempty");
            let first = path.first().expect("nonempty");
            (render_ctx.to_pixel(last.x, last.y), render_ctx.to_pixel(first.x, first.y))
        });
        let open = winding_triples(&rows, width, last_step, Some(&closure));
        let differing = closed.iter().zip(open.iter()).filter(|(a, b)| a != b).count();
        // The dynamic closure retraces the static one; only pixels touched
        // by the final path stride may differ.
        assert!(
            differing < width * height / 100,
            "final-frame winding must match the closed still ({differing} pixels differ)"
        );
    }

    #[test]
    fn partial_sweep_grows_winding_monotonically_in_covered_area() {
        let positions = circle_positions(2_000);
        let (width, height) = (64usize, 64usize);
        let render_ctx = RenderContext::new(width as u32, height as u32, &positions, false);
        let rows = row_crossings(&positions, &render_ctx, height);

        let wound_area = |max_step: u32| -> usize {
            let closure: [((f32, f32), (f32, f32)); 3] = std::array::from_fn(|body| {
                let path = &positions[body];
                let cursor = (max_step as usize).min(path.len() - 1);
                let first = path.first().expect("nonempty");
                (
                    render_ctx.to_pixel(path[cursor].x, path[cursor].y),
                    render_ctx.to_pixel(first.x, first.y),
                )
            });
            winding_triples(&rows, width, max_step, Some(&closure))
                .iter()
                .filter(|&&triple| triple != (0, 0, 0))
                .count()
        };
        let quarter = wound_area(500);
        let half = wound_area(1_000);
        let full = wound_area(1_999);
        assert!(
            quarter <= half && half <= full,
            "wound area must grow with time: {quarter} {half} {full}"
        );
        assert!(full > 0, "the finished loops must wind a region");
    }
}
