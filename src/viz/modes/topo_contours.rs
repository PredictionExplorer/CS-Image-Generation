//! V61 `topo-contours` -- The Terrain of Light.
//!
//! Treats accumulated energy as elevation and draws a topographic survey of
//! the artwork's own brightness: index and intermediate contours, a
//! hypsometric whisper tint from the palette, and prominence-filtered spot
//! heights at the summits. Emitted on archival-cream and deep-black stocks.
//! (In-line contour labels arrive with `text.rs`; values live in the JSON.)

use crate::error::Result;
use crate::oklab::{oklab_to_linear_rec2020, oklab_to_oklch, oklch_to_oklab};
use crate::viz::VizMode;
use crate::viz::VizPhase;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::contours::marching_squares;
use crate::viz::common::display::encode_linear_rec2020_png16;
use crate::viz::common::raster::{Rgb64, blur_field, draw_cross, draw_line};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use tracing::warn;

/// Number of contour levels (every 5th is an index contour).
const LEVEL_COUNT: usize = 12;
/// Field smoothing sigma in pixels.
const FIELD_SIGMA: f64 = 2.4;
/// Number of spot-height markers.
const SPOT_HEIGHTS: usize = 7;
/// Minimum separation between spot heights (pixels).
const SPOT_SEPARATION: f64 = 120.0;

/// The topographic contours mode.
pub struct TopoContours;

/// Paper stock and ink pair.
struct Stock {
    name: &'static str,
    paper: Rgb64,
    ink: Rgb64,
}

fn oklch_rgb(l: f64, c: f64, h: f64) -> Rgb64 {
    let (ll, a, b) = oklch_to_oklab(l, c, h);
    let (r, g, bl) = oklab_to_linear_rec2020(ll, a, b);
    (r.max(0.0), g.max(0.0), bl.max(0.0))
}

impl VizMode for TopoContours {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("topo-contours").expect("topo-contours is in the catalog")
    }

    fn phase(&self) -> VizPhase {
        VizPhase::Spd
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let Some(energy) = ctx.energy_field() else {
            warn!("topo-contours skipped: SPD buffer unavailable");
            return Ok(());
        };
        let width = ctx.width as usize;
        let height = ctx.height as usize;

        // Elevation: asinh-compressed, smoothed, normalized to [0, 1].
        let mut elevation: Vec<f32> = energy.iter().map(|&value| value.asinh()).collect();
        blur_field(&mut elevation, width, height, FIELD_SIGMA);
        let max = elevation.iter().copied().fold(1e-9f32, f32::max);
        for value in &mut elevation {
            *value /= max;
        }

        // Palette hue for the hypsometric tint.
        let anchor_hue = {
            let (l, a, b) = ctx.mean_color(0);
            let (_, _, hue) = oklab_to_oklch(l, a, b);
            hue
        };

        // Spot heights: local maxima with non-maximum suppression.
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let index = y * width + x;
                let value = elevation[index];
                if value < 0.35 {
                    continue;
                }
                let is_peak = [
                    index - 1,
                    index + 1,
                    index - width,
                    index + width,
                    index - width - 1,
                    index - width + 1,
                    index + width - 1,
                    index + width + 1,
                ]
                .iter()
                .all(|&neighbor| elevation[neighbor] <= value);
                if is_peak {
                    candidates.push((index, value));
                }
            }
        }
        candidates.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));
        let mut summits: Vec<(f64, f64, f32)> = Vec::new();
        for (index, value) in candidates {
            let x = (index % width) as f64;
            let y = (index / width) as f64;
            let far_enough = summits.iter().all(|&(sx, sy, _)| {
                ((x - sx).powi(2) + (y - sy).powi(2)).sqrt() >= SPOT_SEPARATION
            });
            if far_enough {
                summits.push((x, y, value));
                if summits.len() >= SPOT_HEIGHTS {
                    break;
                }
            }
        }

        // Contour levels in normalized elevation.
        let levels: Vec<f32> =
            (1..=LEVEL_COUNT).map(|index| index as f32 / (LEVEL_COUNT + 1) as f32).collect();

        let stocks = [
            Stock {
                name: "topo.png",
                paper: oklch_rgb(0.94, 0.02, 95.0),
                ink: oklch_rgb(0.25, 0.03, 60.0),
            },
            Stock {
                name: "topo_dark.png",
                paper: oklch_rgb(0.06, 0.005, 260.0),
                ink: oklch_rgb(0.78, 0.02, 95.0),
            },
        ];

        for stock in &stocks {
            let mut pixels: Vec<Rgb64> = (0..width * height)
                .map(|index| {
                    // Hypsometric whisper tint toward the palette hue.
                    let value = f64::from(elevation[index]);
                    let tint = oklch_rgb(0.5, (0.05 * value).min(0.05), anchor_hue);
                    let blend = 0.35 * value;
                    (
                        stock.paper.0 * (1.0 - blend) + tint.0 * blend,
                        stock.paper.1 * (1.0 - blend) + tint.1 * blend,
                        stock.paper.2 * (1.0 - blend) + tint.2 * blend,
                    )
                })
                .collect();

            for (level_index, &level) in levels.iter().enumerate() {
                let is_index_contour = (level_index + 1) % 5 == 0;
                let stroke = if is_index_contour { 2.4 } else { 1.0 };
                let opacity = if is_index_contour { 0.95 } else { 0.65 };
                for segment in marching_squares(&elevation, width, height, level) {
                    draw_line(
                        &mut pixels,
                        width,
                        height,
                        segment.0,
                        segment.1,
                        stock.ink,
                        stroke,
                        opacity,
                    );
                }
            }

            for &(x, y, _) in &summits {
                draw_cross(
                    &mut pixels,
                    width,
                    height,
                    (x as f32, y as f32),
                    9.0,
                    stock.ink,
                    1.6,
                    0.95,
                );
            }

            sink.save_png16(
                &encode_linear_rec2020_png16(&pixels, ctx.width, ctx.height),
                stock.name,
            )?;
        }

        let meta = serde_json::json!({
            "levels": levels,
            "field_sigma": FIELD_SIGMA,
            "summits": summits
                .iter()
                .map(|&(x, y, value)| serde_json::json!({ "x": x, "y": y, "elevation": value }))
                .collect::<Vec<_>>(),
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("topo.json", &json, "data")?;
        Ok(())
    }
}
