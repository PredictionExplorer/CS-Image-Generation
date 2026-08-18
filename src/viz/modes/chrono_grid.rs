//! V18 `chrono-grid` -- Motion Study Sheet.
//!
//! The run sliced into sixteen equal time windows, each rendered as its own
//! accumulation with the production vocabulary, tiled 4x4 like a Muybridge
//! chronophotography proof sheet. Per-cell auto exposure is clamped around
//! the run's global exposure so all sixteen cells read as one sheet.

use crate::error::Result;
use crate::render::{ImageBuffer, Rgb};
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::Accumulator;
use crate::viz::common::display::{auto_levels, grade_with_levels};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;

/// Number of time windows (grid is 4x4).
const WINDOWS: usize = 16;
/// Grid columns.
const GRID_COLS: usize = 4;
/// Exposure clamp around the run's exposure (factor, +/- half an EV).
const EXPOSURE_CLAMP: f64 = std::f64::consts::SQRT_2;
/// Gutter fraction of cell width.
const GUTTER_FRACTION: f64 = 1.0 / 48.0;

/// The chrono-grid mode.
pub struct ChronoGrid;

impl VizMode for ChronoGrid {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("chrono-grid").expect("chrono-grid is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let cell_w = (ctx.quality.scale_dim(ctx.width) / 2) & !1;
        let cell_h = (ctx.quality.scale_dim(ctx.height) / 2) & !1;
        let clip_black = ctx.settings.resolved_config.clip_black;
        let clip_white = ctx.settings.resolved_config.clip_white;
        let global_exposure = ctx.levels.exposure_scale;
        let steps = ctx.step_count();

        let mut accumulator = Accumulator::new(
            ctx.positions.to_vec(),
            ctx.colors.to_vec(),
            ctx.body_alphas.to_vec(),
            cell_w,
            cell_h,
            ctx.settings.aspect_correction,
            ctx.settings.traits,
            ctx.settings.render_config.hdr_scale,
        );

        let mut cells: Vec<ImageBuffer<Rgb<u16>, Vec<u16>>> = Vec::with_capacity(WINDOWS);
        let mut windows_meta = Vec::new();
        for window in 0..WINDOWS {
            let start = window * steps / WINDOWS;
            let end = ((window + 1) * steps / WINDOWS).min(steps);
            accumulator.clear();
            accumulator.accumulate(start..end);
            let rgba = accumulator.convert();

            // Per-cell exposure, clamped to the run's exposure so sparse
            // windows brighten without breaking sheet consistency.
            let mut levels = auto_levels(&rgba, clip_black, clip_white, 1.0);
            levels.exposure_scale = levels
                .exposure_scale
                .clamp(global_exposure / EXPOSURE_CLAMP, global_exposure * EXPOSURE_CLAMP);
            let cell = grade_with_levels(&rgba, cell_w, cell_h, &levels);
            sink.save_png16(&cell, &format!("cells/window_{window:02}.png"))?;
            windows_meta.push(serde_json::json!({
                "window": window,
                "steps": [start, end],
                "exposure": levels.exposure_scale,
            }));
            cells.push(cell);
        }

        // Compose the 4x4 sheet with gutters on black.
        let gutter = (f64::from(cell_w) * GUTTER_FRACTION) as u32;
        let rows = WINDOWS / GRID_COLS;
        let sheet_w = cell_w * GRID_COLS as u32 + gutter * (GRID_COLS as u32 + 1);
        let sheet_h = cell_h * rows as u32 + gutter * (rows as u32 + 1);
        let mut sheet: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::new(sheet_w, sheet_h);
        for (index, cell) in cells.iter().enumerate() {
            let col = (index % GRID_COLS) as u32;
            let row = (index / GRID_COLS) as u32;
            let origin_x = gutter + col * (cell_w + gutter);
            let origin_y = gutter + row * (cell_h + gutter);
            for (x, y, pixel) in cell.enumerate_pixels() {
                sheet.put_pixel(origin_x + x, origin_y + y, *pixel);
            }
        }
        sink.save_png16(&sheet, "chrono_sheet.png")?;

        let meta = serde_json::json!({
            "windows": windows_meta,
            "cell_size": [cell_w, cell_h],
            "exposure_clamp": EXPOSURE_CLAMP,
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("chrono.json", &json, "data")?;
        Ok(())
    }
}
