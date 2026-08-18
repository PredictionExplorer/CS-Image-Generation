//! V04 `prism-portrait` -- The Artwork Through Glass.
//!
//! One directional prism pass over the accumulated SPD: every wavelength bin
//! is displaced along a seed-chosen axis with a Cauchy dispersion profile
//! (violet furthest, red least, 560 nm anchored), then converted and graded
//! with the run's frozen levels so exposure matches the master exactly.

use crate::error::Result;
use crate::render::context::PixelBuffer;
use crate::spectrum::{NUM_BINS, spd_to_rgba, wavelength_nm_for_bin};
use crate::viz::VizMode;
use crate::viz::VizPhase;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::display::grade_with_levels;
use crate::viz::common::kinematics::principal_axis_xy;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use rayon::prelude::*;
use tracing::warn;

/// Maximum displacement as a fraction of the short edge.
const DISPERSION_FRACTION: f64 = 0.022;
/// Refractive-index spread across the visible range (Cauchy fit target).
const INDEX_SPREAD: f64 = 0.014;
/// Anchor wavelength that stays registered (nm).
const ANCHOR_NM: f64 = 560.0;
/// Axis quantization step in degrees.
const AXIS_QUANT_DEG: f64 = 15.0;

/// The prism-portrait mode.
pub struct PrismPortrait;

impl VizMode for PrismPortrait {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("prism-portrait").expect("prism-portrait is in the catalog")
    }

    fn phase(&self) -> VizPhase {
        VizPhase::Spd
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let Some(buffer) = ctx.accum_spd else {
            warn!("prism-portrait skipped: SPD buffer unavailable");
            return Ok(());
        };
        let width = ctx.width as usize;
        let height = ctx.height as usize;

        // Seeded axis: trajectory principal axis, nudged and quantized.
        let (axis_x, axis_y) = principal_axis_xy(ctx.positions);
        let base_angle = axis_y.atan2(axis_x).to_degrees();
        let mut rng = ctx.fork_rng(self.entry().flag);
        let offset = match rng.next_byte() % 4 {
            0 => -AXIS_QUANT_DEG,
            3 => AXIS_QUANT_DEG,
            _ => 0.0, // bias toward the flow direction itself
        };
        let angle_deg = ((base_angle + offset) / AXIS_QUANT_DEG).round() * AXIS_QUANT_DEG;
        let angle = angle_deg.to_radians();
        let dir = (angle.cos(), angle.sin());

        // Cauchy dispersion: n(lambda) = 1 + B / lambda^2 with the spread
        // anchored so n(380) - n(700) = INDEX_SPREAD.
        let cauchy_b = INDEX_SPREAD / (1.0 / (380.0f64 * 380.0) - 1.0 / (700.0f64 * 700.0));
        let index_at = |nm: f64| cauchy_b / (nm * nm);
        let span = index_at(380.0) - index_at(700.0);
        let max_shift = DISPERSION_FRACTION * f64::from(ctx.width.min(ctx.height));
        let shift_per_bin: Vec<f64> = (0..NUM_BINS)
            .map(|bin| {
                let nm = wavelength_nm_for_bin(bin);
                max_shift * (index_at(nm) - index_at(ANCHOR_NM)) / span
            })
            .collect();

        // Fused gather + convert: per output pixel, sample each bin from its
        // wavelength-shifted source position (bilinear) and convert.
        let rgba: PixelBuffer = (0..width * height)
            .into_par_iter()
            .map(|index| {
                let x = (index % width) as f64;
                let y = (index / width) as f64;
                let mut local = [0.0f64; NUM_BINS];
                for (bin, shift) in shift_per_bin.iter().enumerate() {
                    let sx = x - dir.0 * shift;
                    let sy = y - dir.1 * shift;
                    if sx < 0.0 || sy < 0.0 || sx > (width - 1) as f64 || sy > (height - 1) as f64 {
                        continue;
                    }
                    let x0 = sx.floor() as usize;
                    let y0 = sy.floor() as usize;
                    let x1 = (x0 + 1).min(width - 1);
                    let y1 = (y0 + 1).min(height - 1);
                    let fx = sx - x0 as f64;
                    let fy = sy - y0 as f64;
                    local[bin] = buffer[y0 * width + x0][bin] * (1.0 - fx) * (1.0 - fy)
                        + buffer[y0 * width + x1][bin] * fx * (1.0 - fy)
                        + buffer[y1 * width + x0][bin] * (1.0 - fx) * fy
                        + buffer[y1 * width + x1][bin] * fx * fy;
                }
                spd_to_rgba(&local)
            })
            .collect();

        let image = grade_with_levels(&rgba, ctx.width, ctx.height, ctx.levels);
        sink.save_png16(&image, "prism.png")?;

        let meta = serde_json::json!({
            "axis_degrees": angle_deg,
            "max_shift_px": max_shift,
            "anchor_nm": ANCHOR_NM,
            "cauchy_b": cauchy_b,
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("prism_axis.json", &json, "data")?;
        Ok(())
    }
}
