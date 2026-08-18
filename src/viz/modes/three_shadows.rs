//! V25 `three-shadows` -- The Cave Wall Triptych.
//!
//! Orthographic `XY`, `XZ`, and `YZ` accumulations of the same 3D object as a
//! formal triptych: each panel a complete artwork, all three provably the
//! same trajectory. Panels share one world scale (honest relative sizes) and
//! one joint tonemap so the triptych hangs as a single piece.

use crate::error::Result;
use crate::render::context::{BoundingBox, PixelBuffer};
use crate::render::{ImageBuffer, Rgb};
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::Accumulator;
use crate::viz::common::display::grade_with_levels;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use nalgebra::Vector3;

/// Gutter fraction of panel width in the triptych.
const GUTTER_FRACTION: f64 = 1.0 / 48.0;
/// Joint-histogram sample stride.
const SAMPLE_STRIDE: usize = 17;

/// The three-shadows triptych mode.
pub struct ThreeShadows;

/// Axis permutation for one panel: world -> (`plot_x`, `plot_y`, depth).
fn permute(positions: &[Vec<Vector3<f64>>], panel: usize) -> Vec<Vec<Vector3<f64>>> {
    positions
        .iter()
        .map(|body| {
            body.iter()
                .map(|p| match panel {
                    0 => *p,                          // XY (depth = z)
                    1 => Vector3::new(p.x, p.z, p.y), // XZ (depth = y)
                    _ => Vector3::new(p.y, p.z, p.x), // YZ (depth = x)
                })
                .collect()
        })
        .collect()
}

impl VizMode for ThreeShadows {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("three-shadows").expect("three-shadows is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let panel_w = ctx.quality.scale_dim(ctx.width);
        let panel_h = ctx.quality.scale_dim(ctx.height);
        let clip_black = ctx.settings.resolved_config.clip_black;
        let clip_white = ctx.settings.resolved_config.clip_white;
        let steps = ctx.step_count();

        // Shared world scale: the maximum half-extents across all panels.
        let projections: Vec<Vec<Vec<Vector3<f64>>>> =
            (0..3).map(|panel| permute(ctx.positions, panel)).collect();
        let mut half_extent = 0.0f64;
        let mut centers = Vec::new();
        for projection in &projections {
            let bounds = BoundingBox::from_positions(projection);
            centers
                .push(((bounds.min_x + bounds.max_x) * 0.5, (bounds.min_y + bounds.max_y) * 0.5));
            half_extent = half_extent
                .max((bounds.max_x - bounds.min_x) * 0.5)
                .max((bounds.max_y - bounds.min_y) * 0.5);
        }
        half_extent *= 1.06; // breathing margin
        let aspect = f64::from(panel_w) / f64::from(panel_h);
        let (half_w_world, half_h_world) = if aspect >= 1.0 {
            (half_extent * aspect, half_extent)
        } else {
            (half_extent, half_extent / aspect)
        };

        // Accumulate each panel with shared-scale bounds; pool histogram
        // samples for one joint grading.
        let mut panel_rgba: Vec<PixelBuffer> = Vec::with_capacity(3);
        let mut pooled_samples: Vec<[f64; 3]> = Vec::new();
        for (panel, projection) in projections.into_iter().enumerate() {
            let bounds = BoundingBox {
                min_x: centers[panel].0 - half_w_world,
                max_x: centers[panel].0 + half_w_world,
                min_y: centers[panel].1 - half_h_world,
                max_y: centers[panel].1 + half_h_world,
                width: half_w_world * 2.0,
                height: half_h_world * 2.0,
            };
            let mut accumulator = Accumulator::with_bounds(
                projection,
                ctx.colors.to_vec(),
                ctx.body_alphas.to_vec(),
                panel_w,
                panel_h,
                bounds,
                ctx.settings.traits,
                ctx.settings.render_config.hdr_scale,
            );
            accumulator.accumulate(0..steps);
            let rgba = accumulator.convert();
            pooled_samples.extend(
                rgba.iter().step_by(SAMPLE_STRIDE).map(|&(r, g, b, a)| [r * a, g * a, b * a]),
            );
            panel_rgba.push(rgba);
        }

        let analysis =
            crate::render::histogram::analyze_tonemapping(&pooled_samples, clip_black, clip_white);
        let levels = crate::render::ChannelLevels::with_tone_mapping(
            analysis.black_r,
            analysis.white_r,
            analysis.black_g,
            analysis.white_g,
            analysis.black_b,
            analysis.white_b,
            crate::render::ToneMappingControls {
                exposure_scale: analysis.exposure_scale,
                paper_white: crate::render::constants::DEFAULT_TONEMAP_PAPER_WHITE,
                highlight_rolloff: crate::render::constants::DEFAULT_TONEMAP_HIGHLIGHT_ROLLOFF,
            },
        );

        let names = ["panel_xy.png", "panel_xz.png", "panel_yz.png"];
        let mut panels = Vec::new();
        for (rgba, name) in panel_rgba.iter().zip(names.iter()) {
            let image = grade_with_levels(rgba, panel_w, panel_h, &levels);
            sink.save_png16(&image, name)?;
            panels.push(image);
        }

        // Compose the triptych.
        let gutter = (f64::from(panel_w) * GUTTER_FRACTION) as u32;
        let total_w = panel_w * 3 + gutter * 4;
        let total_h = panel_h + gutter * 2;
        let mut triptych: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::new(total_w, total_h);
        for (index, panel) in panels.iter().enumerate() {
            let origin_x = gutter + index as u32 * (panel_w + gutter);
            for (x, y, pixel) in panel.enumerate_pixels() {
                triptych.put_pixel(origin_x + x, gutter + y, *pixel);
            }
        }
        sink.save_png16(&triptych, "triptych.png")?;
        Ok(())
    }
}
