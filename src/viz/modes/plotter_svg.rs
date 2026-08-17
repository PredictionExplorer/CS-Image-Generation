//! V51 `plotter-svg` -- Ink and Thread.
//!
//! Exports the trajectory as an A3 pen-plotter SVG: one layer per body,
//! Ramer-Douglas-Peucker simplified paths, three physical pen widths driven
//! by orbital speed (slow arcs bold, fast whips fine), and greedy
//! nearest-endpoint ordering to minimize pen-up travel. A stats sidecar
//! reports path lengths and an estimated plot time.

use crate::error::Result;
use crate::render::context::RenderContext;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::vector_export::{
    SvgLayer, SvgPath, greedy_path_order, oklab_to_srgb_hex, polyline_length, rdp_simplify,
    write_svg,
};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;

/// A3 landscape dimensions in millimeters.
const PAGE_W_MM: f64 = 420.0;
/// A3 landscape height in millimeters.
const PAGE_H_MM: f64 = 297.0;
/// Page margin in millimeters.
const MARGIN_MM: f64 = 20.0;
/// RDP tolerance in millimeters (fineliner-scale detail).
const RDP_TOLERANCE_MM: f64 = 0.15;
/// Chunk length (points) before width reclassification.
const CHUNK_POINTS: usize = 900;
/// Pen widths in millimeters, slow to fast.
const PEN_WIDTHS_MM: [f64; 3] = [0.8, 0.5, 0.3];
/// Assumed plot speed for the time estimate (mm/s).
const PLOT_SPEED_MM_S: f64 = 50.0;

/// The pen-plotter SVG export mode.
pub struct PlotterSvg;

impl VizMode for PlotterSvg {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("plotter-svg").expect("plotter-svg is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let render_ctx = RenderContext::new(
            ctx.width,
            ctx.height,
            ctx.positions,
            ctx.settings.aspect_correction,
        );
        let drawable_w = PAGE_W_MM - 2.0 * MARGIN_MM;
        let drawable_h = PAGE_H_MM - 2.0 * MARGIN_MM;
        let scale = (drawable_w / f64::from(ctx.width)).min(drawable_h / f64::from(ctx.height));
        let offset_x = (PAGE_W_MM - f64::from(ctx.width) * scale) * 0.5;
        let offset_y = (PAGE_H_MM - f64::from(ctx.height) * scale) * 0.5;
        let to_mm = |point: &nalgebra::Vector3<f64>| -> (f64, f64) {
            let (px, py) = render_ctx.to_pixel(point.x, point.y);
            (offset_x + f64::from(px) * scale, offset_y + f64::from(py) * scale)
        };

        let kinematics = ctx.kinematics();
        let window = kinematics.speed_window();
        let steps = ctx.step_count();
        // Keep the source polyline tractable before RDP.
        let stride = (steps / 400_000).max(1);

        let mut layers = Vec::new();
        let mut pen_down_mm = 0.0_f64;
        let mut pen_up_mm = 0.0_f64;

        for body in 0..3 {
            // Chunk the strand, classify each chunk by mean speed tercile.
            let mut paths: Vec<SvgPath> = Vec::new();
            let indices: Vec<usize> = (0..steps).step_by(stride).collect();
            for chunk_indices in indices.chunks(CHUNK_POINTS) {
                if chunk_indices.len() < 2 {
                    continue;
                }
                let mean_speed =
                    chunk_indices.iter().map(|&step| kinematics.speeds[body][step]).sum::<f64>()
                        / chunk_indices.len() as f64;
                let normalized = kinematics.normalized_speed(window, mean_speed);
                let width_class = ((normalized * 3.0) as usize).min(2);

                let points_mm: Vec<(f64, f64)> =
                    chunk_indices.iter().map(|&step| to_mm(&ctx.positions[body][step])).collect();
                let simplified = rdp_simplify(&points_mm, RDP_TOLERANCE_MM);
                if simplified.len() < 2 {
                    continue;
                }
                paths.push(SvgPath { width_mm: PEN_WIDTHS_MM[width_class], points: simplified });
            }

            // Greedy pen-up minimization.
            let order = greedy_path_order(&paths);
            let mut cursor = (0.0, 0.0);
            let ordered: Vec<SvgPath> = order
                .iter()
                .map(|&index| {
                    let path = &paths[index];
                    if let (Some(&start), Some(&end)) = (path.points.first(), path.points.last()) {
                        let dx = start.0 - cursor.0;
                        let dy = start.1 - cursor.1;
                        pen_up_mm += (dx * dx + dy * dy).sqrt();
                        pen_down_mm += polyline_length(&path.points);
                        cursor = end;
                    }
                    SvgPath { width_mm: path.width_mm, points: path.points.clone() }
                })
                .collect();

            let (l, a, b) = ctx.mean_color(body);
            layers.push(SvgLayer {
                id: format!("body-{}", body + 1),
                color_hex: oklab_to_srgb_hex(l.max(0.65), a, b),
                paths: ordered,
            });
        }

        write_svg(&sink.path("plotter.svg"), PAGE_W_MM, PAGE_H_MM, &layers)?;
        sink.record("plotter.svg", "vector");

        let stats = serde_json::json!({
            "page_mm": [PAGE_W_MM, PAGE_H_MM],
            "margin_mm": MARGIN_MM,
            "rdp_tolerance_mm": RDP_TOLERANCE_MM,
            "pen_widths_mm": PEN_WIDTHS_MM,
            "pen_down_mm": pen_down_mm.round(),
            "pen_up_mm": pen_up_mm.round(),
            "estimated_minutes": (pen_down_mm + pen_up_mm) / PLOT_SPEED_MM_S / 60.0,
            "layers": layers
                .iter()
                .map(|layer| {
                    serde_json::json!({
                        "id": layer.id,
                        "color": layer.color_hex,
                        "paths": layer.paths.len(),
                    })
                })
                .collect::<Vec<_>>(),
        });
        let json = serde_json::to_string_pretty(&stats).map_err(std::io::Error::other)?;
        sink.write_text("plot_stats.json", &json, "data")?;
        Ok(())
    }
}
