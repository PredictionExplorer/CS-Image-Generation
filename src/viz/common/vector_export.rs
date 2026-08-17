//! Vector export helpers: Ramer-Douglas-Peucker polyline simplification,
//! a physical-units SVG writer for pen plotters, and `OkLab` to sRGB hex
//! conversion for stroke colors.

use std::fs::File;
use std::io::{self, BufWriter, Write};

/// A drawable polyline in millimeter coordinates with a physical pen width.
pub struct SvgPath {
    /// Stroke width in millimeters.
    pub width_mm: f64,
    /// Polyline points in millimeters.
    pub points: Vec<(f64, f64)>,
}

/// One plotter layer (typically one pen / one body).
pub struct SvgLayer {
    /// Layer id (used by plotter tooling for pen changes).
    pub id: String,
    /// Stroke color as `#rrggbb`.
    pub color_hex: String,
    /// Paths in draw order.
    pub paths: Vec<SvgPath>,
}

/// Ramer-Douglas-Peucker simplification with a perpendicular tolerance.
#[must_use]
pub fn rdp_simplify(points: &[(f64, f64)], tolerance: f64) -> Vec<(f64, f64)> {
    if points.len() <= 2 {
        return points.to_vec();
    }
    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;
    let mut stack = vec![(0usize, points.len() - 1)];

    while let Some((start, end)) = stack.pop() {
        if end <= start + 1 {
            continue;
        }
        let (sx, sy) = points[start];
        let (ex, ey) = points[end];
        let dx = ex - sx;
        let dy = ey - sy;
        let seg_len = (dx * dx + dy * dy).sqrt().max(1e-12);

        let mut max_dist = -1.0;
        let mut max_index = start;
        for (index, &(px, py)) in points.iter().enumerate().take(end).skip(start + 1) {
            let dist = ((py - sy) * dx - (px - sx) * dy).abs() / seg_len;
            if dist > max_dist {
                max_dist = dist;
                max_index = index;
            }
        }
        if max_dist > tolerance {
            keep[max_index] = true;
            stack.push((start, max_index));
            stack.push((max_index, end));
        }
    }

    points.iter().zip(keep.iter()).filter_map(|(&point, &kept)| kept.then_some(point)).collect()
}

/// Total polyline length in the same units as its points.
#[must_use]
pub fn polyline_length(points: &[(f64, f64)]) -> f64 {
    points
        .windows(2)
        .map(|pair| {
            let dx = pair[1].0 - pair[0].0;
            let dy = pair[1].1 - pair[0].1;
            (dx * dx + dy * dy).sqrt()
        })
        .sum()
}

/// Greedy nearest-endpoint ordering of paths to minimize pen-up travel.
///
/// Deterministic: ties break on original index. Returns the new order as
/// indices into the input slice.
#[must_use]
pub fn greedy_path_order(paths: &[SvgPath]) -> Vec<usize> {
    let count = paths.len();
    let mut used = vec![false; count];
    let mut order = Vec::with_capacity(count);
    let mut cursor = (0.0, 0.0);

    for _ in 0..count {
        let mut best: Option<(usize, f64)> = None;
        for (index, path) in paths.iter().enumerate() {
            if used[index] || path.points.is_empty() {
                continue;
            }
            let start = path.points[0];
            let dx = start.0 - cursor.0;
            let dy = start.1 - cursor.1;
            let dist = dx * dx + dy * dy;
            if best.is_none_or(|(_, best_dist)| dist < best_dist) {
                best = Some((index, dist));
            }
        }
        let Some((chosen, _)) = best else { break };
        used[chosen] = true;
        if let Some(&last) = paths[chosen].points.last() {
            cursor = last;
        }
        order.push(chosen);
    }
    order
}

/// Write an SVG document sized in millimeters (plotter-ready).
pub fn write_svg(path: &str, width_mm: f64, height_mm: f64, layers: &[SvgLayer]) -> io::Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(
        out,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width_mm}mm\" \
         height=\"{height_mm}mm\" viewBox=\"0 0 {width_mm} {height_mm}\">"
    )?;
    writeln!(out, "<rect width=\"{width_mm}\" height=\"{height_mm}\" fill=\"#0a0a0d\"/>")?;
    for layer in layers {
        writeln!(
            out,
            "<g id=\"{}\" fill=\"none\" stroke=\"{}\" stroke-linecap=\"round\" \
             stroke-linejoin=\"round\">",
            layer.id, layer.color_hex
        )?;
        for svg_path in &layer.paths {
            if svg_path.points.len() < 2 {
                continue;
            }
            use std::fmt::Write as _;
            let mut data = String::with_capacity(svg_path.points.len() * 14);
            for (index, &(x, y)) in svg_path.points.iter().enumerate() {
                let command = if index == 0 { 'M' } else { 'L' };
                let _ = write!(data, "{command}{x:.3} {y:.3} ");
            }
            writeln!(
                out,
                "<path d=\"{}\" stroke-width=\"{:.2}\"/>",
                data.trim_end(),
                svg_path.width_mm
            )?;
        }
        writeln!(out, "</g>")?;
    }
    writeln!(out, "</svg>")?;
    out.flush()
}

/// Convert an `OkLab` color to an sRGB hex string (gamut-clamped).
#[must_use]
pub fn oklab_to_srgb_hex(l: f64, a: f64, b: f64) -> String {
    let (x, y, z) = crate::oklab::oklab_to_xyz(l, a, b);
    // XYZ (D65) -> linear sRGB, IEC 61966-2-1.
    let lr = 3.240_969_941_904_521 * x - 1.537_383_177_570_093 * y - 0.498_610_760_293_003 * z;
    let lg = -0.969_243_636_280_87 * x + 1.875_967_501_507_72 * y + 0.041_555_057_407_175 * z;
    let lb = 0.055_630_079_696_993 * x - 0.203_976_958_888_976 * y + 1.056_971_514_242_878 * z;
    let encode = |channel: f64| -> u8 {
        let clamped = channel.clamp(0.0, 1.0);
        let gamma = if clamped <= 0.003_130_8 {
            12.92 * clamped
        } else {
            1.055 * clamped.powf(1.0 / 2.4) - 0.055
        };
        (gamma * 255.0).round() as u8
    };
    format!("#{:02x}{:02x}{:02x}", encode(lr), encode(lg), encode(lb))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rdp_keeps_endpoints_and_drops_collinear_middles() {
        let points: Vec<(f64, f64)> = (0..100).map(|index| (f64::from(index), 0.0)).collect();
        let simplified = rdp_simplify(&points, 0.01);
        assert_eq!(simplified.len(), 2);
        assert_eq!(simplified[0], (0.0, 0.0));
        assert_eq!(simplified[1], (99.0, 0.0));
    }

    #[test]
    fn rdp_preserves_corners() {
        let points = vec![(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)];
        let simplified = rdp_simplify(&points, 0.1);
        assert_eq!(simplified.len(), 3);
    }

    #[test]
    fn greedy_order_visits_all_paths_once() {
        let paths = vec![
            SvgPath { width_mm: 0.3, points: vec![(10.0, 10.0), (11.0, 10.0)] },
            SvgPath { width_mm: 0.3, points: vec![(0.0, 0.0), (1.0, 0.0)] },
            SvgPath { width_mm: 0.3, points: vec![(1.2, 0.1), (2.0, 0.4)] },
        ];
        let order = greedy_path_order(&paths);
        assert_eq!(order.len(), 3);
        assert_eq!(order[0], 1, "closest to origin first");
        assert_eq!(order[1], 2, "then its neighbour");
    }

    #[test]
    fn oklab_white_is_near_ffffff() {
        let hex = oklab_to_srgb_hex(1.0, 0.0, 0.0);
        assert_eq!(hex, "#ffffff");
    }
}
