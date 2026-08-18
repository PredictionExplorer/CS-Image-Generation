//! The shared poster design language (master plan II.14): layout margins,
//! the modular type scale, the three paper stocks, palette-derived ink
//! accents, and the standard typeset footer every poster carries.

use crate::oklab::{oklab_to_linear_rec2020, oklab_to_oklch, oklch_to_oklab};
use crate::viz::common::raster::Rgb64;
use crate::viz::common::text::{Align, Face, TextStyle, draw_text};
use crate::viz::context::VizContext;

/// Reference short edge for the type scale (the default render height).
const REFERENCE_SHORT_EDGE: f64 = 2234.0;
/// Modular scale ratio.
const SCALE_RATIO: f64 = 1.333;
/// Base type size in pixels at the reference short edge.
const BASE_PX: f64 = 16.0;

/// Poster margin: 1/24 of the short edge (II.14).
#[must_use]
pub fn margin(width: usize, height: usize) -> usize {
    (width.min(height) / 24).max(8)
}

/// Modular type scale: `level` steps above (positive) or below (negative)
/// the 16 px base, scaled to the canvas's short edge.
#[must_use]
pub fn type_px(short_edge: usize, level: i32) -> f64 {
    BASE_PX * SCALE_RATIO.powi(level) * (short_edge as f64 / REFERENCE_SHORT_EDGE)
}

/// The three paper stocks (II.14).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Paper {
    /// The collection's native negative space.
    DeepBlack,
    /// Prussian-blue cyanotype stock.
    Cyanotype,
    /// Warm archival cream.
    ArchivalCream,
}

/// Linear Rec.2020 color of a paper stock.
#[must_use]
pub fn paper_color(paper: Paper) -> Rgb64 {
    let (l, a, b) = match paper {
        Paper::DeepBlack => (0.055, 0.0, 0.0),
        Paper::Cyanotype => oklch_to_oklab(0.28, 0.07, 250.0),
        Paper::ArchivalCream => oklch_to_oklab(0.94, 0.02, 95.0),
    };
    let (r, g, blue) = oklab_to_linear_rec2020(l, a, b);
    (r.max(0.0), g.max(0.0), blue.max(0.0))
}

/// Ink color that reads on a paper stock.
#[must_use]
pub fn ink_color(paper: Paper) -> Rgb64 {
    let (l, a, b) = match paper {
        Paper::DeepBlack => (0.80, 0.0, 0.0),
        Paper::Cyanotype => (0.92, -0.005, -0.01),
        Paper::ArchivalCream => oklch_to_oklab(0.30, 0.035, 65.0),
    };
    let (r, g, blue) = oklab_to_linear_rec2020(l, a, b);
    (r.max(0.0), g.max(0.0), blue.max(0.0))
}

/// Palette-derived accent: the dominant body hue lifted to L = 0.82.
#[must_use]
pub fn accent_color(ctx: &VizContext<'_>) -> Rgb64 {
    let mut best = (0.0f64, 0.0f64);
    for body in 0..3 {
        let (l, a, b) = ctx.mean_color(body);
        let (_, chroma, hue) = oklab_to_oklch(l, a, b);
        if chroma > best.0 {
            best = (chroma, hue);
        }
    }
    let (l, a, b) = oklch_to_oklab(0.82, best.0.clamp(0.05, 0.16), best.1);
    let (r, g, blue) = oklab_to_linear_rec2020(l, a, b);
    (r.max(0.0), g.max(0.0), blue.max(0.0))
}

/// The standard footer text: seed, mode id, resolution, crate version.
#[must_use]
pub fn footer_text(ctx: &VizContext<'_>, mode_id: &str, width: usize, height: usize) -> String {
    format!(
        "0x{seed} \u{b7} {mode} \u{b7} {width}\u{d7}{height} \u{b7} v{version}",
        seed = ctx.seed_hex.to_uppercase(),
        mode = mode_id,
        version = env!("CARGO_PKG_VERSION"),
    )
}

/// Typeset the standard footer centered in the bottom margin.
pub fn draw_footer(
    buffer: &mut [Rgb64],
    width: usize,
    height: usize,
    ctx: &VizContext<'_>,
    mode_id: &str,
    paper: Paper,
) {
    let footer_margin = margin(width, height);
    let px = type_px(width.min(height), -1);
    let style = TextStyle {
        align: Align::Center,
        opacity: 0.85,
        ..TextStyle::caption(Face::Sans, px, ink_color(paper))
    };
    let baseline = height as f64 - footer_margin as f64 * 0.5 + px * 0.35;
    draw_text(
        buffer,
        width,
        height,
        width as f64 / 2.0,
        baseline,
        &style,
        &footer_text(ctx, mode_id, width, height),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn margin_is_a_twentyfourth_of_the_short_edge() {
        assert_eq!(margin(3456, 2234), 2234 / 24);
        assert_eq!(margin(240, 480), 10);
    }

    #[test]
    fn type_scale_is_modular() {
        let base = type_px(2234, 0);
        assert!((base - 16.0).abs() < 1e-9);
        assert!((type_px(2234, 1) / base - SCALE_RATIO).abs() < 1e-9);
        assert!(type_px(1117, 0) < base);
    }

    #[test]
    fn papers_and_inks_contrast() {
        for paper in [Paper::DeepBlack, Paper::Cyanotype, Paper::ArchivalCream] {
            let stock = paper_color(paper);
            let ink = ink_color(paper);
            let stock_lum = stock.0 + stock.1 + stock.2;
            let ink_lum = ink.0 + ink.1 + ink.2;
            assert!(
                (stock_lum - ink_lum).abs() > 0.15,
                "{paper:?}: ink must contrast with the stock"
            );
        }
    }
}
