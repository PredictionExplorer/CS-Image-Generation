//! V58 `ephemeris-poster` -- The Almanac Page.
//!
//! The master still surrounded by typeset data in vintage astronomical-
//! almanac style: identity, orbital elements, the top close encounters,
//! palette genome swatches, and a colophon -- every number traceable to
//! the run's own data, set in tabular Plex over archival cream.

use crate::error::Result;
use crate::oklab::{oklab_to_linear_rec2020, oklab_to_oklch};
use crate::sim::G;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::display::encode_linear_rec2020_png16;
use crate::viz::common::raster::{Rgb64, draw_line};
use crate::viz::common::style::{Paper, ink_color, margin, paper_color, type_px};
use crate::viz::common::text::{Align, Face, TextStyle, draw_text, measure};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use tracing::warn;

/// Display gamma for decoding the master still.
const DISPLAY_GAMMA: f64 = 2.2;
/// Poster re-tone in EV for paper context.
const RETONE_EV: f64 = 0.15;
/// Encounters listed in the table.
const ENCOUNTERS: usize = 5;

/// Format a mission-elapsed-time step count.
pub(crate) fn format_met(step: usize) -> String {
    format!("T+{:09.1}", step as f64 / 10.0)
}

/// Format a physical quantity with tabular-friendly fixed precision.
pub(crate) fn format_quantity(value: f64) -> String {
    if value == 0.0 {
        return "0.000000".to_string();
    }
    let magnitude = value.abs();
    if (1e-3..1e6).contains(&magnitude) { format!("{value:.6}") } else { format!("{value:.4e}") }
}

/// Total mechanical energy of the initial configuration.
pub(crate) fn total_energy(bodies: &[crate::sim::Body]) -> f64 {
    let mut energy = 0.0;
    for (index, body) in bodies.iter().enumerate() {
        energy += 0.5 * body.mass * body.velocity.norm_squared();
        for other in bodies.iter().skip(index + 1) {
            let distance = (body.position - other.position).norm().max(1e-12);
            energy -= G * body.mass * other.mass / distance;
        }
    }
    energy
}

/// Magnitude of the initial total angular momentum.
pub(crate) fn total_angular_momentum(bodies: &[crate::sim::Body]) -> f64 {
    bodies
        .iter()
        .fold(nalgebra::Vector3::zeros(), |acc, body| {
            acc + body.mass * body.position.cross(&body.velocity)
        })
        .norm()
}

/// The ephemeris-poster mode.
pub struct EphemerisPoster;

impl VizMode for EphemerisPoster {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("ephemeris-poster").expect("ephemeris-poster is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let poster_h = ctx.quality.scale_dim(3352).max(300) as usize;
        let poster_w = poster_h * 2 / 3;
        let paper = Paper::ArchivalCream;
        let stock = paper_color(paper);
        let ink = ink_color(paper);
        let mut poster = vec![stock; poster_w * poster_h];
        let page_margin = margin(poster_w, poster_h);
        let short = poster_w.min(poster_h);

        // --- Master render on the upper 62%, re-toned for paper.
        let image_region_h = poster_h * 62 / 100;
        let master_path = format!("{}/images/source/master.png", ctx.seed_dir);
        if let Ok(Ok(decoded)) =
            image::ImageReader::open(&master_path).map(image::ImageReader::decode)
        {
            let master = decoded.into_rgb16();
            let source_w = master.width() as usize;
            let source_h = master.height() as usize;
            let target_w = poster_w - 2 * page_margin;
            let target_h = (target_w * source_h / source_w).min(image_region_h - 2 * page_margin);
            let target_w = target_h * source_w / source_h;
            let origin_x = (poster_w - target_w) / 2;
            let origin_y = page_margin;
            let gain = 2.0_f64.powf(RETONE_EV);
            let raw = master.as_raw();
            for y in 0..target_h {
                let sy = y * source_h / target_h;
                for x in 0..target_w {
                    let sx = x * source_w / target_w;
                    let base = (sy * source_w + sx) * 3;
                    let decode = |v: u16| (f64::from(v) / 65535.0).powf(DISPLAY_GAMMA);
                    poster[(origin_y + y) * poster_w + origin_x + x] = (
                        decode(raw[base]) * gain,
                        decode(raw[base + 1]) * gain,
                        decode(raw[base + 2]) * gain,
                    );
                }
            }
        } else {
            warn!("ephemeris-poster: master.png unavailable; plate renders data only");
        }

        // --- Data plate.
        let plate_top = image_region_h + page_margin / 2;
        let heading_px = type_px(short, 2);
        let body_px = type_px(short, 0);
        let seed_px = type_px(short, 4);
        let rule = |buffer: &mut Vec<Rgb64>, y: f64| {
            draw_line(
                buffer,
                poster_w,
                poster_h,
                (page_margin as f32, y as f32),
                ((poster_w - page_margin) as f32, y as f32),
                ink,
                0.8,
                0.8,
            );
        };
        let heading_style =
            TextStyle { tracking: 0.14, ..TextStyle::caption(Face::Sans, heading_px, ink) };
        let data_style = TextStyle { tabular: true, ..TextStyle::new(Face::Mono, body_px, ink) };
        let label_style = TextStyle::caption(Face::Sans, body_px, ink);

        let mut cursor = plate_top as f64;
        // IDENT: the seed in large numerals.
        let seed_style = TextStyle {
            align: Align::Center,
            tabular: true,
            ..TextStyle::new(Face::Sans, seed_px, ink)
        };
        cursor += seed_px;
        draw_text(
            &mut poster,
            poster_w,
            poster_h,
            poster_w as f64 / 2.0,
            cursor,
            &seed_style,
            &format!("0x{}", ctx.seed_hex.to_uppercase()),
        );
        cursor += seed_px * 0.5;
        rule(&mut poster, cursor);
        cursor += heading_px * 1.6;

        // ELEMENTS: masses, energy, angular momentum.
        draw_text(
            &mut poster,
            poster_w,
            poster_h,
            page_margin as f64,
            cursor,
            &heading_style,
            "Elements",
        );
        cursor += body_px * 1.7;
        let energy = total_energy(ctx.bodies);
        let momentum = total_angular_momentum(ctx.bodies);
        let elements: Vec<(String, String)> = vec![
            ("M1".into(), format_quantity(ctx.bodies[0].mass)),
            ("M2".into(), format_quantity(ctx.bodies[1].mass)),
            ("M3".into(), format_quantity(ctx.bodies[2].mass)),
            ("E TOTAL".into(), format_quantity(energy)),
            ("L TOTAL".into(), format_quantity(momentum)),
            ("STEPS".into(), format!("{}", ctx.step_count())),
        ];
        let column_w = (poster_w - 2 * page_margin) as f64 / 3.0;
        for (index, (label, value)) in elements.iter().enumerate() {
            let column = index % 3;
            let row = index / 3;
            let x = page_margin as f64 + column_w * column as f64;
            let y = cursor + row as f64 * body_px * 1.6;
            draw_text(&mut poster, poster_w, poster_h, x, y, &label_style, label);
            draw_text(&mut poster, poster_w, poster_h, x + column_w * 0.38, y, &data_style, value);
        }
        cursor += body_px * 1.6 * 2.0 + body_px;
        rule(&mut poster, cursor);
        cursor += heading_px * 1.6;

        // ENCOUNTERS: the five deepest approaches.
        draw_text(
            &mut poster,
            poster_w,
            poster_h,
            page_margin as f64,
            cursor,
            &heading_style,
            "Encounters",
        );
        cursor += body_px * 1.7;
        let mut deepest = ctx.events().periapses.clone();
        deepest.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        let speed_of = |pair: (usize, usize), step: usize| -> f64 {
            (ctx.kinematics().velocities[pair.0][step] - ctx.kinematics().velocities[pair.1][step])
                .norm()
        };
        for (row, approach) in deepest.iter().take(ENCOUNTERS).enumerate() {
            let y = cursor + row as f64 * body_px * 1.5;
            let line = format!(
                "{met}  P{a}{b}  R {r}  V {v}",
                met = format_met(approach.step),
                a = approach.pair.0 + 1,
                b = approach.pair.1 + 1,
                r = format_quantity(approach.distance),
                v = format_quantity(speed_of(approach.pair, approach.step)),
            );
            draw_text(&mut poster, poster_w, poster_h, page_margin as f64, y, &data_style, &line);
        }
        cursor += body_px * 1.5 * ENCOUNTERS as f64 + body_px * 0.5;
        rule(&mut poster, cursor);
        cursor += heading_px * 1.6;

        // PALETTE: swatches with OKLCh values.
        draw_text(
            &mut poster,
            poster_w,
            poster_h,
            page_margin as f64,
            cursor,
            &heading_style,
            "Palette",
        );
        cursor += body_px * 1.4;
        for body in 0..3 {
            let (l, a, b) = ctx.mean_color(body);
            let (light, chroma, hue) = oklab_to_oklch(l, a, b);
            let (r, g, blue) = oklab_to_linear_rec2020(l, a, b);
            let x = page_margin as f64 + column_w * body as f64;
            let swatch = body_px * 1.1;
            for sy in 0..swatch as usize {
                for sx in 0..(swatch * 2.2) as usize {
                    let px = x as usize + sx;
                    let py = cursor as usize + sy;
                    if px < poster_w && py < poster_h {
                        poster[py * poster_w + px] = (r.max(0.0), g.max(0.0), blue.max(0.0));
                    }
                }
            }
            draw_text(
                &mut poster,
                poster_w,
                poster_h,
                x + swatch * 2.6,
                cursor + swatch * 0.85,
                &data_style,
                &format!("L{light:.2} C{chroma:.3} H{hue:.0}"),
            );
        }
        cursor += body_px * 2.2;
        rule(&mut poster, cursor);

        // COLOPHON is the standard footer.
        crate::viz::common::style::draw_footer(
            &mut poster,
            poster_w,
            poster_h,
            ctx,
            "V58 EPHEMERIS",
            paper,
        );

        // Sanity: the seed string must fit the plate.
        let (seed_width, _, _) =
            measure(&seed_style, &format!("0x{}", ctx.seed_hex.to_uppercase()));
        if seed_width > (poster_w - 2 * page_margin) as f64 {
            warn!("ephemeris-poster: seed line overflows the plate width");
        }

        let image = encode_linear_rec2020_png16(&poster, poster_w as u32, poster_h as u32);
        sink.save_png16(&image, "ephemeris.png")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn formatters_are_deterministic_and_tabular_friendly() {
        assert_eq!(format_met(1_426), "T+0000142.6");
        assert_eq!(format_quantity(1.5), "1.500000");
        assert_eq!(format_quantity(1.5e9), "1.5000e9");
        assert_eq!(format_quantity(0.0), "0.000000");
    }

    #[test]
    fn two_body_circular_energy_is_negative_half_potential() {
        // Circular two-body orbit (third body negligible and distant):
        // each body circles the COM at r/2 with v = sqrt(G m / (2 r)), and
        // the virial theorem gives E = -G m^2 / (2 r).
        let m = 1.0;
        let r = 2.0;
        let v = (G * m / (2.0 * r)).sqrt();
        let bodies = vec![
            crate::sim::Body::new(m, Vector3::new(-r / 2.0, 0.0, 0.0), Vector3::new(0.0, -v, 0.0)),
            crate::sim::Body::new(m, Vector3::new(r / 2.0, 0.0, 0.0), Vector3::new(0.0, v, 0.0)),
            crate::sim::Body::new(1e-12, Vector3::new(1e6, 0.0, 0.0), Vector3::zeros()),
        ];
        let energy = total_energy(&bodies);
        let expected = -G * m * m / (2.0 * r);
        assert!((energy - expected).abs() < 1e-6, "virial energy mismatch: {energy} vs {expected}");
        assert!(total_angular_momentum(&bodies) > 0.0);
    }
}
