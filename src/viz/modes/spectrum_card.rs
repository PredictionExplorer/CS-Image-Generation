//! V05 `spectrum-card` -- Stellar Classification Card.
//!
//! The whole image's SPD integrated into one emission spectrum and rendered
//! like an observatory classification card: a luminous slit-spectrum
//! photograph over archival-cream stock, the same data as a hairline curve
//! beneath it, and the three bodies' emission lobes marked as ticks.
//! (Typeset labels arrive with `text.rs`; the JSON sidecar carries the data.)

use crate::error::Result;
use crate::oklab::{linear_srgb_to_oklab, oklab_to_linear_rec2020, oklch_to_oklab};
use crate::spectrum::{NUM_BINS, wavelength_nm_for_bin, wavelength_to_rgb};
use crate::viz::VizMode;
use crate::viz::VizPhase;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::display::encode_linear_rec2020_png16;
use crate::viz::common::raster::{Rgb64, draw_line};
use crate::viz::common::spd::aggregate_spectrum;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use tracing::warn;

/// Card width at final quality (3:2 poster).
const CARD_WIDTH: u32 = 3456;
/// Card height at final quality.
const CARD_HEIGHT: u32 = 2304;
/// Spectrum display exponent (compresses dynamic range).
const BAND_EXPONENT: f64 = 0.6;

/// The spectrum-card mode.
pub struct SpectrumCard;

/// Interpolated aggregate spectrum value at a wavelength.
fn spectrum_at(spectrum: &[f64; NUM_BINS], nm: f64) -> f64 {
    let bin_f = crate::spectral_constants::wavelength_to_bin(nm);
    let base = bin_f.floor() as usize;
    let next = (base + 1).min(NUM_BINS - 1);
    let frac = bin_f - base as f64;
    spectrum[base] * (1.0 - frac) + spectrum[next] * frac
}

/// Invert the rendered hue-to-wavelength relationship by nearest match.
fn wavelength_for_hue(target_hue: f64) -> f64 {
    let mut best = (f64::INFINITY, 550.0);
    let mut nm = 400.0;
    while nm <= 690.0 {
        let (r, g, b) = wavelength_to_rgb(nm);
        let (_, a, lab_b) = linear_srgb_to_oklab(r, g, b);
        let hue = lab_b.atan2(a).to_degrees().rem_euclid(360.0);
        let mut diff = (hue - target_hue).abs();
        if diff > 180.0 {
            diff = 360.0 - diff;
        }
        if diff < best.0 {
            best = (diff, nm);
        }
        nm += 1.0;
    }
    best.1
}

impl VizMode for SpectrumCard {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("spectrum-card").expect("spectrum-card is in the catalog")
    }

    fn phase(&self) -> VizPhase {
        VizPhase::Spd
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let Some(buffer) = ctx.accum_spd else {
            warn!("spectrum-card skipped: SPD buffer unavailable");
            return Ok(());
        };
        let spectrum = aggregate_spectrum(buffer);
        let peak = spectrum.iter().copied().fold(1e-12f64, f64::max);

        let width = ctx.quality.scale_dim(CARD_WIDTH);
        let height = ctx.quality.scale_dim(CARD_HEIGHT);
        let w = width as usize;
        let h = height as usize;

        // Archival-cream stock.
        let cream = {
            let (l, a, b) = oklch_to_oklab(0.94, 0.02, 95.0);
            let (r, g, bl) = oklab_to_linear_rec2020(l, a, b);
            (r.max(0.0), g.max(0.0), bl.max(0.0))
        };
        let mut pixels: Vec<Rgb64> = vec![cream; w * h];

        // Layout bands (fractions of height).
        let band_top = h * 16 / 100;
        let band_bottom = h * 52 / 100;
        let curve_top = h * 60 / 100;
        let curve_baseline = h * 88 / 100;
        let margin = w / 24;
        let plot_width = w - 2 * margin;

        let nm_at = |x: usize| 380.0 + 320.0 * (x as f64 / (plot_width - 1) as f64);

        // Slit-spectrum photograph: dark field with the luminous slit.
        for y in band_top..band_bottom {
            let band_center = f64::from((band_top + band_bottom) as u32) * 0.5;
            let band_half = f64::from((band_bottom - band_top) as u32) * 0.5;
            let vignette =
                (1.0 - ((y as f64 - band_center) / band_half).powi(2)).clamp(0.0, 1.0).powf(0.6);
            for x in 0..plot_width {
                let nm = nm_at(x);
                let strength = (spectrum_at(&spectrum, nm) / peak).powf(BAND_EXPONENT);
                let (r, g, b) = wavelength_to_rgb(nm);
                let field = 0.012; // near-black photographic field
                pixels[y * w + margin + x] = (
                    field + r * strength * vignette,
                    field + g * strength * vignette,
                    field + b * strength * vignette,
                );
            }
        }

        // Hairline curve of the same data.
        let ink = {
            let (l, a, b) = oklch_to_oklab(0.28, 0.03, 60.0);
            let (r, g, bl) = oklab_to_linear_rec2020(l, a, b);
            (r.max(0.0), g.max(0.0), bl.max(0.0))
        };
        let curve_height = (curve_baseline - curve_top) as f64;
        let mut previous: Option<(f32, f32)> = None;
        for x in 0..plot_width {
            let nm = nm_at(x);
            let value = (spectrum_at(&spectrum, nm) / peak).powf(BAND_EXPONENT);
            let point =
                ((margin + x) as f32, (curve_baseline as f64 - value * curve_height) as f32);
            if let Some(prev) = previous {
                draw_line(&mut pixels, w, h, prev, point, ink, 2.2, 0.92);
            }
            previous = Some(point);
        }
        // Baseline rule.
        draw_line(
            &mut pixels,
            w,
            h,
            (margin as f32, curve_baseline as f32),
            ((margin + plot_width) as f32, curve_baseline as f32),
            ink,
            1.2,
            0.6,
        );

        // Body emission-lobe ticks under the photograph band.
        let mut lobes = Vec::new();
        for body in 0..3 {
            let (l, a, b) = ctx.mean_color(body);
            let hue = b.atan2(a).to_degrees().rem_euclid(360.0);
            let nm = wavelength_for_hue(hue);
            lobes.push((body, nm));
            let x = margin as f64 + (nm - 380.0) / 320.0 * (plot_width - 1) as f64;
            let (tr, tg, tb) = oklab_to_linear_rec2020(l.max(0.45), a, b);
            draw_line(
                &mut pixels,
                w,
                h,
                (x as f32, (band_bottom + h / 100) as f32),
                (x as f32, (band_bottom + h * 4 / 100) as f32),
                (tr.max(0.0), tg.max(0.0), tb.max(0.0)),
                3.0,
                1.0,
            );
        }

        let image = encode_linear_rec2020_png16(&pixels, width, height);
        sink.save_png16(&image, "card.png")?;

        let meta = serde_json::json!({
            "bins_nm": (0..NUM_BINS).map(wavelength_nm_for_bin).collect::<Vec<_>>(),
            "spectrum": spectrum.to_vec(),
            "body_lobes_nm": lobes
                .iter()
                .map(|&(body, nm)| serde_json::json!({ "body": body, "nm": nm }))
                .collect::<Vec<_>>(),
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("spectrum.json", &json, "data")?;
        Ok(())
    }
}
