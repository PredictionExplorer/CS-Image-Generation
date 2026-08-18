//! V03 `spectral-centroid` -- What Color the Light Really Is.
//!
//! Per pixel: the energy-weighted mean wavelength becomes hue, and spectral
//! purity (one minus normalized spectral entropy) becomes chroma -- a map of
//! the light's true spectral identity. Regions that look white in the master
//! render split into their constituent wavelengths.

use crate::error::Result;
use crate::oklab::{
    linear_srgb_to_oklab, max_display_p3_chroma_for_lh, oklab_to_linear_rec2020, oklch_to_oklab,
};
use crate::spectrum::{NUM_BINS, wavelength_nm_for_bin, wavelength_to_rgb};
use crate::viz::VizMode;
use crate::viz::VizPhase;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::display::encode_linear_rec2020_png16;
use crate::viz::common::raster::Rgb64;
use crate::viz::common::spd;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use rayon::prelude::*;
use tracing::warn;

/// Pixels below this total energy stay pure black.
const ENERGY_FLOOR: f64 = 1e-6;
/// Chroma exponent applied to purity.
const PURITY_EXPONENT: f64 = 0.75;

/// The spectral-centroid mode.
pub struct SpectralCentroid;

/// Per-pixel centroid analysis: (mean wavelength nm, purity, total energy).
fn analyze(pixel: &[f64; NUM_BINS]) -> (f64, f64, f64) {
    let mut total = 0.0f64;
    let mut weighted = 0.0f64;
    for (bin, &energy) in pixel.iter().enumerate() {
        if energy > 0.0 {
            total += energy;
            weighted += energy * wavelength_nm_for_bin(bin);
        }
    }
    if total <= ENERGY_FLOOR {
        return (0.0, 0.0, 0.0);
    }
    let mean_wavelength = weighted / total;
    let mut entropy = 0.0f64;
    for &energy in pixel {
        if energy > 0.0 {
            let p = energy / total;
            entropy -= p * p.ln();
        }
    }
    let purity = (1.0 - entropy / (NUM_BINS as f64).ln()).clamp(0.0, 1.0);
    (mean_wavelength, purity, total)
}

/// `OkLab` hue angle of a rendered wavelength.
fn wavelength_hue(nm: f64) -> f64 {
    let (r, g, b) = wavelength_to_rgb(nm);
    let (_, a, lab_b) = linear_srgb_to_oklab(r, g, b);
    lab_b.atan2(a).to_degrees().rem_euclid(360.0)
}

impl VizMode for SpectralCentroid {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("spectral-centroid").expect("spectral-centroid is in the catalog")
    }

    fn phase(&self) -> VizPhase {
        VizPhase::Spd
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let Some(buffer) = ctx.accum_spd else {
            warn!("spectral-centroid skipped: SPD buffer unavailable");
            return Ok(());
        };
        let width = ctx.width as usize;
        let height = ctx.height as usize;

        // Pass 1: per-pixel (wavelength, purity, energy).
        let mut analysis: Vec<(f64, f64, f64)> = buffer.par_iter().map(analyze).collect();

        // 3x3 range-weighted smoothing on (wavelength, purity), guarded by
        // energy: keeps edges crisp while killing single-pixel hue speckle.
        let original = analysis.clone();
        analysis.par_iter_mut().enumerate().for_each(|(index, value)| {
            let (center_wl, _, center_energy) = original[index];
            if center_energy <= ENERGY_FLOOR {
                return;
            }
            let x = index % width;
            let y = index / width;
            let mut sum_wl = 0.0f64;
            let mut sum_purity = 0.0f64;
            let mut sum_weight = 0.0f64;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let nx = (x as i64 + dx).clamp(0, width as i64 - 1) as usize;
                    let ny = (y as i64 + dy).clamp(0, height as i64 - 1) as usize;
                    let (wl, purity, energy) = original[ny * width + nx];
                    if energy <= ENERGY_FLOOR {
                        continue;
                    }
                    let range_weight = (-((wl - center_wl) / 6.0).powi(2) * 0.5).exp();
                    sum_wl += wl * range_weight;
                    sum_purity += purity * range_weight;
                    sum_weight += range_weight;
                }
            }
            if sum_weight > 0.0 {
                value.0 = sum_wl / sum_weight;
                value.1 = sum_purity / sum_weight;
            }
        });

        // Energy normalization anchor: 99th percentile of a strided sample.
        let mut energy_sample: Vec<f64> = analysis
            .iter()
            .step_by(37)
            .map(|&(_, _, energy)| energy)
            .filter(|&energy| energy > ENERGY_FLOOR)
            .collect();
        energy_sample.sort_by(f64::total_cmp);
        let p99 = energy_sample
            .get(((energy_sample.len().saturating_sub(1)) as f64 * 0.99) as usize)
            .copied()
            .unwrap_or(1.0)
            .max(1e-9);

        // Pass 2: colorize.
        let centroid_pixels: Vec<Rgb64> = analysis
            .par_iter()
            .map(|&(wavelength, purity, energy)| {
                if energy <= ENERGY_FLOOR {
                    return (0.0, 0.0, 0.0);
                }
                let hue = wavelength_hue(wavelength);
                let lightness = 0.04 + 0.82 * (energy / p99).clamp(0.0, 1.0).powf(0.5);
                let chroma =
                    purity.powf(PURITY_EXPONENT) * max_display_p3_chroma_for_lh(lightness, hue);
                let (l, a, b) = oklch_to_oklab(lightness, chroma, hue);
                let (r, g, bl) = oklab_to_linear_rec2020(l, a, b);
                (r.max(0.0), g.max(0.0), bl.max(0.0))
            })
            .collect();
        sink.save_png16(
            &encode_linear_rec2020_png16(&centroid_pixels, ctx.width, ctx.height),
            "centroid.png",
        )?;

        // Purity as its own grayscale print.
        let purity_pixels: Vec<Rgb64> = analysis
            .par_iter()
            .map(|&(_, purity, energy)| {
                if energy <= ENERGY_FLOOR {
                    return (0.0, 0.0, 0.0);
                }
                let (l, a, b) = oklch_to_oklab(0.04 + 0.9 * purity, 0.0, 0.0);
                let (r, g, bl) = oklab_to_linear_rec2020(l, a, b);
                (r.max(0.0), g.max(0.0), bl.max(0.0))
            })
            .collect();
        sink.save_png16(
            &encode_linear_rec2020_png16(&purity_pixels, ctx.width, ctx.height),
            "purity.png",
        )?;

        let aggregate = spd::aggregate_spectrum(buffer);
        let (agg_wl, agg_purity, _) = analyze(&aggregate);
        let meta = serde_json::json!({
            "energy_floor": ENERGY_FLOOR,
            "purity_exponent": PURITY_EXPONENT,
            "image_mean_wavelength_nm": agg_wl,
            "image_purity": agg_purity,
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("centroid.json", &json, "data")?;
        Ok(())
    }
}
