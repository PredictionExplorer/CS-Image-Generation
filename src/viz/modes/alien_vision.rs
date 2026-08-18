//! V01 `alien-vision` -- The Same Light, Other Eyes.
//!
//! Re-renders the identical physical light field (the accumulated per-pixel
//! SPD) through five non-human observers: new receptor response curves are
//! integrated against the true spectra, so metamers split and structure
//! invisible to the human retina surfaces. Each variant carries its own
//! white balance and exposure.

use crate::error::Result;
use crate::oklab::{max_display_p3_chroma_for_lh, oklab_to_linear_rec2020, oklch_to_oklab};
use crate::render::context::PixelBuffer;
use crate::render::{ImageBuffer, Rgb};
use crate::spectrum::{NUM_BINS, spd_to_rgba, wavelength_nm_for_bin};
use crate::viz::VizMode;
use crate::viz::VizPhase;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::display::grade_auto_levels;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use rayon::prelude::*;
use tracing::warn;

/// Per-bin soft-saturation steepness (uniform; auto-levels set exposure).
const TONE_K: f64 = 1.0;
/// Strip panel downscale factor.
const STRIP_SCALE: u32 = 3;
/// Strip gutter width in pixels.
const STRIP_GUTTER: u32 = 16;

/// The alien-vision mode.
pub struct AlienVision;

/// A 64-bin Gaussian receptor response curve, peak-normalized.
fn receptor(center_nm: f64, sigma_nm: f64) -> [f64; NUM_BINS] {
    let mut curve = [0.0; NUM_BINS];
    for (bin, value) in curve.iter_mut().enumerate() {
        let lambda = wavelength_nm_for_bin(bin);
        let d = (lambda - center_nm) / sigma_nm;
        *value = (-0.5 * d * d).exp();
    }
    curve
}

/// One observer: three output-channel response curves plus a label.
struct Observer {
    name: &'static str,
    /// Response curves feeding the output R, G, B channels respectively.
    channels: [[f64; NUM_BINS]; 3],
}

/// Integrate one pixel's SPD through an observer (white-balanced).
#[inline]
fn integrate(spd: &[f64; NUM_BINS], observer: &Observer, white: &[f64; 3]) -> (f64, f64, f64, f64) {
    let mut out = [0.0f64; 3];
    let mut total = 0.0f64;
    for (bin, &energy) in spd.iter().enumerate() {
        if energy <= 1e-10 {
            continue;
        }
        let mapped = 1.0 - (-TONE_K * energy).exp();
        total += mapped;
        for (channel, response) in out.iter_mut().zip(observer.channels.iter()) {
            *channel += mapped * response[bin];
        }
    }
    if total <= 1e-10 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let brightness = 1.0 - (-total).exp();
    (
        (out[0] / white[0]) / total,
        (out[1] / white[1]) / total,
        (out[2] / white[2]) / total,
        brightness,
    )
}

/// White point: the observer's response to a flat unit SPD, per channel.
fn flat_white(observer: &Observer) -> [f64; 3] {
    let mapped = 1.0 - (-TONE_K).exp();
    let mut white = [1e-12f64; 3];
    for (slot, response) in white.iter_mut().zip(observer.channels.iter()) {
        *slot = (response.iter().map(|&weight| mapped * weight).sum::<f64>()
            / (NUM_BINS as f64 * mapped))
            .max(1e-12);
    }
    white
}

/// Render one trichromat-style observer variant as a linear RGBA buffer.
fn render_observer(spd: &[[f64; NUM_BINS]], observer: &Observer) -> PixelBuffer {
    let white = flat_white(observer);
    spd.par_iter().map(|pixel| integrate(pixel, observer, &white)).collect()
}

/// Mantis shrimp: 12 narrow bands, winner-take-all categorical hue wheel.
fn render_mantis(spd: &[[f64; NUM_BINS]]) -> PixelBuffer {
    let bands: Vec<[f64; NUM_BINS]> =
        (0..12).map(|band| receptor(390.0 + 300.0 * f64::from(band) / 11.0, 12.0)).collect();
    spd.par_iter()
        .map(|pixel| {
            let mut responses = [0.0f64; 12];
            let mut total = 0.0f64;
            for (bin, &energy) in pixel.iter().enumerate() {
                if energy <= 1e-10 {
                    continue;
                }
                let mapped = 1.0 - (-TONE_K * energy).exp();
                total += mapped;
                for (response, band) in responses.iter_mut().zip(bands.iter()) {
                    *response += mapped * band[bin];
                }
            }
            if total <= 1e-10 {
                return (0.0, 0.0, 0.0, 0.0);
            }
            let winner = responses
                .iter()
                .enumerate()
                .max_by(|lhs, rhs| lhs.1.total_cmp(rhs.1))
                .map_or(0, |(index, _)| index);
            let hue = f64::from(winner as u32) / 12.0 * 360.0;
            let brightness = 1.0 - (-total).exp();
            let lightness = 0.35 + 0.45 * brightness;
            let chroma = 0.75 * max_display_p3_chroma_for_lh(lightness, hue);
            let (l, a, b) = oklch_to_oklab(lightness, chroma, hue);
            let (r, g, bl) = oklab_to_linear_rec2020(l, a, b);
            (r.max(0.0), g.max(0.0), bl.max(0.0), brightness)
        })
        .collect()
}

/// Scotopic night vision: V-prime luminance in a moonlit blue ramp.
fn render_night(spd: &[[f64; NUM_BINS]]) -> PixelBuffer {
    let v_prime = receptor(507.0, 42.0);
    spd.par_iter()
        .map(|pixel| {
            let mut luminance = 0.0f64;
            for (bin, &energy) in pixel.iter().enumerate() {
                if energy > 1e-10 {
                    luminance += (1.0 - (-TONE_K * energy).exp()) * v_prime[bin];
                }
            }
            if luminance <= 1e-10 {
                return (0.0, 0.0, 0.0, 0.0);
            }
            let brightness = 1.0 - (-luminance * 2.2).exp();
            let lightness = 0.05 + 0.85 * brightness;
            let (l, a, b) = oklch_to_oklab(lightness, 0.035, 250.0);
            let (r, g, bl) = oklab_to_linear_rec2020(l, a, b);
            (r.max(0.0), g.max(0.0), bl.max(0.0), brightness)
        })
        .collect()
}

/// Thermal false color: energy-weighted mean bin as pseudo-temperature.
fn render_thermal(spd: &[[f64; NUM_BINS]]) -> PixelBuffer {
    spd.par_iter()
        .map(|pixel| {
            let mut total = 0.0f64;
            let mut weighted = 0.0f64;
            for (bin, &energy) in pixel.iter().enumerate() {
                if energy > 1e-10 {
                    total += energy;
                    weighted += energy * bin as f64;
                }
            }
            if total <= 1e-10 {
                return (0.0, 0.0, 0.0, 0.0);
            }
            let temperature = weighted / total / (NUM_BINS - 1) as f64;
            let brightness = 1.0 - (-total).exp();
            // Inferno-like path: deep violet -> red -> amber -> pale yellow.
            let lightness = 0.08 + 0.84 * temperature.powf(0.9);
            let hue = 300.0 - 270.0 * temperature;
            let chroma = max_display_p3_chroma_for_lh(lightness, hue)
                * (1.0 - (temperature - 0.85).max(0.0) * 4.0).clamp(0.2, 1.0);
            let (l, a, b) = oklch_to_oklab(lightness, chroma, hue);
            let (r, g, bl) = oklab_to_linear_rec2020(l, a, b);
            (r.max(0.0), g.max(0.0), bl.max(0.0), brightness)
        })
        .collect()
}

/// Human reference: the core spectral converter (for the strip's last panel).
fn render_human(spd: &[[f64; NUM_BINS]]) -> PixelBuffer {
    spd.par_iter().map(spd_to_rgba).collect()
}

/// Box-downscale a 16-bit image by an integer factor.
fn downscale(
    image: &ImageBuffer<Rgb<u16>, Vec<u16>>,
    factor: u32,
) -> ImageBuffer<Rgb<u16>, Vec<u16>> {
    let width = (image.width() / factor).max(1);
    let height = (image.height() / factor).max(1);
    let mut out = ImageBuffer::new(width, height);
    for (x, y, pixel) in out.enumerate_pixels_mut() {
        let mut sums = [0.0f64; 3];
        let mut count = 0.0f64;
        for dy in 0..factor {
            for dx in 0..factor {
                let sx = (x * factor + dx).min(image.width() - 1);
                let sy = (y * factor + dy).min(image.height() - 1);
                let source = image.get_pixel(sx, sy);
                for (sum, &value) in sums.iter_mut().zip(source.0.iter()) {
                    *sum += f64::from(value);
                }
                count += 1.0;
            }
        }
        *pixel = Rgb([
            (sums[0] / count).round() as u16,
            (sums[1] / count).round() as u16,
            (sums[2] / count).round() as u16,
        ]);
    }
    out
}

/// Horizontally concatenate panels with black gutters.
fn hstack(panels: &[ImageBuffer<Rgb<u16>, Vec<u16>>]) -> ImageBuffer<Rgb<u16>, Vec<u16>> {
    let height = panels.iter().map(ImageBuffer::height).max().unwrap_or(1);
    let width: u32 = panels.iter().map(ImageBuffer::width).sum::<u32>()
        + STRIP_GUTTER * (panels.len().saturating_sub(1)) as u32;
    let mut out = ImageBuffer::new(width, height);
    let mut cursor = 0u32;
    for panel in panels {
        for (x, y, pixel) in panel.enumerate_pixels() {
            out.put_pixel(cursor + x, y, *pixel);
        }
        cursor += panel.width() + STRIP_GUTTER;
    }
    out
}

impl VizMode for AlienVision {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("alien-vision").expect("alien-vision is in the catalog")
    }

    fn phase(&self) -> VizPhase {
        VizPhase::Spd
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let Some(spd) = ctx.accum_spd else {
            warn!("alien-vision skipped: SPD buffer unavailable");
            return Ok(());
        };
        let clip_black = ctx.settings.resolved_config.clip_black;
        let clip_white = ctx.settings.resolved_config.clip_white;

        let bee = Observer {
            name: "bee",
            // UV proxy -> blue channel, blue -> green, green -> red.
            channels: [receptor(540.0, 40.0), receptor(440.0, 30.0), receptor(395.0, 25.0)],
        };
        let dog = Observer {
            name: "dog",
            // Dichromat: L (555) drives warm channels, S (429) the blue one.
            channels: [receptor(555.0, 45.0), receptor(555.0, 45.0), receptor(429.0, 30.0)],
        };

        let mut panels = Vec::new();
        let mut order = Vec::new();
        let variants: Vec<(&'static str, PixelBuffer)> = vec![
            (bee.name, render_observer(spd, &bee)),
            (dog.name, render_observer(spd, &dog)),
            ("mantis", render_mantis(spd)),
            ("night", render_night(spd)),
            ("thermal", render_thermal(spd)),
        ];
        for (name, rgba) in variants {
            let image =
                grade_auto_levels(&rgba, ctx.width, ctx.height, clip_black, clip_white, 1.0);
            sink.save_png16(&image, &format!("{name}.png"))?;
            panels.push(downscale(&image, STRIP_SCALE));
            order.push(name);
        }

        // Human reference panel closes the strip.
        let human = grade_auto_levels(
            &render_human(spd),
            ctx.width,
            ctx.height,
            clip_black,
            clip_white,
            1.0,
        );
        panels.push(downscale(&human, STRIP_SCALE));
        order.push("human");

        sink.save_png16(&hstack(&panels), "strip.png")?;

        let meta = serde_json::json!({
            "strip_order": order,
            "receptors": {
                "bee_nm": [395.0, 440.0, 540.0],
                "dog_nm": [429.0, 555.0],
                "mantis_bands": 12,
                "night_peak_nm": 507.0,
            },
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("variants.json", &json, "data")?;
        Ok(())
    }
}
