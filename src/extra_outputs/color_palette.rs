//! Color palette card — extracts dominant colors via k-means clustering in
//! OKLab space, then renders a luxury SVG palette card with hex values and
//! evocative astronomy-inspired color names.

use crate::render::{
    ChannelLevels, SpectralRenderSettings, SpectralScene, render_final_frame_spectral,
};
use std::io::Write;
use tracing::info;

const NUM_COLORS: usize = 7;
const KMEANS_ITERATIONS: usize = 20;
const SAMPLE_STRIDE: usize = 8;

pub fn generate_color_palette(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    seed: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Generating color palette card...");

    let image = render_final_frame_spectral(scene, levels, settings)
        .map_err(|e| format!("Palette render failed: {}", e))?;

    let pixels = sample_pixels(&image);
    let centroids = kmeans_oklab(&pixels, NUM_COLORS, KMEANS_ITERATIONS);
    let mut colors: Vec<LabColor> = centroids.into_iter().filter(|c| c.l > 0.02).collect();
    colors.sort_by(|a, b| a.l.partial_cmp(&b.l).unwrap_or(std::cmp::Ordering::Equal));

    if colors.is_empty() {
        colors.push(LabColor { l: 0.1, a: 0.0, b: 0.0 });
    }

    let svg = build_palette_svg(&colors, seed);
    let mut file = std::fs::File::create(output_path)?;
    file.write_all(svg.as_bytes())?;
    info!("   Saved color palette => {}", output_path);
    Ok(())
}

#[derive(Clone, Copy)]
struct LabColor {
    l: f64,
    a: f64,
    b: f64,
}

fn sample_pixels(image: &image::ImageBuffer<image::Rgb<u16>, Vec<u16>>) -> Vec<LabColor> {
    let (w, h) = image.dimensions();
    let mut pixels = Vec::new();

    for y in (0..h).step_by(SAMPLE_STRIDE) {
        for x in (0..w).step_by(SAMPLE_STRIDE) {
            let px = image.get_pixel(x, y);
            let r = px[0] as f64 / 65535.0;
            let g = px[1] as f64 / 65535.0;
            let b = px[2] as f64 / 65535.0;
            let lab = srgb_to_oklab(r, g, b);
            if lab.l > 0.01 {
                pixels.push(lab);
            }
        }
    }
    pixels
}

fn srgb_to_oklab(r: f64, g: f64, b: f64) -> LabColor {
    let to_linear = |v: f64| {
        if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
    };
    let lr = to_linear(r);
    let lg = to_linear(g);
    let lb = to_linear(b);

    let l = 0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb;
    let m = 0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb;
    let s = 0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb;

    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();

    LabColor {
        l: 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        a: 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        b: 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    }
}

fn oklab_to_srgb(c: &LabColor) -> (f64, f64, f64) {
    let l_ = c.l + 0.3963377774 * c.a + 0.2158037573 * c.b;
    let m_ = c.l - 0.1055613458 * c.a - 0.0638541728 * c.b;
    let s_ = c.l - 0.0894841775 * c.a - 1.2914855480 * c.b;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    let r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
    let g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
    let b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;

    let to_srgb = |v: f64| -> f64 {
        let c = v.clamp(0.0, 1.0);
        if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
    };
    (to_srgb(r), to_srgb(g), to_srgb(b))
}

fn kmeans_oklab(pixels: &[LabColor], k: usize, iterations: usize) -> Vec<LabColor> {
    if pixels.is_empty() {
        return vec![LabColor { l: 0.0, a: 0.0, b: 0.0 }; k];
    }

    let n = pixels.len();
    let step = (n / k).max(1);
    let mut centroids: Vec<LabColor> = (0..k).map(|i| pixels[(i * step) % n]).collect();

    for _ in 0..iterations {
        let mut sums = vec![(0.0f64, 0.0f64, 0.0f64); k];
        let mut counts = vec![0usize; k];

        for px in pixels {
            let mut best_dist = f64::MAX;
            let mut best_idx = 0;
            for (i, c) in centroids.iter().enumerate() {
                let dl = px.l - c.l;
                let da = px.a - c.a;
                let db = px.b - c.b;
                let d = dl * dl + da * da + db * db;
                if d < best_dist {
                    best_dist = d;
                    best_idx = i;
                }
            }
            sums[best_idx].0 += px.l;
            sums[best_idx].1 += px.a;
            sums[best_idx].2 += px.b;
            counts[best_idx] += 1;
        }

        for i in 0..k {
            if counts[i] > 0 {
                let n = counts[i] as f64;
                centroids[i] = LabColor {
                    l: sums[i].0 / n,
                    a: sums[i].1 / n,
                    b: sums[i].2 / n,
                };
            }
        }
    }
    centroids
}

fn color_to_hex(c: &LabColor) -> String {
    let (r, g, b) = oklab_to_srgb(c);
    format!("#{:02X}{:02X}{:02X}", (r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

fn name_color(c: &LabColor) -> &'static str {
    let (r, g, b) = oklab_to_srgb(c);
    let hue = (g - b).atan2(r - 0.5 * g - 0.5 * b).to_degrees().rem_euclid(360.0);

    if c.l < 0.15 { return "Deep Void"; }
    if c.l > 0.85 { return "Stellar White"; }

    match hue as u32 {
        0..=29 => "Solar Flare",
        30..=59 => "Accretion Gold",
        60..=89 => "Pulsar Lime",
        90..=149 => "Nebula Teal",
        150..=209 => "Event Horizon",
        210..=259 => "Cosmic Indigo",
        260..=299 => "Nebula Violet",
        300..=339 => "Chrono Rose",
        _ => "Aurora Ember",
    }
}

fn build_palette_svg(colors: &[LabColor], seed: &str) -> String {
    let card_w = 700;
    let card_h = 220;
    let swatch_w = 70;
    let swatch_h = 100;
    let swatch_gap = 12;
    let total_swatches_w = colors.len() * swatch_w + (colors.len().saturating_sub(1)) * swatch_gap;
    let start_x = (card_w - total_swatches_w) / 2;
    let swatch_y = 70;

    let seed_display = seed.strip_prefix("0x").unwrap_or(seed);
    let seed_short = if seed_display.len() > 16 {
        format!("{}...{}", &seed_display[..8], &seed_display[seed_display.len() - 6..])
    } else {
        seed_display.to_string()
    };

    let mut swatches = String::new();
    for (i, color) in colors.iter().enumerate() {
        let x = start_x + i * (swatch_w + swatch_gap);
        let hex = color_to_hex(color);
        let name = name_color(color);

        swatches.push_str(&format!(
            concat!(
                "  <rect x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\" rx=\"6\" fill=\"{hex}\"/>\n",
                "  <text x=\"{tx}\" y=\"{ny}\" text-anchor=\"middle\" fill=\"#F0EDFF\" ",
                "font-family=\"monospace\" font-size=\"8\">{name}</text>\n",
                "  <text x=\"{tx}\" y=\"{hy}\" text-anchor=\"middle\" fill=\"#8B7BAA\" ",
                "font-family=\"monospace\" font-size=\"7\">{hex}</text>\n",
            ),
            x = x, y = swatch_y, w = swatch_w, h = swatch_h, hex = hex,
            tx = x + swatch_w / 2, ny = swatch_y + swatch_h + 16, name = name,
            hy = swatch_y + swatch_h + 28,
        ));
    }

    format!(
        r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {cw} {ch}" width="{cw}" height="{ch}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0D0521"/>
      <stop offset="100%" stop-color="#1A0B3E"/>
    </linearGradient>
    <linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#6C3CE1"/>
      <stop offset="100%" stop-color="#00D4AA"/>
    </linearGradient>
  </defs>
  <rect width="{cw}" height="{ch}" fill="url(#bg)" rx="12"/>
  <rect x="24" y="24" width="{bar_w}" height="3" fill="url(#accent)" rx="1.5"/>
  <text x="350" y="52" text-anchor="middle" fill="#F0EDFF" font-family="monospace"
        font-size="11" letter-spacing="3">DOMINANT PALETTE</text>
  <text x="350" y="64" text-anchor="middle" fill="#6C3CE1" font-family="monospace"
        font-size="8" letter-spacing="1">0x{seed}</text>
{swatches}  <rect x="24" y="{bot}" width="{bar_w}" height="3" fill="url(#accent)" rx="1.5" opacity="0.5"/>
  <text x="350" y="{foot}" text-anchor="middle" fill="#4A3A6A" font-family="monospace"
        font-size="7">{nc} colors · OKLab k-means extraction</text>
</svg>"##,
        cw = card_w, ch = card_h, bar_w = card_w - 48,
        seed = seed_short, swatches = swatches,
        bot = card_h - 30, foot = card_h - 12,
        nc = colors.len(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srgb_to_oklab_black() {
        let c = srgb_to_oklab(0.0, 0.0, 0.0);
        assert!(c.l.abs() < 0.01);
    }

    #[test]
    fn test_oklab_roundtrip() {
        let lab = srgb_to_oklab(0.5, 0.3, 0.7);
        let (r, g, b) = oklab_to_srgb(&lab);
        assert!((r - 0.5).abs() < 0.02);
        assert!((g - 0.3).abs() < 0.02);
        assert!((b - 0.7).abs() < 0.02);
    }

    #[test]
    fn test_kmeans_returns_k_centroids() {
        let pixels = vec![
            LabColor { l: 0.5, a: 0.1, b: 0.0 },
            LabColor { l: 0.2, a: -0.1, b: 0.05 },
            LabColor { l: 0.8, a: 0.0, b: -0.1 },
        ];
        let centroids = kmeans_oklab(&pixels, 2, 5);
        assert_eq!(centroids.len(), 2);
    }

    #[test]
    fn test_name_color_coverage() {
        let dark = LabColor { l: 0.05, a: 0.0, b: 0.0 };
        assert_eq!(name_color(&dark), "Deep Void");

        let bright = LabColor { l: 0.9, a: 0.0, b: 0.0 };
        assert_eq!(name_color(&bright), "Stellar White");
    }

    #[test]
    fn test_color_to_hex_format() {
        let c = LabColor { l: 0.5, a: 0.0, b: 0.0 };
        let hex = color_to_hex(&c);
        assert!(hex.starts_with('#'));
        assert_eq!(hex.len(), 7);
    }

    #[test]
    fn test_build_palette_svg_valid() {
        let colors = vec![
            LabColor { l: 0.3, a: 0.1, b: 0.0 },
            LabColor { l: 0.6, a: -0.05, b: 0.1 },
        ];
        let svg = build_palette_svg(&colors, "0xDEAD");
        assert!(svg.contains("<svg"));
        assert!(svg.contains("DOMINANT PALETTE"));
    }
}
