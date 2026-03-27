//! Rarity report card — an SVG visualization showing how this NFT's
//! effect combination compares to the probability space, with individual
//! effect probabilities and the combined rarity percentile.

use crate::render::randomizable_config::ResolvedEffectConfig;
use crate::sim::TrajectoryResult;
use std::io::Write;
use tracing::info;

pub struct RarityReportData<'a> {
    pub seed: &'a str,
    pub result: &'a TrajectoryResult,
    pub config: &'a ResolvedEffectConfig,
    pub num_sims: usize,
}

pub fn generate_rarity_report(
    data: &RarityReportData<'_>,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Generating rarity report...");

    let effects = analyze_effects(data.config);
    let combined_rarity = compute_combined_probability(&effects);
    let svg = build_rarity_svg(data, &effects, combined_rarity);

    let mut file = std::fs::File::create(output_path)?;
    file.write_all(svg.as_bytes())?;
    info!("   Saved rarity report => {}", output_path);
    Ok(())
}

struct EffectAnalysis {
    name: &'static str,
    enabled: bool,
    base_probability: f64,
}

fn analyze_effects(config: &ResolvedEffectConfig) -> Vec<EffectAnalysis> {
    vec![
        EffectAnalysis { name: "Micro Contrast", enabled: config.enable_micro_contrast, base_probability: 0.85 },
        EffectAnalysis { name: "Color Grade", enabled: config.enable_color_grade, base_probability: 0.60 },
        EffectAnalysis { name: "Glow", enabled: config.enable_glow, base_probability: 0.55 },
        EffectAnalysis { name: "Edge Luminance", enabled: config.enable_edge_luminance, base_probability: 0.55 },
        EffectAnalysis { name: "Fine Texture", enabled: config.enable_fine_texture, base_probability: 0.45 },
        EffectAnalysis { name: "Aether", enabled: config.enable_aether, base_probability: 0.35 },
        EffectAnalysis { name: "DoG Bloom", enabled: config.enable_bloom, base_probability: 0.28 },
        EffectAnalysis { name: "Champlevé", enabled: config.enable_champleve, base_probability: 0.25 },
        EffectAnalysis { name: "Opalescence", enabled: config.enable_opalescence, base_probability: 0.25 },
        EffectAnalysis { name: "Chromatic Bloom", enabled: config.enable_chromatic_bloom, base_probability: 0.20 },
        EffectAnalysis { name: "Gradient Map", enabled: config.enable_gradient_map, base_probability: 0.18 },
        EffectAnalysis { name: "Atmospheric Depth", enabled: config.enable_atmospheric_depth, base_probability: 0.18 },
        EffectAnalysis { name: "Perceptual Blur", enabled: config.enable_perceptual_blur, base_probability: 0.05 },
    ]
}

fn compute_combined_probability(effects: &[EffectAnalysis]) -> f64 {
    effects.iter().fold(1.0, |acc, e| {
        if e.enabled { acc * e.base_probability } else { acc * (1.0 - e.base_probability) }
    })
}

fn build_rarity_svg(
    data: &RarityReportData<'_>,
    effects: &[EffectAnalysis],
    combined: f64,
) -> String {
    let seed = data.seed.strip_prefix("0x").unwrap_or(data.seed);
    let seed_short = if seed.len() > 16 {
        format!("{}...{}", &seed[..8], &seed[seed.len() - 6..])
    } else {
        seed.to_string()
    };

    let active_count = effects.iter().filter(|e| e.enabled).count();
    let card_h = 320 + effects.len() * 32;

    let mut bars = String::new();
    for (i, effect) in effects.iter().enumerate() {
        let y = 200 + i * 32;
        let bar_w = (effect.base_probability * 360.0) as i32;
        let fill = if effect.enabled { "#6C3CE1" } else { "#2A1854" };
        let text_fill = if effect.enabled { "#F0EDFF" } else { "#4A3A6A" };
        let status = if effect.enabled { "ON" } else { "—" };

        bars.push_str(&format!(
            concat!(
                "  <text x=\"45\" y=\"{y}\" fill=\"{tf}\" font-family=\"monospace\" font-size=\"10\">{name}</text>\n",
                "  <rect x=\"200\" y=\"{by}\" width=\"360\" height=\"16\" fill=\"#0D0521\" stroke=\"#2A1854\" rx=\"3\"/>\n",
                "  <rect x=\"200\" y=\"{by}\" width=\"{bw}\" height=\"16\" fill=\"{fill}\" opacity=\"0.7\" rx=\"3\"/>\n",
                "  <text x=\"570\" y=\"{y}\" text-anchor=\"end\" fill=\"{tf}\" font-family=\"monospace\" font-size=\"9\">{prob:.0}%</text>\n",
                "  <text x=\"590\" y=\"{y}\" fill=\"{tf}\" font-family=\"monospace\" font-size=\"9\">{status}</text>\n",
            ),
            y = y, tf = text_fill, name = effect.name,
            by = y - 12, bw = bar_w, fill = fill,
            prob = effect.base_probability * 100.0, status = status,
        ));
    }

    let summary_y = 220 + effects.len() * 32;

    format!(
        r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 {ch}" width="640" height="{ch}">
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
  <rect width="640" height="{ch}" fill="url(#bg)" rx="16"/>
  <rect x="24" y="24" width="592" height="4" fill="url(#accent)" rx="2"/>
  <text x="320" y="70" text-anchor="middle" fill="#F0EDFF" font-family="monospace" font-size="13"
        letter-spacing="3">RARITY REPORT</text>
  <text x="320" y="95" text-anchor="middle" fill="#6C3CE1" font-family="monospace" font-size="10"
        letter-spacing="1">0x{seed}</text>
  <rect x="24" y="115" width="592" height="1" fill="#2A1854"/>
  <text x="45" y="150" fill="#8B7BAA" font-family="monospace" font-size="9"
        letter-spacing="0.12em">EFFECT</text>
  <text x="350" y="150" text-anchor="middle" fill="#8B7BAA" font-family="monospace" font-size="9"
        letter-spacing="0.12em">BASE PROBABILITY</text>
  <text x="590" y="150" fill="#8B7BAA" font-family="monospace" font-size="9">STATE</text>
  <rect x="24" y="165" width="592" height="1" fill="#2A1854"/>
{bars}
  <rect x="24" y="{sy}" width="592" height="1" fill="#2A1854"/>
  <text x="45" y="{s1y}" fill="#8B7BAA" font-family="monospace" font-size="10"
        letter-spacing="0.12em">COMBINED RARITY</text>
  <text x="45" y="{s2y}" fill="#00D4AA" font-family="monospace" font-size="24">{rarity}</text>
  <text x="45" y="{s3y}" fill="#4A3A6A" font-family="monospace" font-size="9">
    {active} of {total} effects active · Borda score {borda:.3}</text>
  <rect x="24" y="{bot}" width="592" height="4" fill="url(#accent)" rx="2" opacity="0.5"/>
</svg>"##,
        ch = card_h,
        seed = seed_short,
        bars = bars,
        sy = summary_y,
        s1y = summary_y + 30,
        s2y = summary_y + 60,
        s3y = summary_y + 80,
        rarity = format_rarity(combined),
        active = active_count,
        total = effects.len(),
        borda = data.result.total_score_weighted,
        bot = card_h - 24,
    )
}

fn format_rarity(probability: f64) -> String {
    if probability < 0.0001 {
        format!("{:.6}%", probability * 100.0)
    } else if probability < 0.01 {
        format!("{:.4}%", probability * 100.0)
    } else {
        format!("{:.2}%", probability * 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_rarity_very_rare() {
        let s = format_rarity(0.00001);
        assert!(s.contains("0.001"));
    }

    #[test]
    fn test_format_rarity_common() {
        let s = format_rarity(0.5);
        assert!(s.contains("50"));
    }

    #[test]
    fn test_combined_probability_all_common() {
        let effects = vec![
            EffectAnalysis { name: "A", enabled: true, base_probability: 0.85 },
            EffectAnalysis { name: "B", enabled: true, base_probability: 0.60 },
        ];
        let p = compute_combined_probability(&effects);
        assert!((p - 0.51).abs() < 0.01);
    }

    #[test]
    fn test_combined_probability_none_enabled() {
        let effects = vec![
            EffectAnalysis { name: "A", enabled: false, base_probability: 0.85 },
            EffectAnalysis { name: "B", enabled: false, base_probability: 0.60 },
        ];
        let p = compute_combined_probability(&effects);
        assert!((p - 0.06).abs() < 0.01);
    }
}
