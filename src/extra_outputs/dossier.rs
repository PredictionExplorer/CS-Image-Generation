//! Collector's dossier — a multi-page SVG document serving as a luxury
//! certificate of authenticity and scientific profile, suitable for
//! high-quality print production.  Generated as a set of SVG pages
//! (one file per page) to avoid external PDF dependencies.

use crate::render::randomizable_config::ResolvedEffectConfig;
use crate::sim::TrajectoryResult;
use std::io::Write;
use tracing::info;

pub struct DossierData<'a> {
    pub seed: &'a str,
    pub output_name: &'a str,
    pub result: &'a TrajectoryResult,
    pub config: &'a ResolvedEffectConfig,
    pub num_sims: usize,
    pub num_steps: usize,
}

pub fn generate_dossier(
    data: &DossierData<'_>,
    output_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Generating collector's dossier...");

    std::fs::create_dir_all(output_dir)?;

    let seed = data.seed.strip_prefix("0x").unwrap_or(data.seed);
    let survival_rate =
        (data.num_sims - data.result.discarded_count) as f64 / data.num_sims as f64 * 100.0;
    let effects = collect_active_effects(data.config);

    let page1 = build_certificate_page(seed, data);
    let path1 = format!("{}/certificate.svg", output_dir);
    write_svg(&path1, &page1)?;

    let page2 = build_scientific_page(seed, data, survival_rate);
    let path2 = format!("{}/scientific_profile.svg", output_dir);
    write_svg(&path2, &page2)?;

    let page3 = build_effects_page(seed, &effects, data);
    let path3 = format!("{}/effect_chain.svg", output_dir);
    write_svg(&path3, &page3)?;

    info!("   Saved 3 dossier pages => {}/", output_dir);
    Ok(())
}

fn write_svg(path: &str, content: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = std::fs::File::create(path)?;
    file.write_all(content.as_bytes())?;
    Ok(())
}

fn build_certificate_page(seed: &str, data: &DossierData<'_>) -> String {
    let seed_short = if seed.len() > 20 {
        format!("{}...{}", &seed[..10], &seed[seed.len() - 10..])
    } else {
        seed.to_string()
    };

    format!(
        r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 595 842" width="595" height="842">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0D0521"/>
      <stop offset="100%" stop-color="#1A0B3E"/>
    </linearGradient>
    <linearGradient id="gold" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#C4A35A"/>
      <stop offset="50%" stop-color="#E8D5A3"/>
      <stop offset="100%" stop-color="#C4A35A"/>
    </linearGradient>
  </defs>
  <rect width="595" height="842" fill="url(#bg)"/>
  <rect x="30" y="30" width="535" height="782" fill="none" stroke="url(#gold)" stroke-width="2" rx="4"/>
  <rect x="36" y="36" width="523" height="770" fill="none" stroke="url(#gold)" stroke-width="0.5" rx="2"/>
  <text x="297" y="120" text-anchor="middle" fill="url(#gold)"
        font-family="'Cormorant Garamond',Georgia,serif" font-size="36" font-weight="300"
        letter-spacing="0.1em">CERTIFICATE</text>
  <text x="297" y="155" text-anchor="middle" fill="url(#gold)"
        font-family="'Cormorant Garamond',Georgia,serif" font-size="18" font-weight="300"
        letter-spacing="0.2em">OF AUTHENTICITY</text>
  <line x1="180" y1="180" x2="415" y2="180" stroke="url(#gold)" stroke-width="0.5"/>
  <text x="297" y="240" text-anchor="middle" fill="#F0EDFF"
        font-family="'Cormorant Garamond',Georgia,serif" font-size="16" font-weight="300">
    This document certifies that the digital artwork identified below</text>
  <text x="297" y="265" text-anchor="middle" fill="#F0EDFF"
        font-family="'Cormorant Garamond',Georgia,serif" font-size="16" font-weight="300">
    is an authentic Cosmic Signature NFT, deterministically generated</text>
  <text x="297" y="290" text-anchor="middle" fill="#F0EDFF"
        font-family="'Cormorant Garamond',Georgia,serif" font-size="16" font-weight="300">
    from the unique seed below via SHA3-256 seeded simulation.</text>
  <text x="297" y="360" text-anchor="middle" fill="#8B7BAA"
        font-family="monospace" font-size="10" letter-spacing="0.15em">SEED</text>
  <text x="297" y="390" text-anchor="middle" fill="#00D4AA"
        font-family="monospace" font-size="18">0x{seed}</text>
  <line x1="140" y1="420" x2="455" y2="420" stroke="#2A1854" stroke-width="0.5"/>
  <text x="297" y="470" text-anchor="middle" fill="#8B7BAA"
        font-family="monospace" font-size="10" letter-spacing="0.15em">WEIGHTED BORDA SCORE</text>
  <text x="297" y="510" text-anchor="middle" fill="#00D4AA"
        font-family="monospace" font-size="28">{borda:.3}</text>
  <text x="297" y="545" text-anchor="middle" fill="#8B7BAA"
        font-family="monospace" font-size="10">Selected #{sel} from {nsims} candidates</text>
  <line x1="140" y1="575" x2="455" y2="575" stroke="#2A1854" stroke-width="0.5"/>
  <text x="297" y="620" text-anchor="middle" fill="#8B7BAA"
        font-family="monospace" font-size="10" letter-spacing="0.15em">RESOLUTION</text>
  <text x="297" y="648" text-anchor="middle" fill="#F0EDFF"
        font-family="monospace" font-size="14">{width} x {height} · 16-bit RGB</text>
  <text x="297" y="710" text-anchor="middle" fill="#8B7BAA"
        font-family="monospace" font-size="10" letter-spacing="0.15em">PIPELINE</text>
  <text x="297" y="738" text-anchor="middle" fill="#F0EDFF"
        font-family="monospace" font-size="11">SHA3-256 → Yoshida-4 → Borda → Spectral 16-bin</text>
  <text x="297" y="800" text-anchor="middle" fill="#4A3A6A"
        font-family="monospace" font-size="8">COSMIC SIGNATURE · cosmicsignature.com</text>
</svg>"##,
        seed = seed_short,
        borda = data.result.total_score_weighted,
        sel = data.result.selected_index,
        nsims = data.num_sims,
        width = data.config.width,
        height = data.config.height,
    )
}

fn build_scientific_page(seed: &str, data: &DossierData<'_>, survival_rate: f64) -> String {
    let seed_short = if seed.len() > 16 {
        format!("{}...", &seed[..12])
    } else {
        seed.to_string()
    };

    format!(
        r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 595 842" width="595" height="842">
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
  <rect width="595" height="842" fill="url(#bg)"/>
  <rect x="30" y="30" width="535" height="4" fill="url(#accent)" rx="2"/>
  <text x="297" y="75" text-anchor="middle" fill="#F0EDFF"
        font-family="'Cormorant Garamond',Georgia,serif" font-size="24" font-weight="300"
        letter-spacing="0.08em">Scientific Profile</text>
  <text x="297" y="100" text-anchor="middle" fill="#6C3CE1"
        font-family="monospace" font-size="10" letter-spacing="0.1em">0x{seed}</text>
  <rect x="30" y="120" width="535" height="1" fill="#2A1854"/>
  <text x="60" y="165" fill="#8B7BAA" font-family="monospace" font-size="10"
        letter-spacing="0.12em">ORBITAL METRICS</text>
  <text x="60" y="200" fill="#F0EDFF" font-family="monospace" font-size="12">
    Chaos Score           {chaos:.6}</text>
  <text x="60" y="225" fill="#F0EDFF" font-family="monospace" font-size="12">
    Equilateralness       {equil:.6}</text>
  <text x="60" y="250" fill="#F0EDFF" font-family="monospace" font-size="12">
    Chaos Borda Points    {chaos_pts}</text>
  <text x="60" y="275" fill="#F0EDFF" font-family="monospace" font-size="12">
    Equil Borda Points    {equil_pts}</text>
  <text x="60" y="300" fill="#00D4AA" font-family="monospace" font-size="14">
    Weighted Borda Score  {borda:.3}</text>
  <rect x="30" y="330" width="535" height="1" fill="#2A1854"/>
  <text x="60" y="370" fill="#8B7BAA" font-family="monospace" font-size="10"
        letter-spacing="0.12em">SIMULATION PARAMETERS</text>
  <text x="60" y="405" fill="#F0EDFF" font-family="monospace" font-size="12">
    Total Candidates      {nsims}</text>
  <text x="60" y="430" fill="#F0EDFF" font-family="monospace" font-size="12">
    Discarded (escaped)   {discarded}</text>
  <text x="60" y="455" fill="#F0EDFF" font-family="monospace" font-size="12">
    Survival Rate         {surv:.1}%</text>
  <text x="60" y="480" fill="#F0EDFF" font-family="monospace" font-size="12">
    Integration Steps     {nsteps}</text>
  <text x="60" y="505" fill="#F0EDFF" font-family="monospace" font-size="12">
    Selected Index        #{sel}</text>
  <rect x="30" y="535" width="535" height="1" fill="#2A1854"/>
  <text x="60" y="575" fill="#8B7BAA" font-family="monospace" font-size="10"
        letter-spacing="0.12em">INTEGRATOR</text>
  <text x="60" y="610" fill="#F0EDFF" font-family="monospace" font-size="12">
    Method                4th-order Yoshida Symplectic</text>
  <text x="60" y="635" fill="#F0EDFF" font-family="monospace" font-size="12">
    Color Space           OKLab (perceptual)</text>
  <text x="60" y="660" fill="#F0EDFF" font-family="monospace" font-size="12">
    Spectral Bins         16 (380–700 nm)</text>
  <text x="60" y="685" fill="#F0EDFF" font-family="monospace" font-size="12">
    Orbit Selection       Borda rank aggregation</text>
  <text x="60" y="710" fill="#F0EDFF" font-family="monospace" font-size="12">
    Scoring               FFT chaos + equilateralness</text>
  <rect x="30" y="806" width="535" height="4" fill="url(#accent)" rx="2" opacity="0.5"/>
  <text x="297" y="830" text-anchor="middle" fill="#4A3A6A"
        font-family="monospace" font-size="8">COSMIC SIGNATURE · SCIENTIFIC PROFILE · PAGE 2</text>
</svg>"##,
        seed = seed_short,
        chaos = data.result.chaos,
        equil = data.result.equilateralness,
        chaos_pts = data.result.chaos_pts,
        equil_pts = data.result.equil_pts,
        borda = data.result.total_score_weighted,
        nsims = data.num_sims,
        discarded = data.result.discarded_count,
        surv = survival_rate,
        nsteps = data.num_steps,
        sel = data.result.selected_index,
    )
}

fn build_effects_page(seed: &str, effects: &[&str], data: &DossierData<'_>) -> String {
    let seed_short = if seed.len() > 16 {
        format!("{}...", &seed[..12])
    } else {
        seed.to_string()
    };

    let mut effects_svg = String::new();
    for (i, effect) in effects.iter().enumerate() {
        let y = 200 + i * 30;
        effects_svg.push_str(&format!(
            concat!(
                "  <circle cx=\"70\" cy=\"{}\" r=\"4\" fill=\"#6C3CE1\"/>\n",
                "  <text x=\"84\" y=\"{}\" fill=\"#F0EDFF\" font-family=\"monospace\" font-size=\"12\">{}</text>\n",
            ),
            y - 4, y, effect,
        ));
    }

    let rarity = compute_combined_rarity(data.config);

    format!(
        r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 595 842" width="595" height="842">
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
  <rect width="595" height="842" fill="url(#bg)"/>
  <rect x="30" y="30" width="535" height="4" fill="url(#accent)" rx="2"/>
  <text x="297" y="75" text-anchor="middle" fill="#F0EDFF"
        font-family="'Cormorant Garamond',Georgia,serif" font-size="24" font-weight="300"
        letter-spacing="0.08em">Effect Chain</text>
  <text x="297" y="100" text-anchor="middle" fill="#6C3CE1"
        font-family="monospace" font-size="10" letter-spacing="0.1em">0x{seed}</text>
  <rect x="30" y="120" width="535" height="1" fill="#2A1854"/>
  <text x="60" y="165" fill="#8B7BAA" font-family="monospace" font-size="10"
        letter-spacing="0.12em">ACTIVE POST-PROCESSING EFFECTS ({count})</text>
{effects}
  <rect x="30" y="{div_y}" width="535" height="1" fill="#2A1854"/>
  <text x="60" y="{rarity_label_y}" fill="#8B7BAA" font-family="monospace" font-size="10"
        letter-spacing="0.12em">COMBINED EFFECT RARITY</text>
  <text x="60" y="{rarity_val_y}" fill="#00D4AA" font-family="monospace" font-size="22">{rarity:.4}%</text>
  <text x="60" y="{rarity_note_y}" fill="#4A3A6A" font-family="monospace" font-size="9">
    Probability of this exact combination of enabled effects</text>
  <rect x="30" y="806" width="535" height="4" fill="url(#accent)" rx="2" opacity="0.5"/>
  <text x="297" y="830" text-anchor="middle" fill="#4A3A6A"
        font-family="monospace" font-size="8">COSMIC SIGNATURE · EFFECT CHAIN · PAGE 3</text>
</svg>"##,
        seed = seed_short,
        count = effects.len(),
        effects = effects_svg,
        div_y = 220 + effects.len() * 30,
        rarity_label_y = 260 + effects.len() * 30,
        rarity_val_y = 300 + effects.len() * 30,
        rarity_note_y = 330 + effects.len() * 30,
        rarity = rarity * 100.0,
    )
}

fn compute_combined_rarity(config: &ResolvedEffectConfig) -> f64 {
    let effects_and_probs: &[(&dyn Fn(&ResolvedEffectConfig) -> bool, f64)] = &[
        (&|c| c.enable_micro_contrast, 0.85),
        (&|c| c.enable_color_grade, 0.60),
        (&|c| c.enable_glow, 0.55),
        (&|c| c.enable_edge_luminance, 0.55),
        (&|c| c.enable_fine_texture, 0.45),
        (&|c| c.enable_aether, 0.35),
        (&|c| c.enable_bloom, 0.28),
        (&|c| c.enable_champleve, 0.25),
        (&|c| c.enable_opalescence, 0.25),
        (&|c| c.enable_chromatic_bloom, 0.20),
        (&|c| c.enable_gradient_map, 0.18),
        (&|c| c.enable_atmospheric_depth, 0.18),
        (&|c| c.enable_perceptual_blur, 0.05),
    ];

    let mut probability = 1.0f64;
    for (check, base_prob) in effects_and_probs {
        if check(config) {
            probability *= base_prob;
        } else {
            probability *= 1.0 - base_prob;
        }
    }
    probability
}

fn collect_active_effects(config: &ResolvedEffectConfig) -> Vec<&'static str> {
    let mut effects = Vec::new();
    if config.enable_bloom { effects.push("DoG Bloom"); }
    if config.enable_glow { effects.push("Glow Enhancement"); }
    if config.enable_chromatic_bloom { effects.push("Chromatic Bloom"); }
    if config.enable_perceptual_blur { effects.push("Perceptual Blur"); }
    if config.enable_micro_contrast { effects.push("Micro Contrast"); }
    if config.enable_gradient_map { effects.push("Gradient Map"); }
    if config.enable_color_grade { effects.push("Color Grade"); }
    if config.enable_opalescence { effects.push("Opalescence"); }
    if config.enable_champleve { effects.push("Champlevé"); }
    if config.enable_aether { effects.push("Aether Weave"); }
    if config.enable_edge_luminance { effects.push("Edge Luminance"); }
    if config.enable_atmospheric_depth { effects.push("Atmospheric Depth"); }
    if config.enable_fine_texture { effects.push("Fine Texture"); }
    if config.nebula_strength > 0.0 { effects.push("Nebula Clouds"); }
    effects
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_combined_rarity_all_off() {
        let config = make_test_config(false);
        let rarity = compute_combined_rarity(&config);
        assert!(rarity > 0.0);
        assert!(rarity < 1.0);
    }

    #[test]
    fn test_compute_combined_rarity_all_on() {
        let config = make_test_config(true);
        let rarity = compute_combined_rarity(&config);
        assert!(rarity > 0.0);
        assert!(rarity < 0.001, "all effects on should be very rare: {}", rarity);
    }

    fn make_test_config(all_on: bool) -> ResolvedEffectConfig {
        ResolvedEffectConfig {
            width: 1920, height: 1080,
            enable_bloom: all_on, enable_glow: all_on, enable_chromatic_bloom: all_on,
            enable_perceptual_blur: all_on, enable_micro_contrast: all_on,
            enable_gradient_map: all_on, enable_color_grade: all_on,
            enable_champleve: all_on, enable_aether: all_on, enable_opalescence: all_on,
            enable_edge_luminance: all_on, enable_atmospheric_depth: all_on,
            enable_fine_texture: all_on, nebula_strength: 0.0,
            blur_strength: 0.0, blur_radius_scale: 0.0, blur_core_brightness: 0.0,
            dog_strength: 0.0, dog_sigma_scale: 0.0, dog_ratio: 0.0,
            glow_strength: 0.0, glow_threshold: 0.0, glow_radius_scale: 0.0,
            glow_sharpness: 0.0, glow_saturation_boost: 0.0,
            chromatic_bloom_strength: 0.0, chromatic_bloom_radius_scale: 0.0,
            chromatic_bloom_separation_scale: 0.0, chromatic_bloom_threshold: 0.0,
            perceptual_blur_strength: 0.0, color_grade_strength: 0.0,
            vignette_strength: 0.0, vignette_softness: 0.0, vibrance: 0.0,
            clarity_strength: 0.0, tone_curve_strength: 0.0,
            gradient_map_strength: 0.0, gradient_map_hue_preservation: 0.0, gradient_map_palette: 0,
            opalescence_strength: 0.0, opalescence_scale: 0.0, opalescence_layers: 0,
            champleve_flow_alignment: 0.0, champleve_interference_amplitude: 0.0,
            champleve_rim_intensity: 0.0, champleve_rim_warmth: 0.0, champleve_interior_lift: 0.0,
            aether_flow_alignment: 0.0, aether_scattering_strength: 0.0,
            aether_iridescence_amplitude: 0.0, aether_caustic_strength: 0.0,
            micro_contrast_strength: 0.0, micro_contrast_radius: 0,
            edge_luminance_strength: 0.0, edge_luminance_threshold: 0.0,
            edge_luminance_brightness_boost: 0.0,
            atmospheric_depth_strength: 0.0, atmospheric_desaturation: 0.0,
            atmospheric_darkening: 0.0,
            atmospheric_fog_color_r: 0.0, atmospheric_fog_color_g: 0.0, atmospheric_fog_color_b: 0.0,
            fine_texture_strength: 0.0, fine_texture_scale: 0.0, fine_texture_contrast: 0.0,
            hdr_scale: 0.0, clip_black: 0.0, clip_white: 1.0,
            nebula_octaves: 0, nebula_base_frequency: 0.0,
        }
    }
}
