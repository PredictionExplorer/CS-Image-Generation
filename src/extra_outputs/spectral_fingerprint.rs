//! Spectral fingerprint — a radial polar-area chart of the 64-bin spectral
//! distribution, rendered as SVG.  Each wedge is colored by its wavelength
//! and sized by accumulated energy, producing a unique "color DNA" mark.

use crate::render::context::RenderContext;
use crate::render::velocity_hdr;
use crate::render::{
    SpectralRenderSettings, SpectralScene, accumulate_spectral_steps,
    apply_energy_density_shift, default_accumulation_backend,
};
use crate::spectrum::{NUM_BINS, wavelength_nm_for_bin};
use std::io::Write;
use tracing::info;

pub fn generate_spectral_fingerprint(
    scene: SpectralScene<'_>,
    settings: SpectralRenderSettings<'_>,
    seed: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Generating spectral fingerprint...");

    let bin_energies = compute_bin_energies(scene, settings);
    let svg = build_fingerprint_svg(&bin_energies, seed);

    let mut file = std::fs::File::create(output_path)?;
    file.write_all(svg.as_bytes())?;
    info!("   Saved spectral fingerprint => {}", output_path);
    Ok(())
}

fn compute_bin_energies(scene: SpectralScene<'_>, settings: SpectralRenderSettings<'_>) -> [f32; NUM_BINS] {
    let resolved = settings.resolved_config;
    let width = resolved.width;
    let height = resolved.height;
    let ctx = RenderContext::new(width, height, scene.positions, settings.aspect_correction);
    let mut accum_spd = vec![[0.0f32; NUM_BINS]; ctx.pixel_count()];

    let dt = crate::render::constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);

    accumulate_spectral_steps(
        &mut accum_spd,
        None,
        scene,
        &ctx,
        &velocity_calc,
        0,
        scene.step_count(),
        settings.render_config.hdr_scale,
        default_accumulation_backend(),
    );

    apply_energy_density_shift(&mut accum_spd);

    let mut totals = [0.0f32; NUM_BINS];
    for pixel_spd in &accum_spd {
        for (bin, &energy) in pixel_spd.iter().enumerate() {
            totals[bin] += energy;
        }
    }
    totals
}

fn wavelength_to_hex(nm: f64) -> String {
    let (r, g, b) = wavelength_to_rgb_approx(nm);
    format!("#{:02X}{:02X}{:02X}", (r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

fn wavelength_to_rgb_approx(lambda: f64) -> (f64, f64, f64) {
    let (r, g, b) = if (380.0..440.0).contains(&lambda) {
        (-(lambda - 440.0) / 60.0, 0.0, 1.0)
    } else if (440.0..490.0).contains(&lambda) {
        (0.0, (lambda - 440.0) / 50.0, 1.0)
    } else if (490.0..510.0).contains(&lambda) {
        (0.0, 1.0, -(lambda - 510.0) / 20.0)
    } else if (510.0..580.0).contains(&lambda) {
        ((lambda - 510.0) / 70.0, 1.0, 0.0)
    } else if (580.0..645.0).contains(&lambda) {
        (1.0, -(lambda - 645.0) / 65.0, 0.0)
    } else if (645.0..=700.0).contains(&lambda) {
        (1.0, 0.0, 0.0)
    } else {
        (0.0, 0.0, 0.0)
    };

    let factor = if (380.0..420.0).contains(&lambda) {
        0.3 + 0.7 * (lambda - 380.0) / 40.0
    } else if (420.0..645.0).contains(&lambda) {
        1.0
    } else if (645.0..=700.0).contains(&lambda) {
        0.3 + 0.7 * (700.0 - lambda) / 55.0
    } else {
        0.0
    };

    let gamma = |v: f64| v.powf(0.8);
    (gamma(r * factor), gamma(g * factor), gamma(b * factor))
}

fn build_fingerprint_svg(energies: &[f32; NUM_BINS], seed: &str) -> String {
    let max_energy = energies
        .iter()
        .map(|&e| e as f64)
        .fold(0.0f64, f64::max)
        .max(1e-10);

    let cx = 300.0f64;
    let cy = 300.0f64;
    let max_radius = 240.0f64;
    let inner_radius = 50.0f64;
    let angle_per_bin = std::f64::consts::TAU / NUM_BINS as f64;

    let mut wedges = String::new();
    let mut labels = String::new();

    for (bin, &energy) in energies.iter().enumerate().take(NUM_BINS) {
        let nm = wavelength_nm_for_bin(bin);
        let color = wavelength_to_hex(nm);
        let normalized = energy as f64 / max_energy;
        let outer_r = inner_radius + (max_radius - inner_radius) * normalized;

        let start_angle = bin as f64 * angle_per_bin - std::f64::consts::FRAC_PI_2;
        let end_angle = start_angle + angle_per_bin;
        let gap = 0.015;

        let x1_inner = cx + inner_radius * (start_angle + gap).cos();
        let y1_inner = cy + inner_radius * (start_angle + gap).sin();
        let x2_inner = cx + inner_radius * (end_angle - gap).cos();
        let y2_inner = cy + inner_radius * (end_angle - gap).sin();
        let x1_outer = cx + outer_r * (start_angle + gap).cos();
        let y1_outer = cy + outer_r * (start_angle + gap).sin();
        let x2_outer = cx + outer_r * (end_angle - gap).cos();
        let y2_outer = cy + outer_r * (end_angle - gap).sin();

        let large_arc = if angle_per_bin - 2.0 * gap > std::f64::consts::PI { 1 } else { 0 };

        wedges.push_str(&format!(
            "  <path d=\"M {x1i:.1} {y1i:.1} A {ir:.1} {ir:.1} 0 {la} 1 {x2i:.1} {y2i:.1} \
             L {x2o:.1} {y2o:.1} A {or:.1} {or:.1} 0 {la} 0 {x1o:.1} {y1o:.1} Z\" \
             fill=\"{color}\" opacity=\"0.85\"/>\n",
            x1i = x1_inner, y1i = y1_inner, ir = inner_radius,
            la = large_arc, x2i = x2_inner, y2i = y2_inner,
            x2o = x2_outer, y2o = y2_outer, or = outer_r,
            x1o = x1_outer, y1o = y1_outer, color = color,
        ));

        let label_r = max_radius + 18.0;
        let mid_angle = start_angle + angle_per_bin / 2.0;
        let lx = cx + label_r * mid_angle.cos();
        let ly = cy + label_r * mid_angle.sin();
        let rotation = mid_angle.to_degrees() + if mid_angle.cos() < 0.0 { 180.0 } else { 0.0 };
        let anchor = if mid_angle.cos() < 0.0 { "end" } else { "start" };

        labels.push_str(&format!(
            "  <text x=\"{lx:.1}\" y=\"{ly:.1}\" text-anchor=\"{anchor}\" \
             transform=\"rotate({rot:.1},{lx:.1},{ly:.1})\" \
             fill=\"#8B7BAA\" font-family=\"monospace\" font-size=\"8\">{nm:.0}nm</text>\n",
            lx = lx, ly = ly, anchor = anchor, rot = rotation, nm = nm,
        ));
    }

    let seed_display = seed.strip_prefix("0x").unwrap_or(seed);
    let seed_short = if seed_display.len() > 12 {
        format!("{}..{}", &seed_display[..6], &seed_display[seed_display.len() - 4..])
    } else {
        seed_display.to_string()
    };

    format!(
        r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 600" width="600" height="600">
  <defs>
    <radialGradient id="bgGrad" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#1A0B3E"/>
      <stop offset="100%" stop-color="#0D0521"/>
    </radialGradient>
  </defs>
  <rect width="600" height="600" fill="url(#bgGrad)" rx="16"/>
  <circle cx="{cx}" cy="{cy}" r="{ir}" fill="none" stroke="#2A1854" stroke-width="1"/>
  <circle cx="{cx}" cy="{cy}" r="{mr}" fill="none" stroke="#2A1854" stroke-width="0.5" stroke-dasharray="4,4"/>
{wedges}{labels}  <text x="{cx}" y="{cy_off}" text-anchor="middle" fill="#F0EDFF" font-family="monospace"
        font-size="9" letter-spacing="1">0x{seed}</text>
  <text x="{cx}" y="28" text-anchor="middle" fill="#8B7BAA" font-family="monospace"
        font-size="10" letter-spacing="3">SPECTRAL FINGERPRINT</text>
  <text x="{cx}" y="586" text-anchor="middle" fill="#4A3A6A" font-family="monospace"
        font-size="8">64-bin SPD · 380–700 nm</text>
</svg>"##,
        cx = cx, cy = cy, ir = inner_radius, mr = max_radius,
        wedges = wedges, labels = labels,
        cy_off = cy + 4.0, seed = seed_short,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavelength_to_hex_visible() {
        let hex = wavelength_to_hex(550.0);
        assert!(hex.starts_with('#'));
        assert_eq!(hex.len(), 7);
    }

    #[test]
    fn test_build_fingerprint_svg_contains_structure() {
        let mut energies = [0.0; NUM_BINS];
        for (i, v) in energies.iter_mut().enumerate() {
            *v = (i + 1) as f32;
        }
        let svg = build_fingerprint_svg(&energies, "0xCAFEBABE");
        assert!(svg.contains("<svg"));
        assert!(svg.contains("SPECTRAL FINGERPRINT"));
        assert!(svg.contains("CAFEB"));
        assert!(svg.contains("path d="));
    }

    #[test]
    fn test_build_fingerprint_svg_uniform_energy() {
        let energies = [1.0; NUM_BINS];
        let svg = build_fingerprint_svg(&energies, "0xDEAD");
        assert!(svg.contains("<path"));
    }

    #[test]
    fn test_build_fingerprint_svg_zero_energy() {
        let energies = [0.0; NUM_BINS];
        let svg = build_fingerprint_svg(&energies, "0x0");
        assert!(svg.contains("<svg"));
    }
}
