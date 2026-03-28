//! Woven Æther post-effect for creating a flowing, volumetric tapestry of light.
//!
//! This effect transforms the rendered trajectories into a field of luminous,
//! anisotropic filaments. It uses a flow-warped distance field to create an
//! organic, woven texture, and then applies volumetric scattering, iridescent
//! color shifts, and negative space caustics for a deeply textured, ethereal look.

use super::PixelBuffer;
use crate::render::constants;
use rayon::prelude::*;

/// Configuration for the Woven Æther artistic effect.
#[derive(Clone, Debug)]
pub struct AetherConfig {
    pub filament_density: f64,
    pub flow_alignment: f64,
    pub scattering_strength: f64,
    pub scattering_falloff: f64,
    pub iridescence_amplitude: f64,
    pub iridescence_frequency: f64,
    pub caustic_strength: f64,
    pub caustic_softness: f64,
    pub luxury_mode: bool,
}

impl Default for AetherConfig {
    fn default() -> Self {
        Self {
            filament_density: constants::DEFAULT_AETHER_FILAMENT_DENSITY,
            flow_alignment: constants::DEFAULT_AETHER_FLOW_ALIGNMENT,
            scattering_strength: constants::DEFAULT_AETHER_SCATTERING_STRENGTH,
            scattering_falloff: constants::DEFAULT_AETHER_SCATTERING_FALLOFF,
            iridescence_amplitude: constants::DEFAULT_AETHER_IRIDESCENCE_AMPLITUDE,
            iridescence_frequency: constants::DEFAULT_AETHER_IRIDESCENCE_FREQUENCY,
            caustic_strength: constants::DEFAULT_AETHER_CAUSTIC_STRENGTH,
            caustic_softness: constants::DEFAULT_AETHER_CAUSTIC_SOFTNESS,
            luxury_mode: true,
        }
    }
}

use super::utils::hash2;

/// Computes the anisotropic Voronoi distance field, warped by flow direction.
#[inline]
fn anisotropic_voronoi(p: (f64, f64), flow_dir: f64, flow_strength: f64) -> f64 {
    let ix = p.0.floor();
    let iy = p.1.floor();
    let fx = p.0.fract();
    let fy = p.1.fract();

    let mut min_dist = f64::MAX;

    for j in -1..=1 {
        for i in -1..=1 {
            let cell_offset = (i as f64, j as f64);
            let point = hash2((ix + cell_offset.0, iy + cell_offset.1));
            let mut r = (cell_offset.0 + point.0 - fx, cell_offset.1 + point.1 - fy);

            // Anisotropic warp: stretch the distance calculation along the flow direction.
            let warp_factor = 1.0 - flow_strength;
            let dot_r_flow = r.0 * flow_dir.cos() + r.1 * flow_dir.sin();
            r.0 -= warp_factor * dot_r_flow * flow_dir.cos();
            r.1 -= warp_factor * dot_r_flow * flow_dir.sin();

            min_dist = min_dist.min(r.0 * r.0 + r.1 * r.1);
        }
    }
    min_dist.sqrt()
}

/// Applies the Woven Æther effect to a buffer of linear RGBA pixels.
pub fn apply_aether_weave(
    buffer: &mut PixelBuffer,
    width: usize,
    height: usize,
    config: &AetherConfig,
) {
    if buffer.is_empty() {
        return;
    }

    let gradients = super::utils::calculate_gradients(buffer, width, height);

    let cell_scale = (width as f64 * height as f64).sqrt() / config.filament_density.max(1.0);
    let inv_cell_scale = 1.0 / cell_scale;

    buffer.par_iter_mut().enumerate().for_each(|(idx, pixel)| {
        let (r, g, b, a) = *pixel;
        if a <= 0.0 {
            return;
        }
        let sr = r / a;
        let sg = g / a;
        let sb = b / a;

        let u = (idx % width) as f64 * inv_cell_scale;
        let v = (idx / width) as f64 * inv_cell_scale;

        let (gx, gy) = gradients[idx];
        let flow_dir = gy.atan2(gx);
        let flow_mag = (gx * gx + gy * gy).sqrt();

        let field_dist = anisotropic_voronoi((u, v), flow_dir, flow_mag * config.flow_alignment);
        let filament = (1.0 - field_dist).clamp(0.0, 1.0).powf(config.scattering_falloff);

        let base_lum = 0.2126 * sr + 0.7152 * sg + 0.0722 * sb;
        let scatter_lum = base_lum * filament * config.scattering_strength;

        let caustic = (filament - 0.7).max(0.0) * base_lum;
        let caustic_bleed = caustic.powf(config.caustic_softness) * config.caustic_strength;

        let interference =
            (config.iridescence_frequency * (v * 0.6 - u * 0.4 + flow_dir.cos())).sin();
        let spectral = interference * config.iridescence_amplitude;

        let mut final_r = sr + scatter_lum + caustic_bleed;
        let mut final_g = sg + scatter_lum + caustic_bleed;
        let mut final_b = sb + scatter_lum + caustic_bleed;

        if config.luxury_mode {
            // Luxury mode: purple/magenta iridescence
            final_r += spectral;
            final_g -= spectral * 0.3;
            final_b -= spectral;
        } else {
            // Non-luxury mode: rainbow iridescence based on position
            let hue_shift = (u * 2.1 + v * 3.4 + flow_dir).sin() * std::f64::consts::TAU; // Full hue range
            final_r += (hue_shift).cos() * spectral;
            final_g += (hue_shift + 2.09439).cos() * spectral; // +120°
            final_b += (hue_shift + 4.18879).cos() * spectral; // +240°
        }

        *pixel = (final_r.max(0.0) * a, final_g.max(0.0) * a, final_b.max(0.0) * a, a);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient_buffer(w: usize, h: usize) -> PixelBuffer {
        (0..w * h)
            .map(|i| {
                let t = i as f64 / (w * h) as f64;
                (t * 0.7, (1.0 - t) * 0.4, 0.25, 0.85)
            })
            .collect()
    }

    #[test]
    fn test_aether_preserves_buffer_length() {
        let config = AetherConfig::default();
        let mut buf = gradient_buffer(48, 36);
        let len = buf.len();
        apply_aether_weave(&mut buf, 48, 36, &config);
        assert_eq!(buf.len(), len);
    }

    #[test]
    fn test_aether_modifies_pixels() {
        let config = AetherConfig::default();
        let original = gradient_buffer(32, 24);
        let mut buf = original.clone();
        apply_aether_weave(&mut buf, 32, 24, &config);
        let changed = buf.iter().zip(original.iter()).filter(|(a, b)| {
            (a.0 - b.0).abs() > 1e-12 || (a.1 - b.1).abs() > 1e-12
        }).count();
        assert!(changed > 0, "aether weave should modify at least some pixels");
    }

    #[test]
    fn test_aether_all_black_stays_black() {
        let config = AetherConfig::default();
        let mut buf = vec![(0.0, 0.0, 0.0, 0.0); 16 * 16];
        apply_aether_weave(&mut buf, 16, 16, &config);
        for &(r, g, b, a) in &buf {
            assert_eq!(r, 0.0);
            assert_eq!(g, 0.0);
            assert_eq!(b, 0.0);
            assert_eq!(a, 0.0);
        }
    }

    #[test]
    fn test_aether_no_nan_in_output() {
        let config = AetherConfig::default();
        let mut buf = gradient_buffer(48, 32);
        apply_aether_weave(&mut buf, 48, 32, &config);
        for (i, &(r, g, b, a)) in buf.iter().enumerate() {
            assert!(!r.is_nan(), "R NaN at pixel {i}");
            assert!(!g.is_nan(), "G NaN at pixel {i}");
            assert!(!b.is_nan(), "B NaN at pixel {i}");
            assert!(!a.is_nan(), "A NaN at pixel {i}");
        }
    }

    #[test]
    fn test_aether_luxury_mode_differs_from_rainbow() {
        let luxury = AetherConfig { luxury_mode: true, ..AetherConfig::default() };
        let rainbow = AetherConfig { luxury_mode: false, ..AetherConfig::default() };
        let mut buf_l = gradient_buffer(24, 16);
        let mut buf_r = buf_l.clone();
        apply_aether_weave(&mut buf_l, 24, 16, &luxury);
        apply_aether_weave(&mut buf_r, 24, 16, &rainbow);
        assert_ne!(buf_l, buf_r, "luxury vs rainbow modes should produce different output");
    }
}
