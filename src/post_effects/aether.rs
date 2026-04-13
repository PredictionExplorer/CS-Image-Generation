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
    /// Filament scale from image area (higher yields finer weave).
    pub filament_density: f64,
    /// How strongly the flow field warps the Voronoi distance.
    pub flow_alignment: f64,
    /// Strength of volumetric glow along bright filaments.
    pub scattering_strength: f64,
    /// Exponent shaping filament brightness falloff from the field.
    pub scattering_falloff: f64,
    /// Amplitude of iridescent colour modulation.
    pub iridescence_amplitude: f64,
    /// Spatial frequency of iridescence across the image.
    pub iridescence_frequency: f64,
    /// Strength of caustic highlights in negative space.
    pub caustic_strength: f64,
    /// Sharpness of the caustic response curve.
    pub caustic_softness: f64,
    /// When true, purple/magenta iridescence; when false, broader hue cycling.
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

        let base_lum = crate::render::constants::rec709_luminance(sr, sg, sb);
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

    #[test]
    fn test_default_config_uses_constants() {
        let cfg = AetherConfig::default();
        assert_eq!(cfg.filament_density, constants::DEFAULT_AETHER_FILAMENT_DENSITY);
        assert_eq!(cfg.scattering_strength, constants::DEFAULT_AETHER_SCATTERING_STRENGTH);
        assert!(cfg.luxury_mode);
    }

    #[test]
    fn test_anisotropic_voronoi_non_negative() {
        for i in 0..50 {
            let p = (i as f64 * 0.31, i as f64 * 0.47);
            let d = anisotropic_voronoi(p, 0.5, 0.3);
            assert!(d >= 0.0, "Voronoi distance must be >= 0 at {p:?}");
        }
    }

    #[test]
    fn test_empty_buffer_is_noop() {
        let mut buffer: Vec<(f64, f64, f64, f64)> = vec![];
        apply_aether_weave(&mut buffer, 0, 0, &AetherConfig::default());
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_preserves_alpha() {
        let w = 10;
        let h = 10;
        let mut buffer: Vec<(f64, f64, f64, f64)> =
            (0..w * h).map(|i| (0.5, 0.3, 0.2, 0.6 + (i as f64) * 0.002)).collect();
        let alphas: Vec<f64> = buffer.iter().map(|p| p.3).collect();
        apply_aether_weave(&mut buffer, w, h, &AetherConfig::default());
        for (i, (pixel, &orig_a)) in buffer.iter().zip(alphas.iter()).enumerate() {
            assert_eq!(pixel.3, orig_a, "Alpha changed at pixel {i}");
        }
    }

    #[test]
    fn test_modifies_visible_pixels() {
        let w = 20;
        let h = 20;
        let mut buffer = vec![(0.5, 0.4, 0.3, 1.0); w * h];
        let original = buffer.clone();
        apply_aether_weave(&mut buffer, w, h, &AetherConfig::default());
        assert_ne!(buffer, original, "Effect should modify at least some pixels");
    }

    #[test]
    fn test_transparent_pixels_unchanged() {
        let mut buffer = vec![(0.5, 0.3, 0.2, 0.0); 100];
        let original = buffer.clone();
        apply_aether_weave(&mut buffer, 10, 10, &AetherConfig::default());
        assert_eq!(buffer, original);
    }

    #[test]
    fn test_luxury_vs_non_luxury_differ() {
        let w = 15;
        let h = 15;
        let input = vec![(0.6, 0.4, 0.3, 1.0); w * h];

        let mut luxury = input.clone();
        let cfg_lux = AetherConfig { luxury_mode: true, ..AetherConfig::default() };
        apply_aether_weave(&mut luxury, w, h, &cfg_lux);

        let mut non_luxury = input;
        let cfg_no_lux = AetherConfig { luxury_mode: false, ..AetherConfig::default() };
        apply_aether_weave(&mut non_luxury, w, h, &cfg_no_lux);

        assert_ne!(luxury, non_luxury, "Luxury and non-luxury modes should differ");
    }

    #[test]
    fn test_determinism() {
        let w = 12;
        let h = 12;
        let cfg = AetherConfig::default();
        let make_buf = || vec![(0.5, 0.3, 0.2, 1.0); w * h];

        let mut a = make_buf();
        let mut b = make_buf();
        apply_aether_weave(&mut a, w, h, &cfg);
        apply_aether_weave(&mut b, w, h, &cfg);
        assert_eq!(a, b);
    }
}
