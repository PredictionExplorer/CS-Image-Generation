//! Iridescent champlevé post-effect for creating a jewel-like enamel look.
//!
//! This effect partitions the image into Voronoi-like cells, each with a
//! metallic rim and a pearlescent, flow-aligned interior. It is designed to
//! create a high-end, crafted aesthetic reminiscent of fine jewelry or
//! crystalline structures.

use super::PixelBuffer;
use crate::render::constants;
use rayon::prelude::*;

/// Configuration for the champlevé iridescent finish.
#[derive(Clone, Debug)]
pub struct ChampleveConfig {
    /// Target cell count scale from image area (higher yields smaller cells).
    pub cell_density: f64,
    /// How strongly iridescence phase follows local gradient flow.
    pub flow_alignment: f64,
    /// Strength of thin-film interference colour in cell interiors.
    pub interference_amplitude: f64,
    /// Spatial frequency of the interference pattern.
    pub interference_frequency: f64,
    /// Opacity of the warm metallic cell rim.
    pub rim_intensity: f64,
    /// How much warm gold is mixed into the rim colour.
    pub rim_warmth: f64,
    /// Sharpness of the transition from rim to interior.
    pub rim_sharpness: f64,
    /// Brightness lift applied inside cells (away from the rim).
    pub interior_lift: f64,
    /// Directional emphasis of colour along the flow field.
    pub anisotropy: f64,
    /// Softens interior lift falloff from the cell edge.
    pub cell_softness: f64,
}

impl Default for ChampleveConfig {
    fn default() -> Self {
        Self {
            cell_density: constants::DEFAULT_CHAMPLEVE_CELL_DENSITY,
            flow_alignment: constants::DEFAULT_CHAMPLEVE_FLOW_ALIGNMENT,
            interference_amplitude: constants::DEFAULT_CHAMPLEVE_INTERFERENCE_AMPLITUDE,
            interference_frequency: constants::DEFAULT_CHAMPLEVE_INTERFERENCE_FREQUENCY,
            rim_intensity: constants::DEFAULT_CHAMPLEVE_RIM_INTENSITY,
            rim_warmth: constants::DEFAULT_CHAMPLEVE_RIM_WARMTH,
            rim_sharpness: constants::DEFAULT_CHAMPLEVE_RIM_SHARPNESS,
            interior_lift: constants::DEFAULT_CHAMPLEVE_INTERIOR_LIFT,
            anisotropy: constants::DEFAULT_CHAMPLEVE_ANISOTROPY,
            cell_softness: constants::DEFAULT_CHAMPLEVE_CELL_SOFTNESS,
        }
    }
}

use super::utils::hash2;

/// Computes the two closest Voronoi points for a given pixel coordinate.
#[inline]
fn voronoi_distances(p: (f64, f64)) -> (f64, f64) {
    let ix = p.0.floor();
    let iy = p.1.floor();
    let fx = p.0.fract();
    let fy = p.1.fract();

    let mut closest = f64::MAX;
    let mut second_closest = f64::MAX;

    for j in -1..=1 {
        for i in -1..=1 {
            let cell_offset = (f64::from(i), f64::from(j));
            let point = hash2((ix + cell_offset.0, iy + cell_offset.1));
            let r = (cell_offset.0 + point.0 - fx, cell_offset.1 + point.1 - fy);
            let d_sq = r.0 * r.0 + r.1 * r.1;

            if d_sq < closest {
                second_closest = closest;
                closest = d_sq;
            } else if d_sq < second_closest {
                second_closest = d_sq;
            }
        }
    }
    (closest, second_closest)
}

/// Applies the champlevé iridescent effect to a buffer of linear RGBA pixels.
pub fn apply_champleve_iridescence(
    buffer: &mut PixelBuffer,
    width: usize,
    height: usize,
    config: &ChampleveConfig,
) {
    if buffer.is_empty() {
        return;
    }

    let gradients = super::utils::calculate_gradients(buffer, width, height);

    let cell_scale = (width as f64 * height as f64).sqrt() / config.cell_density.max(1.0);
    let inv_cell_scale = 1.0 / cell_scale;
    let warm_rim = [1.0, 0.78, 0.35];

    buffer.par_iter_mut().enumerate().for_each(|(idx, pixel)| {
        let (r, g, b, a) = *pixel;
        if a <= 0.0 {
            return;
        }

        let u = (idx % width) as f64 * inv_cell_scale;
        let v = (idx / width) as f64 * inv_cell_scale;

        let (d1, d2) = voronoi_distances((u, v));
        let rim_dist = (d2.sqrt() - d1.sqrt()).abs();
        let rim = (1.0 - rim_dist.powf(config.rim_sharpness) * 20.0).clamp(0.0, 1.0);
        let interior = 1.0 - rim;

        let (gx, gy) = gradients[idx];
        let flow_dir = gy.atan2(gx);
        let flow_mag = (gx * gx + gy * gy).sqrt();
        let aligned_phase = flow_dir.cos() * config.flow_alignment;

        let interference =
            (config.interference_frequency * (v * 0.7 + u * 0.3 + aligned_phase)).sin();
        let spectral = interference * config.interference_amplitude;

        let mut sr = r / a;
        let mut sg = g / a;
        let mut sb = b / a;

        let rim_mix = rim * config.rim_intensity * config.rim_warmth;
        sr = sr * (1.0 - rim_mix) + warm_rim[0] * rim_mix;
        sg = sg * (1.0 - rim_mix) + warm_rim[1] * rim_mix;
        sb = sb * (1.0 - rim_mix) + warm_rim[2] * rim_mix;

        sr += spectral * 0.5 * interior;
        sg += spectral * -0.2 * interior;
        sb += spectral * -0.5 * interior;

        let anisotropy = config.anisotropy * flow_mag;
        sr *= 1.0 + anisotropy;
        sg *= 1.0 - anisotropy * 0.5;
        sb *= 1.0 - anisotropy;

        let lift = config.interior_lift * interior.powf(config.cell_softness);
        sr += lift;
        sg += lift;
        sb += lift;

        *pixel = (sr.max(0.0) * a, sg.max(0.0) * a, sb.max(0.0) * a, a);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_uses_constants() {
        let cfg = ChampleveConfig::default();
        assert_eq!(cfg.cell_density, constants::DEFAULT_CHAMPLEVE_CELL_DENSITY);
        assert_eq!(cfg.flow_alignment, constants::DEFAULT_CHAMPLEVE_FLOW_ALIGNMENT);
        assert_eq!(cfg.rim_intensity, constants::DEFAULT_CHAMPLEVE_RIM_INTENSITY);
    }

    #[test]
    fn test_voronoi_distances_ordered() {
        for i in 0..50 {
            let p = (f64::from(i) * 0.37, f64::from(i) * 0.53);
            let (d1, d2) = voronoi_distances(p);
            assert!(d1 <= d2, "Closest must be <= second closest at {p:?}");
            assert!(d1 >= 0.0, "Distance cannot be negative");
        }
    }

    #[test]
    fn test_empty_buffer_is_noop() {
        let mut buffer: Vec<(f64, f64, f64, f64)> = vec![];
        apply_champleve_iridescence(&mut buffer, 0, 0, &ChampleveConfig::default());
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_preserves_alpha() {
        let w = 10;
        let h = 10;
        let mut buffer: Vec<(f64, f64, f64, f64)> =
            (0..w * h).map(|i| (0.5, 0.3, 0.2, 0.7 + (i as f64) * 0.001)).collect();
        let alphas: Vec<f64> = buffer.iter().map(|p| p.3).collect();
        apply_champleve_iridescence(&mut buffer, w, h, &ChampleveConfig::default());
        for (i, (pixel, &orig_a)) in buffer.iter().zip(alphas.iter()).enumerate() {
            assert_eq!(pixel.3, orig_a, "Alpha changed at pixel {i}");
        }
    }

    #[test]
    fn test_modifies_visible_pixels() {
        let w = 20;
        let h = 20;
        let mut buffer: Vec<(f64, f64, f64, f64)> = vec![(0.5, 0.4, 0.3, 1.0); w * h];
        let original = buffer.clone();
        apply_champleve_iridescence(&mut buffer, w, h, &ChampleveConfig::default());
        assert_ne!(buffer, original, "Effect should modify at least some pixels");
    }

    #[test]
    fn test_transparent_pixels_unchanged() {
        let mut buffer = vec![(0.5, 0.3, 0.2, 0.0); 100];
        let original = buffer.clone();
        apply_champleve_iridescence(&mut buffer, 10, 10, &ChampleveConfig::default());
        assert_eq!(buffer, original);
    }

    #[test]
    fn test_determinism() {
        let w = 15;
        let h = 15;
        let cfg = ChampleveConfig::default();
        let make_buf = || vec![(0.6, 0.4, 0.3, 1.0); w * h];

        let mut a = make_buf();
        let mut b = make_buf();
        apply_champleve_iridescence(&mut a, w, h, &cfg);
        apply_champleve_iridescence(&mut b, w, h, &cfg);
        assert_eq!(a, b);
    }
}
