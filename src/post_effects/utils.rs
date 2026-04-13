//! Shared utilities for post-processing effects.

use super::PixelBuffer;
use rayon::prelude::*;

const HASH_SCALE_X: f64 = 12.9898;
const HASH_SCALE_Y: f64 = 78.233;
const HASH_FRACT_MULTIPLIER: f64 = 43758.5453;
const HASH_TIME_SCALE: f64 = 1.57;

/// A simple 2D hash to generate pseudo-random points for distance fields.
#[inline]
pub(super) fn hash2(p: (f64, f64)) -> (f64, f64) {
    let h = (p.0 * HASH_SCALE_X + p.1 * HASH_SCALE_Y).sin() * HASH_FRACT_MULTIPLIER;
    (h.fract(), (h * HASH_TIME_SCALE).fract())
}

/// Hermite smoothstep interpolation between two edges.
#[inline]
pub(super) fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    if (edge1 - edge0).abs() < f64::EPSILON {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Soft-knee highlight extraction factor based on luminance.
#[inline]
pub(super) fn highlight_extract_factor(luminance: f64) -> f64 {
    let knee = crate::render::constants::DEFAULT_HIGHLIGHT_EXTRACT_KNEE;
    let threshold = crate::render::constants::DEFAULT_HIGHLIGHT_EXTRACT_THRESHOLD;
    smoothstep(threshold - knee * 0.5, threshold + knee * 0.5, luminance)
}

/// Computes luminance gradients for a given pixel buffer, used for flow-aware effects.
pub(super) fn calculate_gradients(buffer: &PixelBuffer, width: usize, height: usize) -> Vec<(f64, f64)> {
    let mut luminance = vec![0.0f64; buffer.len()];
    luminance.par_iter_mut().enumerate().for_each(|(idx, lum)| {
        let (r, g, b, a) = buffer[idx];
        *lum = if a > 0.0 { crate::render::constants::rec709_luminance(r, g, b) / a } else { 0.0 };
    });

    let mut gradients = vec![(0.0, 0.0); buffer.len()];
    gradients.par_iter_mut().enumerate().for_each(|(idx, grad)| {
        let x = (idx % width) as isize;
        let y = (idx / width) as isize;
        let sample = |sx: isize, sy: isize| -> f64 {
            let sx = sx.clamp(0, width as isize - 1);
            let sy = sy.clamp(0, height as isize - 1);
            luminance[sy as usize * width + sx as usize]
        };
        let gx = sample(x + 1, y) - sample(x - 1, y);
        let gy = sample(x, y + 1) - sample(x, y - 1);

        // Normalize gradients to avoid overpowering chroma modulation
        let magnitude = (gx * gx + gy * gy).sqrt().max(1e-5);
        let scale = (0.85 / magnitude).min(1.0);

        *grad = (gx * scale, gy * scale);
    });
    gradients
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash2_determinism() {
        let a = hash2((1.0, 2.0));
        let b = hash2((1.0, 2.0));
        assert_eq!(a, b);
    }

    #[test]
    fn test_hash2_variation() {
        let a = hash2((0.0, 0.0));
        let b = hash2((1.0, 0.0));
        let c = hash2((0.0, 1.0));
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }

    #[test]
    fn test_hash2_output_is_fractional() {
        for i in 0..20 {
            let (x, y) = hash2((i as f64, (i * 7) as f64));
            assert!(x.is_finite());
            assert!(y.is_finite());
        }
    }

    #[test]
    fn test_smoothstep_below_edge0() {
        assert_eq!(smoothstep(0.0, 1.0, -0.5), 0.0);
    }

    #[test]
    fn test_smoothstep_above_edge1() {
        assert_eq!(smoothstep(0.0, 1.0, 1.5), 1.0);
    }

    #[test]
    fn test_smoothstep_at_midpoint() {
        let mid = smoothstep(0.0, 1.0, 0.5);
        assert!((mid - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_smoothstep_at_edges() {
        assert_eq!(smoothstep(0.0, 1.0, 0.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 1.0), 1.0);
    }

    #[test]
    fn test_smoothstep_equal_edges() {
        assert_eq!(smoothstep(0.5, 0.5, 0.3), 0.0);
        assert_eq!(smoothstep(0.5, 0.5, 0.5), 1.0);
        assert_eq!(smoothstep(0.5, 0.5, 0.7), 1.0);
    }

    #[test]
    fn test_highlight_extract_factor_zero_luminance() {
        let f = highlight_extract_factor(0.0);
        assert_eq!(f, 0.0);
    }

    #[test]
    fn test_highlight_extract_factor_high_luminance() {
        let f = highlight_extract_factor(1.0);
        assert!((f - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_highlight_extract_factor_monotonic() {
        let mut prev = highlight_extract_factor(0.0);
        for i in 1..=100 {
            let lum = i as f64 / 100.0;
            let cur = highlight_extract_factor(lum);
            assert!(cur >= prev - 1e-10, "Not monotonic at lum={lum}");
            prev = cur;
        }
    }

    #[test]
    fn test_calculate_gradients_uniform_buffer() {
        let buffer: PixelBuffer = vec![(0.5, 0.5, 0.5, 1.0); 9];
        let grads = calculate_gradients(&buffer, 3, 3);
        assert_eq!(grads.len(), 9);
        for &(gx, gy) in &grads {
            assert!(gx.abs() < 1e-10, "Expected zero gradient on uniform buffer, got gx={gx}");
            assert!(gy.abs() < 1e-10, "Expected zero gradient on uniform buffer, got gy={gy}");
        }
    }

    #[test]
    fn test_calculate_gradients_horizontal_ramp() {
        let w = 5;
        let h = 3;
        let buffer: PixelBuffer = (0..w * h)
            .map(|idx| {
                let x = (idx % w) as f64 / (w - 1) as f64;
                (x, x, x, 1.0)
            })
            .collect();
        let grads = calculate_gradients(&buffer, w, h);
        let center = grads[w + 2];
        assert!(center.0.abs() > 0.01, "Expected non-zero horizontal gradient");
    }

    #[test]
    fn test_calculate_gradients_empty_buffer() {
        let empty: PixelBuffer = vec![];
        let grads = calculate_gradients(&empty, 0, 0);
        assert!(grads.is_empty());
    }
}
