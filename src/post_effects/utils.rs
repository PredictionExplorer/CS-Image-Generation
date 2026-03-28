//! Shared utilities for post-processing effects.

use super::PixelBuffer;
use rayon::prelude::*;

/// A simple 2D hash to generate pseudo-random points for distance fields.
#[inline]
pub fn hash2(p: (f64, f64)) -> (f64, f64) {
    let h = (p.0 * 12.9898 + p.1 * 78.233).sin() * 43758.5453;
    (h.fract(), (h * 1.57).fract())
}

/// Hermite smoothstep interpolation between two edges.
#[inline]
pub fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    if (edge1 - edge0).abs() < f64::EPSILON {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Soft-knee highlight extraction factor based on luminance.
#[inline]
pub fn highlight_extract_factor(luminance: f64) -> f64 {
    let knee = crate::render::constants::DEFAULT_HIGHLIGHT_EXTRACT_KNEE;
    let threshold = crate::render::constants::DEFAULT_HIGHLIGHT_EXTRACT_THRESHOLD;
    smoothstep(threshold - knee * 0.5, threshold + knee * 0.5, luminance)
}

/// Computes luminance gradients for a given pixel buffer, used for flow-aware effects.
pub fn calculate_gradients(buffer: &PixelBuffer, width: usize, height: usize) -> Vec<(f64, f64)> {
    let mut luminance = vec![0.0f64; buffer.len()];
    luminance.par_iter_mut().enumerate().for_each(|(idx, lum)| {
        let (r, g, b, a) = buffer[idx];
        *lum = if a > 0.0 { (0.2126 * r + 0.7152 * g + 0.0722 * b) / a } else { 0.0 };
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
    fn test_hash2_returns_finite_values() {
        for i in 0..100 {
            let (x, y) = hash2((i as f64 * 1.7, i as f64 * 3.1));
            assert!(x.is_finite(), "hash2 x={x} not finite");
            assert!(y.is_finite(), "hash2 y={y} not finite");
        }
    }

    #[test]
    fn test_hash2_deterministic() {
        let (a1, b1) = hash2((42.0, 99.0));
        let (a2, b2) = hash2((42.0, 99.0));
        assert_eq!(a1.to_bits(), a2.to_bits());
        assert_eq!(b1.to_bits(), b2.to_bits());
    }

    #[test]
    fn test_smoothstep_edges() {
        assert!((smoothstep(0.0, 1.0, 0.0) - 0.0).abs() < 1e-12);
        assert!((smoothstep(0.0, 1.0, 1.0) - 1.0).abs() < 1e-12);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_smoothstep_clamps() {
        assert!((smoothstep(0.0, 1.0, -1.0) - 0.0).abs() < 1e-12);
        assert!((smoothstep(0.0, 1.0, 2.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_smoothstep_degenerate_edges() {
        let result = smoothstep(0.5, 0.5, 0.5);
        assert!(!result.is_nan(), "degenerate edges should not produce NaN");
    }

    #[test]
    fn test_highlight_extract_factor_range() {
        for i in 0..=20 {
            let lum = i as f64 * 0.1;
            let f = highlight_extract_factor(lum);
            assert!((0.0..=1.0).contains(&f), "extract_factor({lum}) = {f} out of [0,1]");
        }
    }

    #[test]
    fn test_highlight_extract_monotonic() {
        let mut prev = highlight_extract_factor(0.0);
        for i in 1..=20 {
            let lum = i as f64 * 0.1;
            let curr = highlight_extract_factor(lum);
            assert!(curr >= prev, "extract_factor should be monotonic: f({lum}) = {curr} < f(prev) = {prev}");
            prev = curr;
        }
    }

    #[test]
    fn test_calculate_gradients_dimensions() {
        let buf: PixelBuffer = vec![(0.5, 0.5, 0.5, 1.0); 16 * 12];
        let grads = calculate_gradients(&buf, 16, 12);
        assert_eq!(grads.len(), 16 * 12);
    }

    #[test]
    fn test_calculate_gradients_uniform_is_zero() {
        let buf: PixelBuffer = vec![(0.4, 0.4, 0.4, 1.0); 8 * 8];
        let grads = calculate_gradients(&buf, 8, 8);
        for &(gx, gy) in &grads {
            assert!(gx.abs() < 1e-10 && gy.abs() < 1e-10, "uniform buffer should have zero gradient");
        }
    }
}
