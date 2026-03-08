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
