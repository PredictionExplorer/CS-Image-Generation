//! Volumetric god rays.
//!
//! Computes a luminance-weighted centroid of the brightest region of the image
//! and then sweeps the highlights along radial rays from that centroid. The
//! result is a set of radial bands emanating from the bright core — the
//! classic crepuscular / volumetric light look.
//!
//! This is a post-pass on the already-tonemapped RGBA buffer; it composites
//! additively on the base image.

use super::{PixelBuffer, PostEffect, PostEffectError, utils};
use rayon::prelude::*;

/// Volumetric god rays effect (post-tonemap, additive).
#[derive(Clone, Debug)]
pub struct GodRays {
    /// Overall blend strength (0 disables).
    pub strength: f64,
    /// Number of samples along each radial ray (more = smoother, slower).
    pub samples: usize,
    /// Per-step decay (0..1). Smaller values make the rays shorter.
    pub decay: f64,
    /// Per-step step size in pixels (scaled by image diagonal in `new`).
    pub step_px: f64,
    /// Whether this effect is enabled.
    pub enabled: bool,
}

impl Default for GodRays {
    fn default() -> Self {
        Self { strength: 0.35, samples: 24, decay: 0.94, step_px: 2.0, enabled: true }
    }
}

impl GodRays {
    /// Build a `GodRays` pass whose sweep length is tuned to image diagonal.
    #[must_use]
    pub fn for_image(width: usize, height: usize, strength: f64) -> Self {
        let diag = ((width * width + height * height) as f64).sqrt();
        let step_px = (diag / 160.0).clamp(1.0, 6.0);
        Self {
            strength: strength.clamp(0.0, 2.0),
            samples: 28,
            decay: 0.95,
            step_px,
            enabled: true,
        }
    }

    /// Compute the luminance-weighted centroid of pixels whose luminance is
    /// above `threshold`. Returns `None` if no pixel clears the threshold.
    fn centroid(
        input: &PixelBuffer,
        width: usize,
        height: usize,
        threshold: f64,
    ) -> Option<(f64, f64)> {
        let mut sx = 0.0f64;
        let mut sy = 0.0f64;
        let mut sw = 0.0f64;
        for y in 0..height {
            for x in 0..width {
                let p = input[y * width + x];
                let a = p.3.max(1e-9);
                let lum = 0.2126 * (p.0 / a) + 0.7152 * (p.1 / a) + 0.0722 * (p.2 / a);
                if lum > threshold {
                    let w = lum - threshold;
                    sx += x as f64 * w;
                    sy += y as f64 * w;
                    sw += w;
                }
            }
        }
        if sw <= 0.0 { None } else { Some((sx / sw, sy / sw)) }
    }

    /// Sample pixel at (px, py) with bilinear interpolation; returns black for
    /// out-of-bounds coordinates.
    #[inline]
    fn sample_bilinear(
        buf: &[(f64, f64, f64, f64)],
        width: usize,
        height: usize,
        px: f64,
        py: f64,
    ) -> (f64, f64, f64) {
        if px < 0.0 || py < 0.0 || px >= (width - 1) as f64 || py >= (height - 1) as f64 {
            return (0.0, 0.0, 0.0);
        }
        let x0 = px.floor() as usize;
        let y0 = py.floor() as usize;
        let fx = px - x0 as f64;
        let fy = py - y0 as f64;
        let p00 = buf[y0 * width + x0];
        let p10 = buf[y0 * width + (x0 + 1)];
        let p01 = buf[(y0 + 1) * width + x0];
        let p11 = buf[(y0 + 1) * width + (x0 + 1)];
        let w00 = (1.0 - fx) * (1.0 - fy);
        let w10 = fx * (1.0 - fy);
        let w01 = (1.0 - fx) * fy;
        let w11 = fx * fy;
        (
            p00.0 * w00 + p10.0 * w10 + p01.0 * w01 + p11.0 * w11,
            p00.1 * w00 + p10.1 * w10 + p01.1 * w01 + p11.1 * w11,
            p00.2 * w00 + p10.2 * w10 + p01.2 * w01 + p11.2 * w11,
        )
    }
}

impl PostEffect for GodRays {
    fn is_enabled(&self) -> bool {
        self.enabled && self.strength > 0.0
    }

    fn process(
        &self,
        input: &PixelBuffer,
        width: usize,
        height: usize,
    ) -> Result<PixelBuffer, PostEffectError> {
        // Build a thresholded highlight buffer to sample from.
        let highlights: PixelBuffer = input
            .par_iter()
            .map(|&(r, g, b, a)| {
                if a <= 0.0 {
                    return (0.0, 0.0, 0.0, 0.0);
                }
                let sr = r / a;
                let sg = g / a;
                let sb = b / a;
                let luminance = crate::render::constants::rec709_luminance(sr, sg, sb);
                let factor = utils::highlight_extract_factor(luminance);
                (r * factor, g * factor, b * factor, a * factor)
            })
            .collect();

        let Some((cx, cy)) = Self::centroid(input, width, height, 0.55) else {
            return Ok(input.clone());
        };

        let samples = self.samples.max(2);
        let decay = self.decay.clamp(0.5, 0.999);
        let step = self.step_px.max(0.5);

        let mut output: PixelBuffer = input.clone();
        output.par_iter_mut().enumerate().for_each(|(idx, out)| {
            let x = (idx % width) as f64;
            let y = (idx / width) as f64;
            let dx = x - cx;
            let dy = y - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < 1e-6 {
                return;
            }
            // Unit vector pointing from centroid to pixel.
            let nx = dx / dist;
            let ny = dy / dist;
            // Walk from pixel back toward centroid, sampling highlights.
            let mut acc_r = 0.0;
            let mut acc_g = 0.0;
            let mut acc_b = 0.0;
            let mut weight = 1.0;
            let mut wsum = 0.0;
            for s in 0..samples {
                let t = s as f64 * step;
                if t > dist {
                    break;
                }
                let sx = x - nx * t;
                let sy = y - ny * t;
                let (r, g, b) = Self::sample_bilinear(&highlights, width, height, sx, sy);
                acc_r += r * weight;
                acc_g += g * weight;
                acc_b += b * weight;
                wsum += weight;
                weight *= decay;
            }
            if wsum > 0.0 {
                let inv = self.strength / wsum;
                // Distance attenuation so the centroid neighbourhood itself
                // doesn't get double-bright.
                let falloff = (dist / 80.0).min(1.0);
                let gain = inv * falloff;
                out.0 += acc_r * gain;
                out.1 += acc_g * gain;
                out.2 += acc_b * gain;
            }
        });

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_strength_is_passthrough() {
        let input = vec![(0.0, 0.0, 0.0, 1.0); 16 * 16];
        let gr = GodRays { strength: 0.0, ..Default::default() };
        let out = gr.process(&input, 16, 16).expect("ok");
        assert_eq!(out, input);
    }

    #[test]
    fn test_god_rays_lighten_periphery_with_bright_centre() {
        let w = 33;
        let h = 33;
        let mut input = vec![(0.0, 0.0, 0.0, 1.0); w * h];
        // Bright 3x3 blob in the centre.
        for dy in -1..=1i32 {
            for dx in -1..=1i32 {
                let x = (w as i32 / 2 + dx) as usize;
                let y = (h as i32 / 2 + dy) as usize;
                input[y * w + x] = (1.5, 1.5, 1.5, 1.0);
            }
        }
        let gr = GodRays::for_image(w, h, 0.6);
        let out = gr.process(&input, w, h).expect("ok");
        // A peripheral pixel should be brighter than before.
        let before = input[5 * w + 5];
        let after = out[5 * w + 5];
        assert!(
            after.0 > before.0 || after.1 > before.1 || after.2 > before.2,
            "peripheral pixel should pick up god-ray energy ({before:?} -> {after:?})"
        );
    }
}
