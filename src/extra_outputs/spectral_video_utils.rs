//! Shared utilities for spectral video effects.
//!
//! Pre-renders per-bin tinted buffers and provides weighted SPD blending,
//! used by the various spectral video modules (cycle, assembly, breathing, etc.).

use crate::render::apply_energy_density_shift;
use crate::spectrum::{NUM_BINS, wavelength_nm_for_bin, wavelength_to_rgb};
use rayon::prelude::*;

/// Gamma value used for spectral bin visualizations (matches spectral_gallery.rs).
const DISPLAY_GAMMA: f64 = 2.2;

/// Pre-rendered per-bin image buffers, normalized and tinted.
///
/// Each bin buffer stores linear-space `[R, G, B]` per pixel, tinted by the
/// bin's centre wavelength colour and normalized so the brightest pixel in
/// that bin is 1.0.
pub struct BinBuffers {
    pub buffers: Vec<Vec<[f64; 3]>>,
    pub pixel_count: usize,
    pub width: u32,
    pub height: u32,
}

impl BinBuffers {
    /// Build all 64 bin buffers from the accumulated SPD.
    /// Applies energy-density shift, then normalizes + tints each bin.
    pub fn from_spd(
        mut accum_spd: Vec<[f64; NUM_BINS]>,
        width: u32,
        height: u32,
    ) -> Self {
        let pixel_count = (width * height) as usize;
        apply_energy_density_shift(&mut accum_spd);

        let buffers: Vec<Vec<[f64; 3]>> = (0..NUM_BINS)
            .into_par_iter()
            .map(|bin| {
                let wavelength = wavelength_nm_for_bin(bin);
                let (tint_r, tint_g, tint_b) = wavelength_to_rgb(wavelength);

                let max_val: f64 = accum_spd
                    .iter()
                    .map(|spd| spd[bin])
                    .fold(0.0f64, f64::max)
                    .max(1e-10);

                let mut buf = vec![[0.0f64; 3]; pixel_count];
                for (px, spd) in buf.iter_mut().zip(accum_spd.iter()) {
                    let normalized = (spd[bin] / max_val).clamp(0.0, 1.0);
                    px[0] = (normalized * tint_r).powf(1.0 / DISPLAY_GAMMA);
                    px[1] = (normalized * tint_g).powf(1.0 / DISPLAY_GAMMA);
                    px[2] = (normalized * tint_b).powf(1.0 / DISPLAY_GAMMA);
                }
                buf
            })
            .collect();

        Self { buffers, pixel_count, width, height }
    }

    /// Lerp between two bins at a fractional position, writing into `dest`.
    /// `bin_f` is a floating-point bin index that wraps at `NUM_BINS`.
    pub fn lerp_bins(&self, bin_f: f64, dest: &mut Vec<u16>) {
        let bin_f = bin_f.rem_euclid(NUM_BINS as f64);
        let lo = bin_f.floor() as usize % NUM_BINS;
        let hi = (lo + 1) % NUM_BINS;
        let t = bin_f.fract();

        dest.resize(self.pixel_count * 3, 0u16);
        let buf_lo = &self.buffers[lo];
        let buf_hi = &self.buffers[hi];

        dest.par_chunks_mut(3)
            .enumerate()
            .for_each(|(i, chunk)| {
                let r = buf_lo[i][0] * (1.0 - t) + buf_hi[i][0] * t;
                let g = buf_lo[i][1] * (1.0 - t) + buf_hi[i][1] * t;
                let b = buf_lo[i][2] * (1.0 - t) + buf_hi[i][2] * t;
                chunk[0] = (r.clamp(0.0, 1.0) * 65535.0).round() as u16;
                chunk[1] = (g.clamp(0.0, 1.0) * 65535.0).round() as u16;
                chunk[2] = (b.clamp(0.0, 1.0) * 65535.0).round() as u16;
            });
    }

    /// Blend multiple bins with a weight vector, writing 16-bit output into `dest`.
    pub fn weighted_blend(&self, weights: &[f64; NUM_BINS], dest: &mut Vec<u16>) {
        dest.resize(self.pixel_count * 3, 0u16);

        dest.par_chunks_mut(3)
            .enumerate()
            .for_each(|(i, chunk)| {
                let mut r = 0.0f64;
                let mut g = 0.0f64;
                let mut b = 0.0f64;
                for bin in 0..NUM_BINS {
                    let w = weights[bin];
                    if w > 0.0 {
                        r += self.buffers[bin][i][0] * w;
                        g += self.buffers[bin][i][1] * w;
                        b += self.buffers[bin][i][2] * w;
                    }
                }
                chunk[0] = (r.clamp(0.0, 1.0) * 65535.0).round() as u16;
                chunk[1] = (g.clamp(0.0, 1.0) * 65535.0).round() as u16;
                chunk[2] = (b.clamp(0.0, 1.0) * 65535.0).round() as u16;
            });
    }

    /// Compose a frame where each pixel selects from a per-pixel bin (with lerp).
    /// `bin_per_pixel` provides a floating-point bin index for each pixel.
    pub fn per_pixel_bin_select(&self, bin_per_pixel: &[f64], dest: &mut Vec<u16>) {
        dest.resize(self.pixel_count * 3, 0u16);

        dest.par_chunks_mut(3)
            .enumerate()
            .for_each(|(i, chunk)| {
                let bin_f = bin_per_pixel[i].rem_euclid(NUM_BINS as f64);
                let lo = bin_f.floor() as usize % NUM_BINS;
                let hi = (lo + 1) % NUM_BINS;
                let t = bin_f.fract();

                let r = self.buffers[lo][i][0] * (1.0 - t) + self.buffers[hi][i][0] * t;
                let g = self.buffers[lo][i][1] * (1.0 - t) + self.buffers[hi][i][1] * t;
                let b = self.buffers[lo][i][2] * (1.0 - t) + self.buffers[hi][i][2] * t;
                chunk[0] = (r.clamp(0.0, 1.0) * 65535.0).round() as u16;
                chunk[1] = (g.clamp(0.0, 1.0) * 65535.0).round() as u16;
                chunk[2] = (b.clamp(0.0, 1.0) * 65535.0).round() as u16;
            });
    }

    /// Build a full-composite buffer by summing all bins equally.
    pub fn full_composite(&self, dest: &mut Vec<u16>) {
        let mut weights = [0.0f64; NUM_BINS];
        let w = 1.0 / NUM_BINS as f64;
        for weight in &mut weights {
            *weight = w;
        }
        self.weighted_blend(&weights, dest);
    }

    /// Per-pixel dominant bin index and tinted color (for heatmap/ripple effects).
    pub fn dominant_bin_colors(&self, accum_spd: &[[f64; NUM_BINS]]) -> Vec<([f64; 3], usize)> {
        accum_spd
            .par_iter()
            .map(|spd| {
                let dominant = spd
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let wl = wavelength_nm_for_bin(dominant);
                let (r, g, b) = wavelength_to_rgb(wl);
                ([r, g, b], dominant)
            })
            .collect()
    }
}

/// 8 perceptual color groups mapping the 64 bins into wavelength bands.
pub const NUM_GROUPS: usize = 8;
pub const BINS_PER_GROUP: usize = NUM_BINS / NUM_GROUPS;

pub const GROUP_NAMES: [&str; NUM_GROUPS] = [
    "deep violet",
    "violet-blue",
    "blue",
    "cyan-green",
    "green",
    "yellow",
    "orange",
    "red",
];

/// Return the group index (0..NUM_GROUPS) for a given bin.
#[inline]
pub fn bin_group(bin: usize) -> usize {
    (bin / BINS_PER_GROUP).min(NUM_GROUPS - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bin_group_boundaries() {
        assert_eq!(bin_group(0), 0);
        assert_eq!(bin_group(BINS_PER_GROUP - 1), 0);
        assert_eq!(bin_group(BINS_PER_GROUP), 1);
        assert_eq!(bin_group(NUM_BINS - 1), NUM_GROUPS - 1);
    }

    #[test]
    fn test_group_count() {
        assert_eq!(NUM_BINS / BINS_PER_GROUP, NUM_GROUPS);
    }
}
