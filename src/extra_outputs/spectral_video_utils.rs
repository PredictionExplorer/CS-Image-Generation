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
                #[allow(clippy::needless_range_loop)]
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
        weights.fill(1.0 / NUM_BINS as f64);
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

    #[test]
    fn test_bin_group_saturates_beyond_last_bin() {
        assert_eq!(bin_group(NUM_BINS), NUM_GROUPS - 1);
        assert_eq!(bin_group(NUM_BINS + 100), NUM_GROUPS - 1);
    }

    fn make_test_spd(width: u32, height: u32) -> Vec<[f64; NUM_BINS]> {
        let pixel_count = (width * height) as usize;
        let mut spd = vec![[0.0; NUM_BINS]; pixel_count];
        for (i, px) in spd.iter_mut().enumerate() {
            let bin = i % NUM_BINS;
            px[bin] = 1.0 + i as f64 * 0.1;
        }
        spd
    }

    #[test]
    fn test_from_spd_buffer_count() {
        let bins = BinBuffers::from_spd(make_test_spd(2, 2), 2, 2);
        assert_eq!(bins.buffers.len(), NUM_BINS);
    }

    #[test]
    fn test_from_spd_pixel_count() {
        let bins = BinBuffers::from_spd(make_test_spd(3, 2), 3, 2);
        assert_eq!(bins.pixel_count, 6);
        for buf in &bins.buffers {
            assert_eq!(buf.len(), 6);
        }
    }

    #[test]
    fn test_from_spd_values_in_unit_range() {
        let bins = BinBuffers::from_spd(make_test_spd(2, 2), 2, 2);
        for buf in &bins.buffers {
            for px in buf {
                for &ch in px {
                    assert!(ch >= 0.0 && ch <= 1.0, "channel {ch} out of [0,1]");
                }
            }
        }
    }

    #[test]
    fn test_from_spd_nonzero_for_nonzero_input() {
        let bins = BinBuffers::from_spd(make_test_spd(2, 2), 2, 2);
        let any_nonzero = bins.buffers.iter().any(|buf| {
            buf.iter().any(|px| px[0] > 0.0 || px[1] > 0.0 || px[2] > 0.0)
        });
        assert!(any_nonzero, "non-zero SPD should produce non-zero bin buffers");
    }

    #[test]
    fn test_from_spd_zero_input_produces_black() {
        let spd = vec![[0.0; NUM_BINS]; 4];
        let bins = BinBuffers::from_spd(spd, 2, 2);
        for buf in &bins.buffers {
            for px in buf {
                assert_eq!(*px, [0.0, 0.0, 0.0]);
            }
        }
    }

    #[test]
    fn test_lerp_bins_output_length() {
        let bins = BinBuffers::from_spd(make_test_spd(2, 2), 2, 2);
        let mut dest = Vec::new();
        bins.lerp_bins(0.0, &mut dest);
        assert_eq!(dest.len(), 4 * 3);
    }

    #[test]
    fn test_lerp_bins_integer_returns_exact_bin() {
        let bins = BinBuffers::from_spd(make_test_spd(2, 2), 2, 2);
        let mut dest_int = Vec::new();
        let mut dest_lerp = Vec::new();
        bins.lerp_bins(5.0, &mut dest_int);
        bins.lerp_bins(5.0001, &mut dest_lerp);
        for (a, b) in dest_int.iter().zip(dest_lerp.iter()) {
            assert!((*a as i32 - *b as i32).unsigned_abs() <= 10,
                "integer bin and near-integer should be close: {a} vs {b}");
        }
    }

    #[test]
    fn test_lerp_bins_wraps_at_num_bins() {
        let bins = BinBuffers::from_spd(make_test_spd(2, 2), 2, 2);
        let mut dest_a = Vec::new();
        let mut dest_b = Vec::new();
        bins.lerp_bins(0.0, &mut dest_a);
        bins.lerp_bins(NUM_BINS as f64, &mut dest_b);
        assert_eq!(dest_a, dest_b, "bin 0 and bin NUM_BINS should wrap identically");
    }

    #[test]
    fn test_lerp_bins_negative_wraps() {
        let bins = BinBuffers::from_spd(make_test_spd(2, 2), 2, 2);
        let mut dest_neg = Vec::new();
        let mut dest_pos = Vec::new();
        bins.lerp_bins(-1.0, &mut dest_neg);
        bins.lerp_bins((NUM_BINS - 1) as f64, &mut dest_pos);
        assert_eq!(dest_neg, dest_pos, "negative index should wrap");
    }

    #[test]
    fn test_weighted_blend_zero_weights_produces_black() {
        let bins = BinBuffers::from_spd(make_test_spd(2, 2), 2, 2);
        let weights = [0.0; NUM_BINS];
        let mut dest = Vec::new();
        bins.weighted_blend(&weights, &mut dest);
        assert!(dest.iter().all(|&v| v == 0), "zero weights should produce all-black");
    }

    #[test]
    fn test_weighted_blend_uniform_matches_full_composite() {
        let bins = BinBuffers::from_spd(make_test_spd(2, 2), 2, 2);
        let mut dest_uniform = Vec::new();
        let mut dest_composite = Vec::new();
        let w = 1.0 / NUM_BINS as f64;
        let weights = [w; NUM_BINS];
        bins.weighted_blend(&weights, &mut dest_uniform);
        bins.full_composite(&mut dest_composite);
        assert_eq!(dest_uniform, dest_composite);
    }

    #[test]
    fn test_per_pixel_bin_select_output_length() {
        let bins = BinBuffers::from_spd(make_test_spd(2, 2), 2, 2);
        let bin_map = vec![3.0; 4];
        let mut dest = Vec::new();
        bins.per_pixel_bin_select(&bin_map, &mut dest);
        assert_eq!(dest.len(), 4 * 3);
    }

    #[test]
    fn test_per_pixel_bin_select_negative_wraps() {
        let bins = BinBuffers::from_spd(make_test_spd(2, 2), 2, 2);
        let bin_map_neg = vec![-1.0; 4];
        let bin_map_pos = vec![(NUM_BINS - 1) as f64; 4];
        let mut dest_neg = Vec::new();
        let mut dest_pos = Vec::new();
        bins.per_pixel_bin_select(&bin_map_neg, &mut dest_neg);
        bins.per_pixel_bin_select(&bin_map_pos, &mut dest_pos);
        assert_eq!(dest_neg, dest_pos, "negative bin index should wrap");
    }

    #[test]
    fn test_full_composite_nonzero() {
        let bins = BinBuffers::from_spd(make_test_spd(2, 2), 2, 2);
        let mut dest = Vec::new();
        bins.full_composite(&mut dest);
        let any_nonzero = dest.iter().any(|&v| v > 0);
        assert!(any_nonzero, "full composite of non-zero SPD should have non-zero pixels");
    }

    #[test]
    fn test_dominant_bin_colors_single_hot_bin() {
        let mut spd = vec![[0.0; NUM_BINS]; 2];
        spd[0][10] = 5.0;
        spd[1][50] = 3.0;
        let bins = BinBuffers::from_spd(vec![[1.0; NUM_BINS]; 2], 2, 1);
        let result = bins.dominant_bin_colors(&spd);
        assert_eq!(result[0].1, 10, "pixel 0 dominant should be bin 10");
        assert_eq!(result[1].1, 50, "pixel 1 dominant should be bin 50");
    }

    #[test]
    fn test_dominant_bin_colors_zero_energy() {
        let spd = vec![[0.0; NUM_BINS]; 2];
        let bins = BinBuffers::from_spd(vec![[0.0; NUM_BINS]; 2], 2, 1);
        let result = bins.dominant_bin_colors(&spd);
        assert_eq!(result.len(), 2);
        assert!(result[0].1 < NUM_BINS, "dominant bin should be a valid index");
    }
}
