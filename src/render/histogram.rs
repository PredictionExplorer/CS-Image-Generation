//! Histogram computation and analysis
//!
//! This module provides efficient histogram storage and computation of color statistics
//! for automatic exposure and color grading.
//!
//! Public API methods are provided for library consumers even if not used internally.

use crate::render::constants;

/// Combined channel and luminance analysis used to drive tone mapping.
#[derive(Clone, Copy, Debug)]
pub struct TonemappingAnalysis {
    pub black_r: f64,
    pub white_r: f64,
    pub black_g: f64,
    pub white_g: f64,
    pub black_b: f64,
    pub white_b: f64,
    pub normalized_luma_white: f64,
    pub near_clip_ratio: f64,
    pub exposure_scale: f64,
}

/// Optimized histogram storage with better memory layout
pub struct HistogramData {
    /// Interleaved RGB data for better cache locality
    data: Vec<[f64; 3]>,
}

impl HistogramData {
    /// Create new histogram storage with given capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self { data: Vec::with_capacity(capacity) }
    }

    /// Add a pixel's RGB values to the histogram
    #[inline]
    pub fn push(&mut self, r: f64, g: f64, b: f64) {
        self.data.push([r, g, b]);
    }

    /// Reserve additional capacity to reduce reallocations
    ///
    /// Call this if you know approximately how many samples you'll collect.
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
    }

    /// Extend the histogram with a pre-built slice of samples.
    pub fn extend_from_slice(&mut self, samples: &[[f64; 3]]) {
        self.data.extend_from_slice(samples);
    }

    /// Get raw histogram data for custom analysis
    pub fn data(&self) -> &[[f64; 3]] {
        &self.data
    }
}

/// Select the value at a given percentile index using O(n) partial sort.
fn select_percentile(data: &mut [f64], idx: usize) -> f64 {
    let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
    data.select_nth_unstable_by(idx, cmp);
    data[idx]
}

/// Compute black/white points from histogram data using O(n) partial selection
/// instead of O(n log n) full sorts.
fn compute_black_white_gamma(
    all_r: &mut [f64],
    all_g: &mut [f64],
    all_b: &mut [f64],
    clip_black: f64,
    clip_white: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    let total_pix = all_r.len();
    if total_pix == 0 {
        return (0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    }

    let black_idx =
        ((clip_black * total_pix as f64).round() as usize).min(total_pix.saturating_sub(1));
    let white_idx =
        ((clip_white * total_pix as f64).round() as usize).min(total_pix.saturating_sub(1));

    // Parallel partial selection: each channel finds its black and white
    // percentiles independently in O(n) average time.
    let ((black_r, white_r), ((black_g, white_g), (black_b, white_b))) = rayon::join(
        || {
            let br = select_percentile(all_r, black_idx);
            let wr = select_percentile(all_r, white_idx);
            (br, wr)
        },
        || {
            rayon::join(
                || {
                    let bg = select_percentile(all_g, black_idx);
                    let wg = select_percentile(all_g, white_idx);
                    (bg, wg)
                },
                || {
                    let bb = select_percentile(all_b, black_idx);
                    let wb = select_percentile(all_b, white_idx);
                    (bb, wb)
                },
            )
        },
    );

    (black_r, white_r, black_g, white_g, black_b, white_b)
}

/// Analyze samples to produce channel levels plus a luminance-driven exposure scale.
pub fn analyze_tonemapping(
    samples: &[[f64; 3]],
    clip_black: f64,
    clip_white: f64,
) -> TonemappingAnalysis {
    if samples.is_empty() {
        return TonemappingAnalysis {
            black_r: 0.0,
            white_r: 1.0,
            black_g: 0.0,
            white_g: 1.0,
            black_b: 0.0,
            white_b: 1.0,
            normalized_luma_white: 1.0,
            near_clip_ratio: 0.0,
            exposure_scale: 1.0,
        };
    }

    let mut all_r = Vec::with_capacity(samples.len());
    let mut all_g = Vec::with_capacity(samples.len());
    let mut all_b = Vec::with_capacity(samples.len());
    for &[r, g, b] in samples {
        all_r.push(r);
        all_g.push(g);
        all_b.push(b);
    }

    let (black_r, white_r, black_g, white_g, black_b, white_b) =
        compute_black_white_gamma(&mut all_r, &mut all_g, &mut all_b, clip_black, clip_white);

    let range_r = (white_r - black_r).max(1e-14);
    let range_g = (white_g - black_g).max(1e-14);
    let range_b = (white_b - black_b).max(1e-14);

    let mut normalized_luminances = Vec::with_capacity(samples.len());
    for &[r, g, b] in samples {
        let nr = ((r - black_r).max(0.0)) / range_r;
        let ng = ((g - black_g).max(0.0)) / range_g;
        let nb = ((b - black_b).max(0.0)) / range_b;
        normalized_luminances.push(0.2126 * nr + 0.7152 * ng + 0.0722 * nb);
    }

    let total_samples = normalized_luminances.len();
    let white_idx =
        ((clip_white * total_samples as f64).round() as usize).min(total_samples.saturating_sub(1));
    let normalized_luma_white = select_percentile(&mut normalized_luminances, white_idx).max(1e-6);

    let mut exposure_scale =
        (constants::DEFAULT_PRETONEMAP_LUMA_TARGET / normalized_luma_white).min(1.0);
    exposure_scale = exposure_scale.max(constants::DEFAULT_MIN_EXPOSURE_SCALE);

    let near_clip_count = normalized_luminances
        .iter()
        .filter(|&&luma| luma * exposure_scale > constants::DEFAULT_PRETONEMAP_NEAR_CLIP_THRESHOLD)
        .count();
    let near_clip_ratio = near_clip_count as f64 / total_samples as f64;

    if near_clip_ratio > constants::DEFAULT_PRETONEMAP_NEAR_CLIP_BUDGET {
        let excess = near_clip_ratio / constants::DEFAULT_PRETONEMAP_NEAR_CLIP_BUDGET - 1.0;
        let darken_factor = 1.0 + excess * constants::DEFAULT_PRETONEMAP_BUDGET_RESPONSE;
        exposure_scale =
            (exposure_scale / darken_factor).max(constants::DEFAULT_MIN_EXPOSURE_SCALE);
    }

    TonemappingAnalysis {
        black_r,
        white_r,
        black_g,
        white_g,
        black_b,
        white_b,
        normalized_luma_white,
        near_clip_ratio,
        exposure_scale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_tonemapping_reduces_exposure_for_hot_samples() {
        let mut samples = vec![[0.05, 0.05, 0.05]; 900];
        samples.extend(std::iter::repeat_n([3.0, 2.9, 2.8], 100));

        let analysis = analyze_tonemapping(&samples, 0.01, 0.95);

        assert!(analysis.exposure_scale < 1.0);
        assert!(analysis.normalized_luma_white >= 1.0);
    }

    #[test]
    fn test_analyze_tonemapping_budget_governor_reacts_to_high_near_clip_ratio() {
        let mut samples = vec![[0.4, 0.4, 0.4]; 900];
        samples.extend(std::iter::repeat_n([3.0, 3.1, 3.0], 90));
        samples.extend(std::iter::repeat_n([12.0, 12.5, 12.2], 10));

        let analysis = analyze_tonemapping(&samples, 0.01, 0.95);

        assert!(analysis.near_clip_ratio > 0.0);
        assert!(analysis.exposure_scale <= 1.0);
    }

    #[test]
    fn test_extend_from_slice_matches_sequential_push() {
        let samples = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]];

        let mut via_push = HistogramData::with_capacity(3);
        for &[r, g, b] in &samples {
            via_push.push(r, g, b);
        }

        let mut via_extend = HistogramData::with_capacity(3);
        via_extend.extend_from_slice(&samples);

        assert_eq!(via_push.data().len(), via_extend.data().len());
        for (a, b) in via_push.data().iter().zip(via_extend.data()) {
            assert_eq!(a[0].to_bits(), b[0].to_bits());
            assert_eq!(a[1].to_bits(), b[1].to_bits());
            assert_eq!(a[2].to_bits(), b[2].to_bits());
        }
    }

    #[test]
    fn test_histogram_data_reserve_and_capacity() {
        let mut hist = HistogramData::with_capacity(100);
        hist.reserve(200);
        for i in 0..300 {
            hist.push(i as f64, i as f64 * 2.0, i as f64 * 3.0);
        }
        assert_eq!(hist.data().len(), 300);
    }

    #[test]
    fn test_analyze_tonemapping_empty_returns_defaults() {
        let analysis = analyze_tonemapping(&[], 0.01, 0.99);
        assert_eq!(analysis.black_r, 0.0);
        assert_eq!(analysis.white_r, 1.0);
        assert_eq!(analysis.exposure_scale, 1.0);
    }

    #[test]
    fn test_analyze_tonemapping_all_identical_samples() {
        let samples = vec![[0.5, 0.5, 0.5]; 1000];
        let analysis = analyze_tonemapping(&samples, 0.01, 0.99);
        assert!(analysis.exposure_scale > 0.0);
        assert!(analysis.exposure_scale <= 1.0);
        assert!(!analysis.normalized_luma_white.is_nan());
    }

    #[test]
    fn test_analyze_tonemapping_single_sample() {
        let samples = vec![[0.3, 0.6, 0.9]];
        let analysis = analyze_tonemapping(&samples, 0.0, 1.0);
        assert!(analysis.exposure_scale > 0.0);
    }

    #[test]
    fn test_select_percentile_correctness() {
        let mut data = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        let median = super::select_percentile(&mut data, 2);
        assert_eq!(median, 3.0);

        let mut data2 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let p90 = super::select_percentile(&mut data2, 4);
        assert_eq!(p90, 50.0);

        let p0 = super::select_percentile(&mut data2, 0);
        assert_eq!(p0, 10.0);
    }
}
