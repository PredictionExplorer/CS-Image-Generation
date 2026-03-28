use crate::render::constants;
use nalgebra::Vector3;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use smallvec::SmallVec;

#[cfg(not(target_endian = "little"))]
compile_error!("This crate requires a little-endian target for rgb48le video frame encoding");

/// Reinterpret a `&[u16]` slice as `&[u8]` with native (little-endian) byte order.
///
/// This is the safe, centralized replacement for the repeated `unsafe {
/// from_raw_parts(...) }` pattern used when piping rgb48le frames to FFmpeg.
#[inline]
pub fn u16_slice_as_bytes(slice: &[u16]) -> &[u8] {
    // SAFETY: u16 has no padding, the slice is a contiguous valid allocation,
    // and the compile_error! above guarantees little-endian layout.
    unsafe {
        std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), slice.len() * std::mem::size_of::<u16>())
    }
}

/// Standard epsilon for float comparisons
pub const FLOAT_EPSILON: f64 = 1e-10;

/// Check if a float is approximately zero
#[inline]
pub fn is_zero(x: f64) -> bool {
    x.abs() < FLOAT_EPSILON
}

/// Check if two floats are approximately equal
#[cfg(test)]
#[inline]
pub fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < FLOAT_EPSILON
}

/// Reusable FFT planner that caches plans across calls of the same length.
/// Avoids the cost of re-planning every invocation (significant for N=1M).
pub struct FftCache {
    planner: FftPlanner<f64>,
}

impl Default for FftCache {
    fn default() -> Self {
        Self { planner: FftPlanner::new() }
    }
}

impl FftCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn transform(&mut self, input: &[f64]) -> Vec<Complex<f64>> {
        let fft = self.planner.plan_fft_forward(input.len());
        let mut data: Vec<_> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft.process(&mut data);
        data
    }
}

/// Compute Fourier transform of a real-valued signal
pub fn fourier_transform(input: &[f64]) -> Vec<Complex<f64>> {
    let mut cache = FftCache::new();
    cache.transform(input)
}

/// 2D bounding box: (min_x, max_x, min_y, max_y)
pub fn bounding_box_2d(positions: &[Vec<Vector3<f64>>]) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for body in positions {
        for p in body {
            min_x = min_x.min(p[0]);
            max_x = max_x.max(p[0]);
            min_y = min_y.min(p[1]);
            max_y = max_y.max(p[1]);
        }
    }
    (min_x, max_x, min_y, max_y)
}

/// 2D bounding box with padding, always non-degenerate
pub fn bounding_box(positions: &[Vec<Vector3<f64>>]) -> (f64, f64, f64, f64) {
    let (mut min_x, mut max_x, mut min_y, mut max_y) = bounding_box_2d(positions);
    if (max_x - min_x).abs() < 1e-12 {
        min_x -= constants::BOUNDING_BOX_PADDING;
        max_x += constants::BOUNDING_BOX_PADDING;
    }
    if (max_y - min_y).abs() < 1e-12 {
        min_y -= constants::BOUNDING_BOX_PADDING;
        max_y += constants::BOUNDING_BOX_PADDING;
    }
    let wx = max_x - min_x;
    let wy = max_y - min_y;
    min_x -= 0.05 * wx;
    max_x += 0.05 * wx;
    min_y -= 0.05 * wy;
    max_y += 0.05 * wy;
    (min_x, max_x, min_y, max_y)
}

/// Build a simple 1D Gaussian kernel
pub fn build_gaussian_kernel(radius: usize) -> SmallVec<[f64; 32]> {
    if radius == 0 {
        return SmallVec::new();
    }
    let sigma =
        (radius as f64 / constants::GAUSSIAN_SIGMA_FACTOR).max(constants::GAUSSIAN_SIGMA_MIN);
    let kernel_size = 2 * radius + 1;
    let mut kernel = SmallVec::with_capacity(kernel_size);
    let two_sigma2 = constants::GAUSSIAN_TWO_FACTOR * sigma * sigma;
    let mut sum = 0.0;
    for i in 0..kernel_size {
        let x = i as f64 - radius as f64;
        let val = (-x * x / two_sigma2).exp();
        kernel.push(val);
        sum += val;
    }
    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn test_is_zero_positive() {
        assert!(is_zero(0.0));
        assert!(is_zero(1e-11));
        assert!(is_zero(-1e-11));
    }

    #[test]
    fn test_is_zero_negative() {
        assert!(!is_zero(1e-9));
        assert!(!is_zero(0.1));
        assert!(!is_zero(-0.1));
    }

    #[test]
    fn test_approx_eq() {
        assert!(approx_eq(1.0, 1.0));
        assert!(approx_eq(1.0, 1.0 + 1e-11));
        assert!(approx_eq(1.0, 1.0 - 1e-11));
        assert!(!approx_eq(1.0, 1.1));
        assert!(!approx_eq(0.0, 1e-9));
    }

    #[test]
    fn test_bounding_box_single_point() {
        let positions = vec![vec![Vector3::new(1.0, 2.0, 3.0)]];
        let (min_x, max_x, min_y, max_y) = bounding_box(&positions);

        // Should have padding
        assert!(min_x < 1.0);
        assert!(max_x > 1.0);
        assert!(min_y < 2.0);
        assert!(max_y > 2.0);
    }

    #[test]
    fn test_bounding_box_multiple_points() {
        let positions = vec![
            vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(10.0, 0.0, 0.0)],
            vec![Vector3::new(0.0, 10.0, 0.0), Vector3::new(10.0, 10.0, 0.0)],
        ];
        let (min_x, max_x, min_y, max_y) = bounding_box(&positions);

        // Should include all points with padding
        assert!(min_x < 0.0);
        assert!(max_x > 10.0);
        assert!(min_y < 0.0);
        assert!(max_y > 10.0);

        // Width and height should be reasonable
        let width = max_x - min_x;
        let height = max_y - min_y;
        assert!(width > 10.0); // At least the span plus padding
        assert!(height > 10.0);
    }

    #[test]
    fn test_gaussian_kernel_zero_radius() {
        let kernel = build_gaussian_kernel(0);
        assert!(kernel.is_empty());
    }

    #[test]
    fn test_gaussian_kernel_properties() {
        let radius = 5;
        let kernel = build_gaussian_kernel(radius);

        // Kernel should have correct size
        assert_eq!(kernel.len(), 2 * radius + 1);

        // Kernel should sum to approximately 1.0 (normalized)
        let sum: f64 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Kernel sum = {}, expected 1.0", sum);

        // Kernel should be symmetric
        for i in 0..radius {
            assert!(approx_eq(kernel[i], kernel[2 * radius - i]));
        }

        // Center should be the maximum value
        let center = kernel[radius];
        for &value in &kernel {
            assert!(value <= center + 1e-10);
        }
    }

    #[test]
    fn test_fourier_transform_length() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = fourier_transform(&input);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_fourier_transform_zero() {
        let input = vec![0.0; 10];
        let output = fourier_transform(&input);

        for c in output {
            assert!(c.norm() < 1e-10);
        }
    }

    #[test]
    fn test_fft_cache_matches_uncached() {
        let input: Vec<f64> = (0..128).map(|i| (i as f64 * 0.1).sin()).collect();
        let uncached = fourier_transform(&input);
        let mut cache = FftCache::new();
        let cached = cache.transform(&input);

        assert_eq!(uncached.len(), cached.len());
        for (i, (a, b)) in uncached.iter().zip(cached.iter()).enumerate() {
            assert_eq!(a.re.to_bits(), b.re.to_bits(), "re differs at index {i}");
            assert_eq!(a.im.to_bits(), b.im.to_bits(), "im differs at index {i}");
        }
    }

    #[test]
    fn test_fft_cache_reuse_same_length() {
        let mut cache = FftCache::new();
        let input1: Vec<f64> = (0..64).map(|i| (i as f64).sin()).collect();
        let input2: Vec<f64> = (0..64).map(|i| (i as f64).cos()).collect();

        let result1 = cache.transform(&input1);
        let result2 = cache.transform(&input2);

        assert_ne!(result1[1].re.to_bits(), result2[1].re.to_bits());
    }

    #[test]
    fn test_fft_cache_different_lengths() {
        let mut cache = FftCache::new();
        let small: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let large: Vec<f64> = (0..256).map(|i| i as f64).collect();

        let r_small = cache.transform(&small);
        let r_large = cache.transform(&large);

        assert_eq!(r_small.len(), 16);
        assert_eq!(r_large.len(), 256);
    }

    #[test]
    fn test_fft_cache_deterministic() {
        let input: Vec<f64> = (0..512).map(|i| (i as f64 * 0.05).sin()).collect();
        let mut cache = FftCache::new();
        let run1 = cache.transform(&input);
        let run2 = cache.transform(&input);

        for (i, (a, b)) in run1.iter().zip(run2.iter()).enumerate() {
            assert_eq!(a.re.to_bits(), b.re.to_bits(), "determinism: re at {i}");
            assert_eq!(a.im.to_bits(), b.im.to_bits(), "determinism: im at {i}");
        }
    }

    // ── u16_slice_as_bytes tests ────────────────────────────────────

    #[test]
    fn test_u16_slice_as_bytes_empty() {
        let empty: &[u16] = &[];
        assert!(super::u16_slice_as_bytes(empty).is_empty());
    }

    #[test]
    fn test_u16_slice_as_bytes_length() {
        let data = vec![0u16; 100];
        assert_eq!(super::u16_slice_as_bytes(&data).len(), 200);
    }

    #[test]
    fn test_u16_slice_as_bytes_le_byte_order() {
        let data: &[u16] = &[0x0102];
        let bytes = super::u16_slice_as_bytes(data);
        assert_eq!(bytes[0], 0x02);
        assert_eq!(bytes[1], 0x01);
    }

    #[test]
    fn test_u16_slice_as_bytes_roundtrip() {
        let original: Vec<u16> = (0u16..512).collect();
        let bytes = super::u16_slice_as_bytes(&original);
        for (i, &val) in original.iter().enumerate() {
            let lo = bytes[i * 2];
            let hi = bytes[i * 2 + 1];
            assert_eq!(u16::from_le_bytes([lo, hi]), val);
        }
    }

    #[test]
    fn test_u16_slice_as_bytes_boundary_values() {
        let data: &[u16] = &[0, 1, 0x7FFF, 0x8000, 0xFFFE, 0xFFFF];
        let bytes = super::u16_slice_as_bytes(data);
        assert_eq!(bytes.len(), 12);
        assert_eq!(u16::from_le_bytes([bytes[0], bytes[1]]), 0);
        assert_eq!(u16::from_le_bytes([bytes[10], bytes[11]]), 0xFFFF);
    }
}
