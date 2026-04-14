//! Utility functions: float comparison, Fourier transforms, bounding boxes, and Gaussian kernels.

use crate::render::constants;
use nalgebra::Vector3;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use smallvec::SmallVec;

/// Standard epsilon for float comparisons
pub const FLOAT_EPSILON: f64 = 1e-10;

/// Saturating conversion from f64 to usize, clamping to `[0, usize::MAX]`.
///
/// Prefer this over bare `x as usize` which has undefined-ish overflow semantics.
#[inline]
#[must_use]
pub fn f64_to_usize_saturating(x: f64) -> usize {
    if x <= 0.0 {
        0
    } else if x >= usize::MAX as f64 {
        usize::MAX
    } else {
        x as usize
    }
}

/// Saturating conversion from f64 to u16, clamping to `[0, 65535]`.
///
/// Prefer this over bare `x as u16` which truncates silently.
#[inline]
#[must_use]
pub fn f64_to_u16_saturating(x: f64) -> u16 {
    if x <= 0.0 {
        0
    } else if x >= f64::from(u16::MAX) {
        u16::MAX
    } else {
        x as u16
    }
}

/// Check if a float is approximately zero
#[inline]
#[must_use]
pub fn is_zero(x: f64) -> bool {
    x.abs() < FLOAT_EPSILON
}

/// Check if two floats are approximately equal
#[cfg(test)]
#[inline]
#[must_use]
pub fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < FLOAT_EPSILON
}

/// Compute Fourier transform of a real-valued signal.
///
/// Reuses a thread-local `FftPlanner` to avoid repeated allocation when called
/// many times with the same (or different) input lengths.
#[must_use]
pub fn fourier_transform(input: &[f64]) -> Vec<Complex<f64>> {
    use std::cell::RefCell;
    thread_local! {
        static PLANNER: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::new());
    }
    PLANNER.with(|planner| {
        let fft = planner.borrow_mut().plan_fft_forward(input.len());
        let mut data: Vec<_> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft.process(&mut data);
        data
    })
}

/// 2D bounding box: (`min_x`, `max_x`, `min_y`, `max_y`)
#[must_use]
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
#[must_use]
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
#[must_use]
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
        assert!((sum - 1.0).abs() < 1e-10, "Kernel sum = {sum}, expected 1.0");

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
    fn test_f64_to_usize_saturating_normal() {
        assert_eq!(f64_to_usize_saturating(0.0), 0);
        assert_eq!(f64_to_usize_saturating(1.0), 1);
        assert_eq!(f64_to_usize_saturating(42.0), 42);
        assert_eq!(f64_to_usize_saturating(999.9), 999);
    }

    #[test]
    fn test_f64_to_usize_saturating_negative_clamps_to_zero() {
        assert_eq!(f64_to_usize_saturating(-1.0), 0);
        assert_eq!(f64_to_usize_saturating(-1e18), 0);
        assert_eq!(f64_to_usize_saturating(f64::NEG_INFINITY), 0);
    }

    #[test]
    fn test_f64_to_usize_saturating_large_clamps_to_max() {
        assert_eq!(f64_to_usize_saturating(1e30), usize::MAX);
        assert_eq!(f64_to_usize_saturating(f64::INFINITY), usize::MAX);
    }

    #[test]
    fn test_f64_to_usize_saturating_nan_returns_zero() {
        assert_eq!(f64_to_usize_saturating(f64::NAN), 0);
    }

    #[test]
    fn test_f64_to_u16_saturating_normal() {
        assert_eq!(f64_to_u16_saturating(0.0), 0);
        assert_eq!(f64_to_u16_saturating(1.0), 1);
        assert_eq!(f64_to_u16_saturating(255.0), 255);
        assert_eq!(f64_to_u16_saturating(65535.0), 65535);
    }

    #[test]
    fn test_f64_to_u16_saturating_negative_clamps_to_zero() {
        assert_eq!(f64_to_u16_saturating(-1.0), 0);
        assert_eq!(f64_to_u16_saturating(-1e10), 0);
        assert_eq!(f64_to_u16_saturating(f64::NEG_INFINITY), 0);
    }

    #[test]
    fn test_f64_to_u16_saturating_large_clamps_to_max() {
        assert_eq!(f64_to_u16_saturating(65536.0), u16::MAX);
        assert_eq!(f64_to_u16_saturating(1e10), u16::MAX);
        assert_eq!(f64_to_u16_saturating(f64::INFINITY), u16::MAX);
    }

    #[test]
    fn test_f64_to_u16_saturating_nan_returns_zero() {
        assert_eq!(f64_to_u16_saturating(f64::NAN), 0);
    }

    #[test]
    fn test_f64_to_u16_saturating_truncates_fractional() {
        assert_eq!(f64_to_u16_saturating(0.9), 0);
        assert_eq!(f64_to_u16_saturating(255.5), 255);
        assert_eq!(f64_to_u16_saturating(65534.999), 65534);
    }

    #[test]
    fn test_bounding_box_2d_basic() {
        let positions = vec![
            vec![Vector3::new(-5.0, 3.0, 0.0), Vector3::new(10.0, -2.0, 0.0)],
            vec![Vector3::new(0.0, 7.0, 0.0)],
        ];
        let (min_x, max_x, min_y, max_y) = bounding_box_2d(&positions);
        assert_eq!(min_x, -5.0);
        assert_eq!(max_x, 10.0);
        assert_eq!(min_y, -2.0);
        assert_eq!(max_y, 7.0);
    }

    #[test]
    fn test_bounding_box_2d_single_point() {
        let positions = vec![vec![Vector3::new(3.0, 4.0, 5.0)]];
        let (min_x, max_x, min_y, max_y) = bounding_box_2d(&positions);
        assert_eq!(min_x, 3.0);
        assert_eq!(max_x, 3.0);
        assert_eq!(min_y, 4.0);
        assert_eq!(max_y, 4.0);
    }

    #[test]
    fn test_bounding_box_2d_ignores_z() {
        let positions = vec![vec![Vector3::new(0.0, 0.0, -100.0), Vector3::new(0.0, 0.0, 100.0)]];
        let (min_x, max_x, min_y, max_y) = bounding_box_2d(&positions);
        assert_eq!(min_x, 0.0);
        assert_eq!(max_x, 0.0);
        assert_eq!(min_y, 0.0);
        assert_eq!(max_y, 0.0);
    }
}
