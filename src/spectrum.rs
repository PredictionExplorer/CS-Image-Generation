//! Spectral utilities: 16-bin SPD handling and conversions.
//!
//! Our "spectral accumulation" keeps one energy value per wavelength bin
//! (bins are equally spaced from 380-700 nm).  Rendering draws into this
//! SPD buffer, then we convert the spectrum → linear-sRGB right before
//! the normal tone-mapping / bloom pipeline.

use crate::spectrum_simd;
use once_cell::sync::Lazy;

/// Number of wavelength buckets in the SPD.
pub const NUM_BINS: usize = 16;
/// Start / end wavelengths in nanometres.
const LAMBDA_START: f64 = 380.0;
const LAMBDA_END: f64 = 700.0;

/// Centre wavelength for a bin.
#[inline]
pub fn wavelength_nm_for_bin(bin: usize) -> f64 {
    LAMBDA_START + (bin as f64 + 0.5) * (LAMBDA_END - LAMBDA_START) / NUM_BINS as f64
}

/// Approximate (linear-sRGB) colour corresponding to a given wavelength.
/// Formula adapted from Dan Bruton's reference (gamma removed → stay linear).
fn wavelength_to_rgb(lambda: f64) -> (f64, f64, f64) {
    let (r, g, b) = if (380.0..440.0).contains(&lambda) {
        (-(lambda - 440.0) / (440.0 - 380.0), 0.0, 1.0)
    } else if (440.0..490.0).contains(&lambda) {
        (0.0, (lambda - 440.0) / (490.0 - 440.0), 1.0)
    } else if (490.0..510.0).contains(&lambda) {
        (0.0, 1.0, -(lambda - 510.0) / (510.0 - 490.0))
    } else if (510.0..580.0).contains(&lambda) {
        ((lambda - 510.0) / (580.0 - 510.0), 1.0, 0.0)
    } else if (580.0..645.0).contains(&lambda) {
        (1.0, -(lambda - 645.0) / (645.0 - 580.0), 0.0)
    } else if (645.0..=700.0).contains(&lambda) {
        (1.0, 0.0, 0.0)
    } else {
        (0.0, 0.0, 0.0)
    };

    // Intensity falloff near ends of visible range (simple linear ramp).
    let factor = if (380.0..420.0).contains(&lambda) {
        0.3 + 0.7 * (lambda - 380.0) / (420.0 - 380.0)
    } else if (420.0..645.0).contains(&lambda) {
        1.0
    } else if (645.0..=700.0).contains(&lambda) {
        0.3 + 0.7 * (700.0 - lambda) / (700.0 - 645.0)
    } else {
        0.0
    };

    (r * factor, g * factor, b * factor)
}

/// Combined lookup table for cache-friendly SPD conversion
/// Stores (R, G, B, tone_k) in a single cache line for better performance
pub static BIN_COMBINED_LUT: Lazy<[(f64, f64, f64, f64); NUM_BINS]> = Lazy::new(|| {
    let mut arr = [(0.0, 0.0, 0.0, 0.0); NUM_BINS];
    #[allow(clippy::needless_range_loop)] // Direct indexing is clearer here
    for i in 0..NUM_BINS {
        let (r, g, b) = wavelength_to_rgb(wavelength_nm_for_bin(i));
        let lambda = wavelength_nm_for_bin(i);
        
        // Compute tone-mapping strength inline
        let k = if lambda < 450.0 {
            2.2 + 0.3 * (450.0 - lambda) / 70.0
        } else if lambda < 490.0 {
            2.0
        } else if lambda < 550.0 {
            1.8
        } else if lambda < 590.0 {
            1.6
        } else if lambda < 650.0 {
            1.4 - 0.2 * (lambda - 590.0) / 60.0
        } else {
            1.2 - 0.2 * (lambda - 650.0) / 50.0
        };
        
        arr[i] = (r, g, b, k);
    }
    arr
});

/// Convert an SPD sample (per-bin energy) to linear-sRGB premultiplied RGBA.
/// Alpha equals total energy (capped at 1.0) so downstream blending treats it
/// similarly to our old pipeline.
///
/// Automatically selects the best SIMD path for the current platform:
/// - x86_64 AVX2: 4 bins/iter via 256-bit FMA
/// - aarch64 NEON: 2 bins/iter via 128-bit FMA
/// - Scalar fallback for all other targets
#[inline]
pub fn spd_to_rgba(spd: &[f64; NUM_BINS]) -> (f64, f64, f64, f64) {
    spectrum_simd::spd_to_rgba_simd(spd)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_tuple_bits_eq(
        lhs: (f64, f64, f64, f64),
        rhs: (f64, f64, f64, f64),
        label: &str,
    ) {
        assert_eq!(lhs.0.to_bits(), rhs.0.to_bits(), "{label}: R differs");
        assert_eq!(lhs.1.to_bits(), rhs.1.to_bits(), "{label}: G differs");
        assert_eq!(lhs.2.to_bits(), rhs.2.to_bits(), "{label}: B differs");
        assert_eq!(lhs.3.to_bits(), rhs.3.to_bits(), "{label}: A differs");
    }

    #[test]
    fn test_public_spd_to_rgba_is_deterministic() {
        let spd = [0.0, 0.1, 0.2, 0.8, 1.2, 0.6, 0.3, 0.1, 0.0, 0.4, 0.9, 0.7, 0.2, 0.1, 0.0, 0.0];
        let reference = spd_to_rgba(&spd);
        for _ in 0..256 {
            let value = spd_to_rgba(&spd);
            assert_tuple_bits_eq(value, reference, "public_api_determinism");
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(miri)))]
    #[test]
    fn test_public_spd_to_rgba_uses_simd_on_x86_avx2() {
        let spd = [0.0, 0.3, 0.6, 0.9, 1.1, 0.8, 0.4, 0.2, 0.1, 0.5, 0.7, 0.6, 0.3, 0.1, 0.0, 0.0];
        let via_public = spd_to_rgba(&spd);
        let via_simd = crate::spectrum_simd::spd_to_rgba_simd(&spd);
        assert_tuple_bits_eq(via_public, via_simd, "x86_avx2_dispatch");
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", not(miri))))]
    #[test]
    fn test_public_spd_to_rgba_uses_simd_dispatch() {
        let spd = [0.0, 0.2, 0.5, 0.7, 1.0, 0.9, 0.4, 0.1, 0.0, 0.3, 0.6, 0.8, 0.5, 0.2, 0.0, 0.0];
        let via_public = spd_to_rgba(&spd);
        let via_simd = crate::spectrum_simd::spd_to_rgba_simd(&spd);
        assert_tuple_bits_eq(via_public, via_simd, "simd_dispatch");
    }
}
