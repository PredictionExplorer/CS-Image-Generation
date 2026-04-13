//! Shared constants for spectral color processing and wavelength conversions
//!
//! This module centralizes all spectral-related constants used throughout the rendering
//! pipeline, ensuring consistency and eliminating duplication.

use crate::spectrum::NUM_BINS;

/// Start of visible spectrum in nanometers
pub const LAMBDA_START: f64 = 380.0;

/// End of visible spectrum in nanometers  
pub const LAMBDA_END: f64 = 700.0;

/// Total range of visible spectrum
pub const LAMBDA_RANGE: f64 = LAMBDA_END - LAMBDA_START;

/// Width of each spectral bin in nanometers
pub const BIN_WIDTH: f64 = LAMBDA_RANGE / NUM_BINS as f64;

/// Convert wavelength (nm) to fractional bin position
///
/// # Arguments
/// * `wavelength` - Wavelength in nanometers, should be in range [`LAMBDA_START`, `LAMBDA_END`]
///
/// # Returns
/// Fractional bin position, clamped to valid range [0, NUM_BINS-1]
#[must_use]
#[inline]
pub fn wavelength_to_bin(wavelength: f64) -> f64 {
    ((wavelength - LAMBDA_START) / BIN_WIDTH).clamp(0.0, (NUM_BINS - 1) as f64)
}

/// Convert bin index to center wavelength (nm)
///
/// # Arguments
/// * `bin` - Bin index in range [0, `NUM_BINS`)
///
/// # Returns
/// Center wavelength of the bin in nanometers
#[cfg(test)]
#[must_use]
#[inline]
pub fn bin_to_wavelength(bin: usize) -> f64 {
    LAMBDA_START + (bin as f64 + 0.5) * BIN_WIDTH
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavelength_to_bin() {
        // Test boundaries
        assert_eq!(wavelength_to_bin(LAMBDA_START), 0.0);
        assert!(wavelength_to_bin(LAMBDA_END) <= (NUM_BINS - 1) as f64);

        // Test middle of spectrum
        let mid_wavelength = f64::midpoint(LAMBDA_START, LAMBDA_END);
        let mid_bin = wavelength_to_bin(mid_wavelength);
        assert!(mid_bin > 0.0 && mid_bin < (NUM_BINS - 1) as f64);
    }

    #[test]
    fn test_bin_to_wavelength() {
        // Test boundaries
        assert!(bin_to_wavelength(0) >= LAMBDA_START);
        assert!(bin_to_wavelength(NUM_BINS - 1) <= LAMBDA_END);

        // Test that bins are evenly spaced
        let w0 = bin_to_wavelength(0);
        let w1 = bin_to_wavelength(1);
        let spacing = w1 - w0;
        assert!((spacing - BIN_WIDTH).abs() < 1e-10);
    }

    #[test]
    fn test_roundtrip_conversion() {
        for bin in 0..NUM_BINS {
            let wavelength = bin_to_wavelength(bin);
            let bin_f = wavelength_to_bin(wavelength);
            assert!((bin_f - bin as f64).abs() < 0.51);
        }
    }

    #[test]
    fn test_bin_width_is_5nm_for_64_bins() {
        assert!(
            (BIN_WIDTH - 5.0).abs() < 1e-10,
            "BIN_WIDTH should be 5.0nm for 64 bins, got {BIN_WIDTH}"
        );
    }

    #[test]
    fn test_lambda_range_is_320() {
        assert!((LAMBDA_RANGE - 320.0).abs() < 1e-10);
    }

    #[test]
    fn test_wavelength_to_bin_clamped_below() {
        assert_eq!(wavelength_to_bin(100.0), 0.0);
    }

    #[test]
    fn test_wavelength_to_bin_clamped_above() {
        assert_eq!(wavelength_to_bin(1000.0), (NUM_BINS - 1) as f64);
    }

    #[test]
    fn test_wavelength_to_bin_monotonic() {
        let mut prev = -1.0;
        for wl in (380..=700).step_by(1) {
            let bin = wavelength_to_bin(f64::from(wl));
            assert!(bin >= prev, "wavelength_to_bin should be monotonic");
            prev = bin;
        }
    }

    #[test]
    fn test_all_64_bins_reachable() {
        let mut reached = [false; NUM_BINS];
        for wl_x10 in 3800..=7000 {
            let wl = f64::from(wl_x10) / 10.0;
            let bin = wavelength_to_bin(wl).round() as usize;
            if bin < NUM_BINS {
                reached[bin] = true;
            }
        }
        for (i, &r) in reached.iter().enumerate() {
            assert!(r, "bin {i} should be reachable by some wavelength");
        }
    }

    #[test]
    fn test_bin_width_times_num_bins_equals_range() {
        let product = BIN_WIDTH * NUM_BINS as f64;
        assert!(
            (product - LAMBDA_RANGE).abs() < 1e-10,
            "BIN_WIDTH * NUM_BINS should equal LAMBDA_RANGE: {product} vs {LAMBDA_RANGE}"
        );
    }

    #[test]
    fn test_wavelength_to_bin_at_exact_bin_centers() {
        // bin_to_wavelength(bin) = LAMBDA_START + (bin + 0.5) * BIN_WIDTH
        // wavelength_to_bin(wl)  = ((wl - LAMBDA_START) / BIN_WIDTH).clamp(0, NUM_BINS-1)
        // So the raw result is bin + 0.5, but the last bin (63.5) gets clamped to 63.
        for bin in 0..NUM_BINS {
            let center = bin_to_wavelength(bin);
            let result = wavelength_to_bin(center);
            let raw_expected = bin as f64 + 0.5;
            let clamped_expected = raw_expected.min((NUM_BINS - 1) as f64);
            assert!(
                (result - clamped_expected).abs() < 1e-10,
                "bin center {center}nm: expected {clamped_expected}, got {result}",
            );
        }
    }

    #[test]
    fn test_wavelength_to_bin_at_bin_edges() {
        let edge = LAMBDA_START + BIN_WIDTH;
        let bin = wavelength_to_bin(edge);
        assert!(
            (bin - 1.0).abs() < 0.01,
            "wavelength at first bin edge should map to bin ~1.0, got {bin}"
        );
    }

    #[test]
    fn test_wavelength_to_bin_linearity() {
        let wl1 = 400.0;
        let wl2 = 600.0;
        let b1 = wavelength_to_bin(wl1);
        let b2 = wavelength_to_bin(wl2);
        let expected_diff = (wl2 - wl1) / BIN_WIDTH;
        assert!(
            (b2 - b1 - expected_diff).abs() < 0.01,
            "wavelength_to_bin should be linear: diff={}, expected={}",
            b2 - b1,
            expected_diff
        );
    }
}
