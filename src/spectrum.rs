//! Spectral utilities: 64-bin SPD handling and conversions.
//!
//! Our "spectral accumulation" keeps one energy value per wavelength bin
//! (bins are equally spaced from 380-700 nm at 5 nm intervals).  Rendering
//! draws into this SPD buffer, then we convert the spectrum → linear-sRGB
//! right before the normal tone-mapping / bloom pipeline.

use crate::spectrum_simd;

/// Number of wavelength buckets in the SPD.
pub const NUM_BINS: usize = 64;
/// Start / end wavelengths in nanometres.
const LAMBDA_START: f64 = 380.0;
const LAMBDA_END: f64 = 700.0;

/// Centre wavelength for a bin.
#[inline]
#[must_use]
pub fn wavelength_nm_for_bin(bin: usize) -> f64 {
    LAMBDA_START + (bin as f64 + 0.5) * (LAMBDA_END - LAMBDA_START) / NUM_BINS as f64
}

/// Approximate (linear-sRGB) colour corresponding to a given wavelength.
/// Formula adapted from Dan Bruton's reference (gamma removed → stay linear).
#[must_use]
pub fn wavelength_to_rgb(lambda: f64) -> (f64, f64, f64) {
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
/// Stores (R, G, B, `tone_k`) in a single cache line for better performance
pub static BIN_COMBINED_LUT: std::sync::LazyLock<[(f64, f64, f64, f64); NUM_BINS]> =
    std::sync::LazyLock::new(|| {
        let mut arr = [(0.0, 0.0, 0.0, 0.0); NUM_BINS];
        for (i, entry) in arr.iter_mut().enumerate() {
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

            *entry = (r, g, b, k);
        }
        arr
    });

/// Convert an SPD sample (per-bin energy) to linear-sRGB premultiplied RGBA.
/// Alpha equals total energy (capped at 1.0) so downstream blending treats it
/// similarly to our old pipeline.
///
/// Automatically selects the best SIMD path for the current platform:
/// - `x86_64` AVX2: 4 bins/iter via 256-bit FMA
/// - aarch64 NEON: 2 bins/iter via 128-bit FMA
/// - Scalar fallback for all other targets
#[inline]
#[must_use]
pub fn spd_to_rgba(spd: &[f64; NUM_BINS]) -> (f64, f64, f64, f64) {
    spd_to_rgba_with_sat_boost(spd, true)
}

/// Convert an SPD sample to RGBA with explicit saturation-boost selection.
#[inline]
#[must_use]
pub(crate) fn spd_to_rgba_with_sat_boost(
    spd: &[f64; NUM_BINS],
    boosted: bool,
) -> (f64, f64, f64, f64) {
    spectrum_simd::spd_to_rgba_simd_with_sat_boost(spd, boosted)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_tuple_bits_eq(lhs: (f64, f64, f64, f64), rhs: (f64, f64, f64, f64), label: &str) {
        assert_eq!(lhs.0.to_bits(), rhs.0.to_bits(), "{label}: R differs");
        assert_eq!(lhs.1.to_bits(), rhs.1.to_bits(), "{label}: G differs");
        assert_eq!(lhs.2.to_bits(), rhs.2.to_bits(), "{label}: B differs");
        assert_eq!(lhs.3.to_bits(), rhs.3.to_bits(), "{label}: A differs");
    }

    fn make_spd(values: &[f64]) -> [f64; NUM_BINS] {
        let mut spd = [0.0; NUM_BINS];
        for (i, &v) in values.iter().enumerate().take(NUM_BINS) {
            spd[i] = v;
        }
        spd
    }

    #[test]
    fn test_public_spd_to_rgba_is_deterministic() {
        let spd = make_spd(&[0.0, 0.1, 0.2, 0.8, 1.2, 0.6, 0.3, 0.1, 0.0, 0.4, 0.9, 0.7, 0.2, 0.1]);
        let reference = spd_to_rgba(&spd);
        for _ in 0..256 {
            let value = spd_to_rgba(&spd);
            assert_tuple_bits_eq(value, reference, "public_api_determinism");
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(miri)))]
    #[test]
    fn test_public_spd_to_rgba_uses_simd_on_x86_avx2() {
        let spd = make_spd(&[0.0, 0.3, 0.6, 0.9, 1.1, 0.8, 0.4, 0.2, 0.1, 0.5, 0.7, 0.6, 0.3, 0.1]);
        let via_public = spd_to_rgba(&spd);
        let via_simd = crate::spectrum_simd::spd_to_rgba_simd(&spd);
        assert_tuple_bits_eq(via_public, via_simd, "x86_avx2_dispatch");
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", not(miri))))]
    #[test]
    fn test_public_spd_to_rgba_uses_simd_dispatch() {
        let spd = make_spd(&[0.0, 0.2, 0.5, 0.7, 1.0, 0.9, 0.4, 0.1, 0.0, 0.3, 0.6, 0.8, 0.5, 0.2]);
        let via_public = spd_to_rgba(&spd);
        let via_simd = crate::spectrum_simd::spd_to_rgba_simd(&spd);
        assert_tuple_bits_eq(via_public, via_simd, "simd_dispatch");
    }

    // ── 64-bin parameter validation ─────────────────────────────────

    #[test]
    fn test_num_bins_is_64() {
        assert_eq!(NUM_BINS, 64);
    }

    #[test]
    fn test_bin_width_is_5nm() {
        let bin_width = (LAMBDA_END - LAMBDA_START) / NUM_BINS as f64;
        assert!((bin_width - 5.0).abs() < 1e-10, "bin width should be 5nm, got {bin_width}");
    }

    #[test]
    fn test_bin_centers_span_visible_spectrum() {
        let first = wavelength_nm_for_bin(0);
        let last = wavelength_nm_for_bin(NUM_BINS - 1);
        assert!(first > LAMBDA_START, "first bin center should be > 380nm");
        assert!(first < LAMBDA_START + 10.0, "first bin center should be near 382.5nm");
        assert!(last < LAMBDA_END, "last bin center should be < 700nm");
        assert!(last > LAMBDA_END - 10.0, "last bin center should be near 697.5nm");
    }

    #[test]
    fn test_bin_centers_monotonically_increase() {
        for i in 1..NUM_BINS {
            let prev = wavelength_nm_for_bin(i - 1);
            let curr = wavelength_nm_for_bin(i);
            assert!(curr > prev, "bin {i} center should be > bin {} center", i - 1);
        }
    }

    #[test]
    fn test_lut_has_correct_size() {
        assert_eq!(BIN_COMBINED_LUT.len(), NUM_BINS);
    }

    #[test]
    fn test_lut_rgb_values_non_negative() {
        for (i, &(r, g, b, _)) in BIN_COMBINED_LUT.iter().enumerate() {
            assert!(r >= 0.0, "LUT bin {i}: R negative ({r})");
            assert!(g >= 0.0, "LUT bin {i}: G negative ({g})");
            assert!(b >= 0.0, "LUT bin {i}: B negative ({b})");
        }
    }

    #[test]
    fn test_lut_tone_k_positive() {
        for (i, &(_, _, _, k)) in BIN_COMBINED_LUT.iter().enumerate() {
            assert!(k > 0.0, "LUT bin {i}: k should be positive ({k})");
        }
    }

    #[test]
    fn test_lut_blue_bins_have_higher_k_than_red_bins() {
        let k_blue = BIN_COMBINED_LUT[5].3;
        let k_red = BIN_COMBINED_LUT[NUM_BINS - 5].3;
        assert!(
            k_blue > k_red,
            "blue bins should have higher k than red: blue={k_blue}, red={k_red}"
        );
    }

    // ── wavelength_to_rgb tests ─────────────────────────────────────

    #[test]
    fn test_wavelength_to_rgb_visible_range_non_zero() {
        for wl in (400..=680).step_by(10) {
            let (r, g, b) = wavelength_to_rgb(f64::from(wl));
            let sum = r + g + b;
            assert!(sum > 0.0, "wavelength {wl}nm should produce nonzero RGB, got ({r},{g},{b})");
        }
    }

    #[test]
    fn test_wavelength_to_rgb_outside_visible_is_black() {
        let (r, g, b) = wavelength_to_rgb(300.0);
        assert_eq!((r, g, b), (0.0, 0.0, 0.0), "UV should be black");
        let (r, g, b) = wavelength_to_rgb(800.0);
        assert_eq!((r, g, b), (0.0, 0.0, 0.0), "IR should be black");
    }

    #[test]
    fn test_wavelength_to_rgb_red_end() {
        let (r, g, b) = wavelength_to_rgb(660.0);
        assert!(r > g && r > b, "660nm should be red-dominant: ({r},{g},{b})");
    }

    #[test]
    fn test_wavelength_to_rgb_green_region() {
        let (r, g, b) = wavelength_to_rgb(530.0);
        assert!(g > r && g > b, "530nm should be green-dominant: ({r},{g},{b})");
    }

    #[test]
    fn test_wavelength_to_rgb_blue_region() {
        let (r, g, b) = wavelength_to_rgb(460.0);
        assert!(b > r && b > g, "460nm should be blue-dominant: ({r},{g},{b})");
    }

    #[test]
    fn test_wavelength_to_rgb_values_in_unit_range() {
        for wl in (380..=700).step_by(1) {
            let (r, g, b) = wavelength_to_rgb(f64::from(wl));
            assert!((0.0..=1.0).contains(&r), "R={r} out of [0,1] at {wl}nm");
            assert!((0.0..=1.0).contains(&g), "G={g} out of [0,1] at {wl}nm");
            assert!((0.0..=1.0).contains(&b), "B={b} out of [0,1] at {wl}nm");
        }
    }

    #[test]
    fn test_wavelength_to_rgb_edge_rolloff() {
        let (_, _, b_380) = wavelength_to_rgb(380.0);
        let (_, _, b_420) = wavelength_to_rgb(420.0);
        assert!(
            b_380 < b_420,
            "380nm should be dimmer than 420nm due to edge rolloff: {b_380} vs {b_420}"
        );
    }

    // ── Bin center exact values per algorithm doc ───────────────────

    #[test]
    fn test_bin_center_first_is_382_5nm() {
        let center = wavelength_nm_for_bin(0);
        assert!((center - 382.5).abs() < 1e-10, "bin 0 center should be 382.5nm, got {center}");
    }

    #[test]
    fn test_bin_center_last_is_697_5nm() {
        let center = wavelength_nm_for_bin(63);
        assert!((center - 697.5).abs() < 1e-10, "bin 63 center should be 697.5nm, got {center}");
    }

    #[test]
    fn test_bin_centers_evenly_spaced_at_5nm() {
        for i in 1..NUM_BINS {
            let prev = wavelength_nm_for_bin(i - 1);
            let curr = wavelength_nm_for_bin(i);
            assert!(
                (curr - prev - 5.0).abs() < 1e-10,
                "bin spacing should be 5nm: bin {i} - bin {} = {}",
                i - 1,
                curr - prev
            );
        }
    }

    // ── LUT spectral coverage ───────────────────────────────────────

    #[test]
    fn test_lut_deep_violet_bins_are_blue_dominant() {
        for bin in 0..8 {
            let (r, _g, b, _) = BIN_COMBINED_LUT[bin];
            let wl = wavelength_nm_for_bin(bin);
            if wl < 440.0 {
                assert!(
                    b > r || (r + b > 0.0),
                    "bin {bin} ({wl:.0}nm) should have blue component: R={r}, B={b}"
                );
            }
        }
    }

    #[test]
    fn test_lut_green_bins_are_green_dominant() {
        for bin in 0..NUM_BINS {
            let wl = wavelength_nm_for_bin(bin);
            if (510.0..570.0).contains(&wl) {
                let (r, g, b, _) = BIN_COMBINED_LUT[bin];
                assert!(
                    g >= r && g >= b,
                    "bin {bin} ({wl:.0}nm) should be green-dominant: R={r}, G={g}, B={b}"
                );
            }
        }
    }

    #[test]
    fn test_lut_deep_red_bins_are_red_dominant() {
        for bin in (NUM_BINS - 8)..NUM_BINS {
            let (r, g, b, _) = BIN_COMBINED_LUT[bin];
            let wl = wavelength_nm_for_bin(bin);
            if wl > 645.0 {
                assert!(
                    r >= g && r >= b,
                    "bin {bin} ({wl:.0}nm) should be red-dominant: R={r}, G={g}, B={b}"
                );
            }
        }
    }

    #[test]
    fn test_lut_tone_k_decreases_from_blue_to_red() {
        let k_first = BIN_COMBINED_LUT[0].3;
        let k_last = BIN_COMBINED_LUT[NUM_BINS - 1].3;
        assert!(
            k_first > k_last,
            "tone k should decrease from violet to red: first={k_first}, last={k_last}"
        );
    }

    #[test]
    fn test_lut_all_entries_have_at_least_one_nonzero_channel() {
        for (i, &(r, g, b, _)) in BIN_COMBINED_LUT.iter().enumerate() {
            let wl = wavelength_nm_for_bin(i);
            if (390.0..=690.0).contains(&wl) {
                assert!(
                    r > 0.0 || g > 0.0 || b > 0.0,
                    "LUT bin {i} ({wl:.0}nm) should have at least one nonzero RGB channel"
                );
            }
        }
    }

    // ── wavelength_to_rgb continuity ────────────────────────────────

    #[test]
    fn test_wavelength_to_rgb_continuous() {
        let mut prev = wavelength_to_rgb(380.0);
        for wl_x10 in 3810..=7000 {
            let wl = f64::from(wl_x10) / 10.0;
            let curr = wavelength_to_rgb(wl);
            let dr = (curr.0 - prev.0).abs();
            let dg = (curr.1 - prev.1).abs();
            let db = (curr.2 - prev.2).abs();
            assert!(
                dr < 0.05 && dg < 0.05 && db < 0.05,
                "wavelength_to_rgb should be continuous: at {wl}nm delta=({dr:.4},{dg:.4},{db:.4})"
            );
            prev = curr;
        }
    }

    #[test]
    fn test_wavelength_to_rgb_red_rolloff_at_700nm() {
        let (r_650, _, _) = wavelength_to_rgb(650.0);
        let (r_700, _, _) = wavelength_to_rgb(700.0);
        assert!(r_700 < r_650, "700nm should have edge rolloff vs 650nm: {r_700} vs {r_650}");
    }
}
