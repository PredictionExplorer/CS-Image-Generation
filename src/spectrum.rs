//! Spectral utilities: 64-bin SPD handling and conversions.
//!
//! Our "spectral accumulation" keeps one energy value per wavelength bin
//! (bins are equally spaced from 380-700 nm at 5 nm intervals).  Rendering
//! draws into this SPD buffer, then we convert the spectrum → linear-sRGB
//! right before the normal tone-mapping / bloom pipeline.

use crate::oklab::GamutMapMode;
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

const D65_WHITE: (f64, f64, f64) = (0.95047, 1.0, 1.08883);

#[rustfmt::skip]
const CIE_1931_2_DEG_5NM: [(f64, f64, f64, f64); 65] = [
    (380.0, 0.001368, 0.000039, 0.006450),
    (385.0, 0.002236, 0.000064, 0.010550),
    (390.0, 0.004243, 0.000120, 0.020050),
    (395.0, 0.007650, 0.000217, 0.036210),
    (400.0, 0.014310, 0.000396, 0.067850),
    (405.0, 0.023190, 0.000640, 0.110200),
    (410.0, 0.043510, 0.001210, 0.207400),
    (415.0, 0.077630, 0.002180, 0.371300),
    (420.0, 0.134380, 0.004000, 0.645600),
    (425.0, 0.214770, 0.007300, 1.039050),
    (430.0, 0.283900, 0.011600, 1.385600),
    (435.0, 0.328500, 0.016840, 1.622960),
    (440.0, 0.348280, 0.023000, 1.747060),
    (445.0, 0.348060, 0.029800, 1.782600),
    (450.0, 0.336200, 0.038000, 1.772110),
    (455.0, 0.318700, 0.048000, 1.744100),
    (460.0, 0.290800, 0.060000, 1.669200),
    (465.0, 0.251100, 0.073900, 1.528100),
    (470.0, 0.195360, 0.090980, 1.287640),
    (475.0, 0.142100, 0.112600, 1.041900),
    (480.0, 0.095640, 0.139020, 0.812950),
    (485.0, 0.057950, 0.169300, 0.616200),
    (490.0, 0.032010, 0.208020, 0.465180),
    (495.0, 0.014700, 0.258600, 0.353300),
    (500.0, 0.004900, 0.323000, 0.272000),
    (505.0, 0.002400, 0.407300, 0.212300),
    (510.0, 0.009300, 0.503000, 0.158200),
    (515.0, 0.029100, 0.608200, 0.111700),
    (520.0, 0.063270, 0.710000, 0.078250),
    (525.0, 0.109600, 0.793200, 0.057250),
    (530.0, 0.165500, 0.862000, 0.042160),
    (535.0, 0.225750, 0.914850, 0.029840),
    (540.0, 0.290400, 0.954000, 0.020300),
    (545.0, 0.359700, 0.980300, 0.013400),
    (550.0, 0.433450, 0.994950, 0.008750),
    (555.0, 0.512050, 1.000000, 0.005750),
    (560.0, 0.594500, 0.995000, 0.003900),
    (565.0, 0.678400, 0.978600, 0.002750),
    (570.0, 0.762100, 0.952000, 0.002100),
    (575.0, 0.842500, 0.915400, 0.001800),
    (580.0, 0.916300, 0.870000, 0.001650),
    (585.0, 0.978600, 0.816300, 0.001400),
    (590.0, 1.026300, 0.757000, 0.001100),
    (595.0, 1.056700, 0.694900, 0.001000),
    (600.0, 1.062200, 0.631000, 0.000800),
    (605.0, 1.045600, 0.566800, 0.000600),
    (610.0, 1.002600, 0.503000, 0.000340),
    (615.0, 0.938400, 0.441200, 0.000240),
    (620.0, 0.854450, 0.381000, 0.000190),
    (625.0, 0.751400, 0.321000, 0.000100),
    (630.0, 0.642400, 0.265000, 0.000050),
    (635.0, 0.541900, 0.217000, 0.000030),
    (640.0, 0.447900, 0.175000, 0.000020),
    (645.0, 0.360800, 0.138200, 0.000010),
    (650.0, 0.283500, 0.107000, 0.000000),
    (655.0, 0.218700, 0.081600, 0.000000),
    (660.0, 0.164900, 0.061000, 0.000000),
    (665.0, 0.121200, 0.044580, 0.000000),
    (670.0, 0.087400, 0.032000, 0.000000),
    (675.0, 0.063600, 0.023200, 0.000000),
    (680.0, 0.046770, 0.017000, 0.000000),
    (685.0, 0.032900, 0.011920, 0.000000),
    (690.0, 0.022700, 0.008210, 0.000000),
    (695.0, 0.015840, 0.005723, 0.000000),
    (700.0, 0.011359, 0.004102, 0.000000),
];

#[inline]
fn cie_xyz_at_nm(lambda: f64) -> (f64, f64, f64) {
    if !(LAMBDA_START..=LAMBDA_END).contains(&lambda) {
        return (0.0, 0.0, 0.0);
    }

    let position = ((lambda - LAMBDA_START) / 5.0).clamp(0.0, 64.0);
    let left = position.floor() as usize;
    let right = (left + 1).min(CIE_1931_2_DEG_5NM.len() - 1);
    let t = position.fract();
    let (_, x0, y0, z0) = CIE_1931_2_DEG_5NM[left];
    let (_, x1, y1, z1) = CIE_1931_2_DEG_5NM[right];

    (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, z0 + (z1 - z0) * t)
}

#[inline]
fn raw_bin_xyz(bin: usize) -> (f64, f64, f64) {
    let center = wavelength_nm_for_bin(bin);
    let left = (center - 2.5).max(LAMBDA_START);
    let right = (center + 2.5).min(LAMBDA_END);
    let (xl, yl, zl) = cie_xyz_at_nm(left);
    let (xc, yc, zc) = cie_xyz_at_nm(center);
    let (xr, yr, zr) = cie_xyz_at_nm(right);

    ((xl + 4.0 * xc + xr) / 6.0, (yl + 4.0 * yc + yr) / 6.0, (zl + 4.0 * zc + zr) / 6.0)
}

#[inline]
fn tone_k_for_wavelength(lambda: f64) -> f64 {
    let normalized = ((lambda - LAMBDA_START) / (LAMBDA_END - LAMBDA_START)).clamp(0.0, 1.0);
    2.05 - 0.68 * normalized
}

/// Convert D65-relative CIE XYZ to linear sRGB.
#[must_use]
#[inline]
pub fn xyz_to_linear_srgb(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let r = 3.240_454_2 * x - 1.537_138_5 * y - 0.498_531_4 * z;
    let g = -0.969_266 * x + 1.876_010_8 * y + 0.041_556 * z;
    let b = 0.055_643_4 * x - 0.204_025_9 * y + 1.057_225_2 * z;

    (r, g, b)
}

/// Convert D65-relative CIE XYZ to linear Rec.2020.
#[must_use]
#[inline]
pub fn xyz_to_linear_rec2020(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let r = 1.716_651_187_971_268 * x - 0.355_670_783_776_392 * y - 0.253_366_281_373_66 * z;
    let g = -0.666_684_351_832_489 * x + 1.616_481_236_634_939 * y + 0.015_768_545_813_911_1 * z;
    let b = 0.017_639_857_445_310_8 * x - 0.042_770_613_257_808_5 * y + 0.942_103_121_235_474 * z;

    (r, g, b)
}

/// Convert linear Rec.2020 to D65-relative CIE XYZ.
#[must_use]
#[inline]
pub fn linear_rec2020_to_xyz(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let x = 0.636_958_048_301_291_4 * r + 0.144_616_903_586_208_3 * g + 0.168_880_975_164_172_1 * b;
    let y = 0.262_700_212_011_267_1 * r + 0.677_998_071_518_870_8 * g + 0.059_301_716_469_862 * b;
    let z = 0.028_072_693_049_087_4 * g + 1.060_985_057_710_791 * b;

    (x, y, z)
}

/// Convert linear Rec.2020 to linear Display P3.
#[must_use]
#[inline]
pub fn linear_rec2020_to_display_p3(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let (x, y, z) = linear_rec2020_to_xyz(r, g, b);
    let p3_r = 2.493_496_911_941_425 * x - 0.931_383_617_919_124 * y - 0.402_710_784_450_717 * z;
    let p3_g = -0.829_488_969_561_574 * x + 1.762_664_060_318_346 * y + 0.023_624_685_841_943 * z;
    let p3_b = 0.035_845_830_243_784 * x - 0.076_172_389_268_041 * y + 0.956_884_524_007_687 * z;

    (p3_r, p3_g, p3_b)
}

/// Convert linear sRGB to linear Display P3 through D65-relative CIE XYZ.
#[must_use]
#[inline]
pub fn linear_srgb_to_display_p3(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let x = 0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b;
    let y = 0.212_672_9 * r + 0.715_152_2 * g + 0.072_175 * b;
    let z = 0.019_333_9 * r + 0.119_192 * g + 0.950_304_1 * b;
    let p3_r = 2.493_496_911_941_425 * x - 0.931_383_617_919_124 * y - 0.402_710_784_450_717 * z;
    let p3_g = -0.829_488_969_561_574 * x + 1.762_664_060_318_346 * y + 0.023_624_685_841_943 * z;
    let p3_b = 0.035_845_830_243_784 * x - 0.076_172_389_268_041 * y + 0.956_884_524_007_687 * z;

    (p3_r, p3_g, p3_b)
}

/// CIE-derived linear-sRGB colour for a display tint at the given wavelength.
#[must_use]
pub fn wavelength_to_rgb(lambda: f64) -> (f64, f64, f64) {
    let (x, y, z) = cie_xyz_at_nm(lambda);
    let max_xyz = x.max(y).max(z);
    if max_xyz <= 0.0 {
        return (0.0, 0.0, 0.0);
    }

    let (mut r, mut g, mut b) = xyz_to_linear_srgb(x / max_xyz, y / max_xyz, z / max_xyz);
    let min_channel = r.min(g).min(b);
    if min_channel < 0.0 {
        r -= min_channel;
        g -= min_channel;
        b -= min_channel;
    }
    let max_channel = r.max(g).max(b);
    if max_channel > 1.0 {
        r /= max_channel;
        g /= max_channel;
        b /= max_channel;
    }

    let (r, g, b) = GamutMapMode::Clamp.map_to_gamut(r, g, b);
    let factor = if (380.0..420.0).contains(&lambda) {
        0.35 + 0.65 * (lambda - 380.0) / 40.0
    } else if (420.0..645.0).contains(&lambda) {
        1.0
    } else if (645.0..=700.0).contains(&lambda) {
        0.35 + 0.65 * (700.0 - lambda) / 55.0
    } else {
        0.0
    };

    (r * factor, g * factor, b * factor)
}

/// Combined CIE XYZ lookup table for cache-friendly SPD conversion.
/// Stores (X, Y, Z, `tone_k`) in a single cache line for better performance.
pub static BIN_XYZ_LUT: std::sync::LazyLock<[(f64, f64, f64, f64); NUM_BINS]> =
    std::sync::LazyLock::new(|| {
        let mut arr = [(0.0, 0.0, 0.0, 0.0); NUM_BINS];
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_z = 0.0;
        for i in 0..NUM_BINS {
            let (x, y, z) = raw_bin_xyz(i);
            sum_x += x;
            sum_y += y;
            sum_z += z;
        }

        let norm_x = if sum_x > 0.0 { D65_WHITE.0 / sum_x } else { 1.0 };
        let norm_y = if sum_y > 0.0 { D65_WHITE.1 / sum_y } else { 1.0 };
        let norm_z = if sum_z > 0.0 { D65_WHITE.2 / sum_z } else { 1.0 };

        for (i, entry) in arr.iter_mut().enumerate() {
            let lambda = wavelength_nm_for_bin(i);
            let (x, y, z) = raw_bin_xyz(i);
            *entry = (x * norm_x, y * norm_y, z * norm_z, tone_k_for_wavelength(lambda));
        }
        arr
    });

/// CIE-derived display RGB lookup for diagnostics and legacy callers.
pub static BIN_COMBINED_LUT: std::sync::LazyLock<[(f64, f64, f64, f64); NUM_BINS]> =
    std::sync::LazyLock::new(|| {
        let mut arr = [(0.0, 0.0, 0.0, 0.0); NUM_BINS];
        for (i, entry) in arr.iter_mut().enumerate() {
            let lambda = wavelength_nm_for_bin(i);
            let (r, g, b) = wavelength_to_rgb(lambda);
            *entry = (r, g, b, tone_k_for_wavelength(lambda));
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
    spectrum_simd::spd_to_rgba_simd(spd)
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
