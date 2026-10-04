//! Display P3 → sRGB conversion for the main renderer's web derivatives.
//!
//! The main renderer encodes its frames as Display P3: the P3 primaries, the D65 white and the
//! sRGB (IEC 61966-2-1) transfer curve (`PngColorTag::DisplayP3`). The archival HEVC videos keep
//! those frames and are tagged with exactly that (`smpte432`, `iec61966-2-1`). The files a
//! browser shows directly are sRGB instead, the encoding every browser and image viewer
//! reproduces: the web H.264 of `main.mp4` and of the spectral sweep, and the WebP derivatives
//! of the master (which carry no colour profile at all).
//!
//! The conversion is colorimetric (the ICC relative colorimetric intent): decode the transfer
//! curve, convert linear P3 to linear sRGB with [`DISPLAY_P3_TO_SRGB`], clip each channel to
//! `[0, 1]`, encode again. A colour inside the sRGB gamut keeps its exact value; a colour
//! outside it is clipped the way a colour-managed viewer clips P3 content on an sRGB display,
//! so the web files show the same colours as the archival films. Measured on five production
//! masters (2026-10): 17–50% of the lit pixels lie slightly outside sRGB (99th percentile
//! 0.02–0.11 in linear light). A smooth gamut compression would have changed 20–78% of the
//! in-gamut colours and moved most out-of-gamut ones further than clipping does.
//!
//! `master.png` and the spectral PNGs hold the same code values but carry only `gAMA` 1/2.2
//! and `cHRM` (no `cICP`, see `PngColorTag::DisplayP3`), so colour-managed viewers decode them
//! with a pure 2.2 power curve: their deepest shadows show a little darker than in the films
//! and the web files, which follow the declared sRGB curve.
//!
//! Neutral pixels (three equal channels, among them the black that fills most of a frame) are
//! copied unchanged: the two encodings share their white point and transfer curve.

use std::sync::OnceLock;

use image::{ImageBuffer, Rgb};
use rayon::prelude::*;

use super::constants;

/// Linear Display P3 → linear sRGB (BT.709 primaries), both relative to D65: `M_sRGB⁻¹ · M_P3`
/// from the primaries' chromaticities (re-derived by the test `matrix_matches_the_primaries`).
///
/// The blue primaries of the two sets coincide, so red and green do not depend on P3 blue.
pub const DISPLAY_P3_TO_SRGB: [[f64; 3]; 3] = [
    [1.224_940_176_280_559_6, -0.224_940_176_280_559_8, 0.0],
    [-0.042_056_954_709_688_09, 1.042_056_954_709_688, 0.0],
    [-0.019_637_554_590_334_443, -0.078_636_045_550_631_82, 1.098_273_600_140_966_5],
];

/// Pixels converted per parallel task (a few kilobytes of samples each).
const PIXELS_PER_TASK: usize = 4096;

/// The IEC 61966-2-1 decoding of a code value in `[0, 1]` to linear light.
fn srgb_curve_to_linear(encoded: f64) -> f64 {
    if encoded <= 0.040_45 { encoded / 12.92 } else { ((encoded + 0.055) / 1.055).powf(2.4) }
}

/// Linear light, clipped to `[0, 1]`, as a 16-bit IEC 61966-2-1 code value.
fn linear_to_srgb_code(linear: f64) -> u16 {
    let linear = linear.clamp(0.0, 1.0);
    let encoded =
        if linear <= 0.003_130_8 { linear * 12.92 } else { 1.055 * linear.powf(1.0 / 2.4) - 0.055 };
    (encoded * constants::U16_MAX_F64).round() as u16
}

/// Linear light of every 16-bit code value of the shared transfer curve.
fn decoding_table() -> &'static [f64] {
    static TABLE: OnceLock<Vec<f64>> = OnceLock::new();
    TABLE.get_or_init(|| {
        (0..=u16::MAX)
            .map(|code| srgb_curve_to_linear(f64::from(code) / constants::U16_MAX_F64))
            .collect()
    })
}

/// Converts one Display P3 pixel (three code values) into `out`.
fn convert_pixel(table: &[f64], p3: &[u16], out: &mut [u16]) {
    if p3[0] == p3[1] && p3[1] == p3[2] {
        out.copy_from_slice(p3);
        return;
    }
    let linear = [table[usize::from(p3[0])], table[usize::from(p3[1])], table[usize::from(p3[2])]];
    for (row, channel) in DISPLAY_P3_TO_SRGB.iter().zip(out) {
        *channel =
            linear_to_srgb_code(row[0] * linear[0] + row[1] * linear[1] + row[2] * linear[2]);
    }
}

/// Converts interleaved 16-bit Display P3 RGB samples (`[r, g, b, r, g, b, …]`) to sRGB.
///
/// # Panics
/// If the number of samples is not a multiple of three.
#[must_use]
pub fn to_srgb_samples(p3: &[u16]) -> Vec<u16> {
    assert!(p3.len().is_multiple_of(3), "RGB samples come in threes, got {}", p3.len());
    let table = decoding_table();
    let mut srgb = vec![0u16; p3.len()];
    srgb.par_chunks_mut(3 * PIXELS_PER_TASK).zip(p3.par_chunks(3 * PIXELS_PER_TASK)).for_each(
        |(out, input)| {
            for (pixel_out, pixel_in) in out.chunks_exact_mut(3).zip(input.chunks_exact(3)) {
                convert_pixel(table, pixel_in, pixel_out);
            }
        },
    );
    srgb
}

/// [`to_srgb_samples`] of a whole 16-bit Display P3 image.
#[must_use]
pub fn to_srgb_image(p3: &ImageBuffer<Rgb<u16>, Vec<u16>>) -> ImageBuffer<Rgb<u16>, Vec<u16>> {
    let (width, height) = p3.dimensions();
    ImageBuffer::from_raw(width, height, to_srgb_samples(p3.as_raw()))
        .expect("the converted samples have the source's dimensions")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RGB → XYZ matrix of a set of primaries with the D65 white (column-scaled so that
    /// `(1, 1, 1)` maps to the white).
    fn rgb_to_xyz(primaries: [(f64, f64); 3]) -> nalgebra::Matrix3<f64> {
        let column = |(x, y): (f64, f64)| nalgebra::Vector3::new(x / y, 1.0, (1.0 - x - y) / y);
        let unscaled = nalgebra::Matrix3::from_columns(&primaries.map(column));
        let white = column((0.3127, 0.3290));
        let scale = unscaled.lu().solve(&white).expect("independent primaries");
        unscaled * nalgebra::Matrix3::from_diagonal(&scale)
    }

    #[test]
    fn matrix_matches_the_primaries() {
        let p3 = rgb_to_xyz([(0.680, 0.320), (0.265, 0.690), (0.150, 0.060)]);
        let srgb = rgb_to_xyz([(0.640, 0.330), (0.300, 0.600), (0.150, 0.060)]);
        let derived = srgb.try_inverse().expect("invertible") * p3;
        for (row, expected) in DISPLAY_P3_TO_SRGB.iter().enumerate() {
            for (col, value) in expected.iter().enumerate() {
                assert!(
                    (derived[(row, col)] - value).abs() < 1e-12,
                    "[{row}][{col}]: {} vs {value}",
                    derived[(row, col)]
                );
            }
        }
    }

    #[test]
    fn white_maps_to_white_through_the_matrix() {
        for row in DISPLAY_P3_TO_SRGB {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12, "{row:?}");
        }
    }

    #[test]
    fn neutral_pixels_are_unchanged() {
        let greys: Vec<u16> = [0u16, 1, 257, 4_000, 32_768, 65_534, 65_535]
            .iter()
            .flat_map(|&value| [value; 3])
            .collect();
        assert_eq!(to_srgb_samples(&greys), greys);
    }

    /// A colour inside the sRGB gamut keeps its value: sRGB code values sent through the
    /// inverse matrix to P3 come back within rounding.
    #[test]
    fn colours_inside_srgb_round_trip() {
        let inverse = nalgebra::Matrix3::from_row_slice(&DISPLAY_P3_TO_SRGB.concat())
            .try_inverse()
            .expect("invertible");
        let to_code = |linear: f64| {
            let encoded = if linear <= 0.003_130_8 {
                linear * 12.92
            } else {
                1.055 * linear.powf(1.0 / 2.4) - 0.055
            };
            (encoded * 65_535.0).round() as u16
        };
        let srgb_colours: [[u16; 3]; 6] = [
            [45_000, 9_000, 4_000],
            [3_000, 40_000, 12_000],
            [8_000, 12_000, 50_000],
            [65_535, 65_535, 0],
            [20_000, 20_500, 21_000],
            [12, 900, 30],
        ];
        for colour in srgb_colours {
            let linear = nalgebra::Vector3::from_iterator(
                colour.iter().map(|&code| srgb_curve_to_linear(f64::from(code) / 65_535.0)),
            );
            let p3: Vec<u16> = (inverse * linear).iter().map(|&value| to_code(value)).collect();
            let back = to_srgb_samples(&p3);
            for (got, want) in back.iter().zip(colour) {
                assert!(
                    got.abs_diff(want) <= 24,
                    "{colour:?} → P3 {p3:?} → {back:?} (16-bit rounding of the P3 step)"
                );
            }
        }
    }

    /// The P3 primaries lie outside sRGB: they are clipped channel by channel, keeping the
    /// dominant channel at full scale and no negative light.
    #[test]
    fn p3_primaries_are_clipped_into_srgb() {
        let red = to_srgb_samples(&[65_535, 0, 0]);
        assert_eq!(red, vec![65_535, 0, 0]);
        let green = to_srgb_samples(&[0, 65_535, 0]);
        assert_eq!((green[0], green[1]), (0, 65_535));
        let blue = to_srgb_samples(&[0, 0, 65_535]);
        assert_eq!((blue[0], blue[2]), (0, 65_535));
    }

    /// A mid-saturation P3 colour inside sRGB gains saturation when shown correctly: the
    /// unconverted code values (read as sRGB) understate it. This is the visible effect of the
    /// fix on the web files.
    #[test]
    fn conversion_expands_saturation_of_in_gamut_colours() {
        let p3 = [40_000u16, 20_000, 12_000];
        let srgb = to_srgb_samples(&p3);
        assert!(srgb[0] > p3[0], "red rises: {srgb:?}");
        assert!(srgb[1] < p3[1] && srgb[2] < p3[2], "green and blue fall: {srgb:?}");
    }

    #[test]
    fn image_conversion_keeps_dimensions() {
        let image = ImageBuffer::from_raw(2, 1, vec![65_535u16, 0, 0, 100, 100, 100]).unwrap();
        let converted = to_srgb_image(&image);
        assert_eq!(converted.dimensions(), (2, 1));
        assert_eq!(converted.as_raw(), &vec![65_535, 0, 0, 100, 100, 100]);
    }

    #[test]
    fn decoding_table_matches_the_curve() {
        let table = decoding_table();
        assert_eq!(table.len(), 65_536);
        assert_eq!((table[0], table[65_535]), (0.0, 1.0));
        for code in [1u16, 2_650, 32_768, 60_000] {
            let encoded = f64::from(code) / 65_535.0;
            assert_eq!(table[usize::from(code)], srgb_curve_to_linear(encoded));
            assert_eq!(linear_to_srgb_code(table[usize::from(code)]), code);
        }
    }

    #[test]
    #[should_panic(expected = "RGB samples come in threes")]
    fn incomplete_pixels_are_rejected() {
        let _ = to_srgb_samples(&[1, 2]);
    }
}
