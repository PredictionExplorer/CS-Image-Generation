//! Numerical comparison of decoded image channels across rendering runs.
//!
//! These metrics describe RGB code-value differences. They neither apply color
//! profile transforms nor establish perceptual equivalence by themselves.

use std::path::Path;

use serde_json::{Value, json};

use super::SilkResult;

/// Compare two equally sized images after decoding each into 16-bit RGB.
///
/// Maximum, mean and root-mean-square errors use a normalized `[0, 1]` scale;
/// `max_abs_error_code_values` additionally reports the largest RGB16 difference.
/// `psnr_db` is null for exact equality, whose peak signal-to-noise ratio is
/// infinite. Eight-bit input channels expand to their corresponding RGB16 values.
/// Alpha channels, if present, are discarded by the RGB conversion.
pub fn compare_images(first: &Path, second: &Path) -> SilkResult<Value> {
    let first = image::open(first)?.to_rgb16();
    let second = image::open(second)?.to_rgb16();
    if first.dimensions() != second.dimensions() {
        return Err(format!(
            "Image dimensions differ: {}x{} versus {}x{}",
            first.width(),
            first.height(),
            second.width(),
            second.height(),
        )
        .into());
    }
    let channels = first.as_raw().len();
    if channels == 0 {
        return Err("Image comparison requires nonempty images".into());
    }
    let mut differing_channels = 0_usize;
    let mut maximum = 0_u16;
    let mut absolute_sum = 0_u128;
    let mut squared_sum = 0_u128;
    for (&a, &b) in first.as_raw().iter().zip(second.as_raw()) {
        let difference = a.abs_diff(b);
        differing_channels += usize::from(difference != 0);
        maximum = maximum.max(difference);
        let integer_difference = u128::from(difference);
        absolute_sum += integer_difference;
        squared_sum += integer_difference * integer_difference;
    }
    let peak = f64::from(u16::MAX);
    let mean_abs_error = absolute_sum as f64 / channels as f64 / peak;
    let rms_error = (squared_sum as f64 / channels as f64).sqrt() / peak;
    let psnr_db = (squared_sum != 0).then(|| -20.0 * rms_error.log10());
    Ok(json!({
        "schema_version": 1,
        "width": first.width(),
        "height": first.height(),
        "comparison_space": "decoded-rgb16-code-values",
        "total_channels": channels,
        "differing_channels": differing_channels,
        "exact": differing_channels == 0,
        "max_abs_error_code_values": maximum,
        "max_abs_error": f64::from(maximum) / peak,
        "mean_abs_error": mean_abs_error,
        "rms_error": rms_error,
        "psnr_db": psnr_db,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn identical_images_have_zero_error_and_unbounded_psnr() {
        let directory = tempfile::tempdir().unwrap();
        let first = directory.path().join("first.png");
        let second = directory.path().join("second.png");
        let pixels = ImageBuffer::from_pixel(2, 1, Rgb([0_u16, 12_345, u16::MAX]));
        pixels.save(&first).unwrap();
        pixels.save(&second).unwrap();
        let report = compare_images(&first, &second).unwrap();
        assert_eq!(report["total_channels"], 6);
        assert_eq!(report["differing_channels"], 0);
        assert_eq!(report["exact"], true);
        assert_eq!(report["max_abs_error_code_values"], 0);
        assert_eq!(report["max_abs_error"], 0.0);
        assert_eq!(report["mean_abs_error"], 0.0);
        assert_eq!(report["rms_error"], 0.0);
        assert!(report["psnr_db"].is_null());
    }

    #[test]
    fn equal_channel_counts_with_different_dimensions_are_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let first = directory.path().join("wide.png");
        let second = directory.path().join("tall.png");
        ImageBuffer::from_pixel(3, 2, Rgb([0_u16; 3])).save(&first).unwrap();
        ImageBuffer::from_pixel(2, 3, Rgb([0_u16; 3])).save(&second).unwrap();
        let error = compare_images(&first, &second).unwrap_err();
        assert!(error.to_string().contains("dimensions differ"));
    }

    #[test]
    fn known_channel_differences_produce_correct_normalized_metrics() {
        let directory = tempfile::tempdir().unwrap();
        let first = directory.path().join("first.png");
        let second = directory.path().join("second.png");
        ImageBuffer::from_pixel(1, 1, Rgb([1000_u16, 2000, 3000])).save(&first).unwrap();
        ImageBuffer::from_pixel(1, 1, Rgb([1001_u16, 1998, 3000])).save(&second).unwrap();
        let report = compare_images(&first, &second).unwrap();
        assert_eq!(report["differing_channels"], 2);
        assert_eq!(report["exact"], false);
        assert_eq!(report["max_abs_error_code_values"], 2);
        let peak = 65_535.0;
        assert!((report["max_abs_error"].as_f64().unwrap() - 2.0 / peak).abs() < 1e-15);
        assert!((report["mean_abs_error"].as_f64().unwrap() - 1.0 / peak).abs() < 1e-15);
        let rms = (5.0_f64 / 3.0).sqrt() / peak;
        assert!((report["rms_error"].as_f64().unwrap() - rms).abs() < 1e-15);
        let psnr = 20.0 * peak.log10() - 10.0 * (5.0_f64 / 3.0).log10();
        assert!((report["psnr_db"].as_f64().unwrap() - psnr).abs() < 1e-10);
    }
}
