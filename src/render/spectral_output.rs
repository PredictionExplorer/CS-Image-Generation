//! Spectral gallery and sweep video generation.
//!
//! After the main rendering pass accumulates the full SPD buffer, this module
//! produces per-bin wavelength images (the "spectral gallery") and a single
//! spectral sweep video that cycles through all 64 bins from violet to red.

use super::constants;
use super::error::{RenderError, Result};
use super::video::{VideoEncodingOptions, create_video_from_frames_singlepass};
use crate::spectrum::{NUM_BINS, wavelength_nm_for_bin, wavelength_to_rgb};
use image::{ImageBuffer, Rgb};
use rayon::prelude::*;
use tracing::info;

/// Pre-computed per-bin float RGB images used by both the gallery and sweep video.
pub struct BinBuffers {
    pub(crate) buffers: Vec<Vec<[f32; 3]>>,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl BinBuffers {
    /// Build 64 bin images from the accumulated SPD buffer.
    ///
    /// Each bin image normalizes that bin's energy across all pixels, tints by the
    /// bin's wavelength color, and applies display gamma.
    #[must_use]
    pub fn new(accum_spd: &[[f64; NUM_BINS]], width: usize, height: usize) -> Self {
        let pixel_count = width * height;
        assert_eq!(accum_spd.len(), pixel_count);

        let inv_gamma = 1.0 / constants::DISPLAY_GAMMA;

        let buffers: Vec<Vec<[f32; 3]>> = (0..NUM_BINS)
            .into_par_iter()
            .map(|bin| {
                let wavelength = wavelength_nm_for_bin(bin);
                let (tint_r, tint_g, tint_b) = wavelength_to_rgb(wavelength);

                let max_val =
                    accum_spd.iter().map(|spd| spd[bin]).fold(0.0f64, f64::max).max(1e-10);

                let mut buf = vec![[0.0f32; 3]; pixel_count];
                for (i, pixel) in buf.iter_mut().enumerate() {
                    let normalized = (accum_spd[i][bin] / max_val).clamp(0.0, 1.0);
                    pixel[0] = ((normalized * tint_r).powf(inv_gamma)) as f32;
                    pixel[1] = ((normalized * tint_g).powf(inv_gamma)) as f32;
                    pixel[2] = ((normalized * tint_b).powf(inv_gamma)) as f32;
                }
                buf
            })
            .collect();

        Self { buffers, width, height }
    }

    /// Number of pixels (`width * height`) per bin buffer.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }
}

impl std::fmt::Debug for BinBuffers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BinBuffers")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("num_bins", &self.buffers.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Spectral Gallery
// ---------------------------------------------------------------------------

/// Generate 64 per-bin 16-bit PNGs into `output_dir`.
pub fn generate_spectral_gallery(
    accum_spd: &[[f64; NUM_BINS]],
    width: u32,
    height: u32,
    output_dir: &str,
) -> Result<()> {
    info!("Building BinBuffers ({NUM_BINS} bins)...");
    let bin_buffers = BinBuffers::new(accum_spd, width as usize, height as usize);

    info!("Generating spectral gallery ({NUM_BINS} bin images)...");
    (0..NUM_BINS).into_par_iter().try_for_each(|bin| {
        let wavelength = wavelength_nm_for_bin(bin);
        let filename = format!("{output_dir}/{bin:02}_{wavelength:.0}nm.png");
        save_bin_image(&bin_buffers.buffers[bin], width, height, &filename)
    })?;

    info!("   Spectral gallery complete ({NUM_BINS} images)");
    Ok(())
}

fn save_bin_image(buf: &[[f32; 3]], width: u32, height: u32, path: &str) -> Result<()> {
    use crate::utils::f64_to_u16_saturating;
    let pixel_count = (width * height) as usize;
    let mut raw = Vec::with_capacity(pixel_count * 3);
    for pixel in buf {
        raw.push(f64_to_u16_saturating(
            f64::from(pixel[0].clamp(0.0, 1.0)) * super::constants::U16_MAX_F64,
        ));
        raw.push(f64_to_u16_saturating(
            f64::from(pixel[1].clamp(0.0, 1.0)) * super::constants::U16_MAX_F64,
        ));
        raw.push(f64_to_u16_saturating(
            f64::from(pixel[2].clamp(0.0, 1.0)) * super::constants::U16_MAX_F64,
        ));
    }

    let img: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::from_raw(width, height, raw)
        .ok_or_else(|| {
            RenderError::InvalidConfig("Failed to create bin image buffer".to_string())
        })?;

    let dyn_img = image::DynamicImage::ImageRgb16(img);
    dyn_img.save(path).map_err(|e| RenderError::ImageEncoding(e.to_string()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Spectral Sweep Video
// ---------------------------------------------------------------------------

/// Per-pixel output format for frame interpolation.
trait FramePixel: Copy + Default + Send + Sync {
    fn from_rgb(r: f32, g: f32, b: f32) -> Self;
}

impl FramePixel for [u16; 3] {
    #[inline]
    fn from_rgb(r: f32, g: f32, b: f32) -> Self {
        [
            (r.clamp(0.0, 1.0) * super::constants::U16_MAX_F64 as f32).round() as u16,
            (g.clamp(0.0, 1.0) * super::constants::U16_MAX_F64 as f32).round() as u16,
            (b.clamp(0.0, 1.0) * super::constants::U16_MAX_F64 as f32).round() as u16,
        ]
    }
}

/// Interpolate between two adjacent bin images at fractional bin index (with wrapping).
fn lerp_bins_frame(bin_buffers: &BinBuffers, bin_f: f64, output: &mut Vec<[u16; 3]>) {
    let pixel_count = bin_buffers.pixel_count();
    output.resize(pixel_count, [0u16; 3]);

    let wrapped = ((bin_f % NUM_BINS as f64) + NUM_BINS as f64) % NUM_BINS as f64;
    let lo = wrapped.floor() as usize % NUM_BINS;
    let hi = (lo + 1) % NUM_BINS;
    let t = wrapped.fract() as f32;
    let one_minus_t = 1.0 - t;

    let lo_buf = &bin_buffers.buffers[lo];
    let hi_buf = &bin_buffers.buffers[hi];

    output.par_iter_mut().enumerate().for_each(|(i, pixel)| {
        let r = lo_buf[i][0] * one_minus_t + hi_buf[i][0] * t;
        let g = lo_buf[i][1] * one_minus_t + hi_buf[i][1] * t;
        let b = lo_buf[i][2] * one_minus_t + hi_buf[i][2] * t;
        *pixel = <[u16; 3]>::from_rgb(r, g, b);
    });
}

/// Generate a single spectral sweep video (violet to red) at the given path.
pub fn generate_spectral_sweep_video(
    accum_spd: &[[f64; NUM_BINS]],
    width: u32,
    height: u32,
    output_path: &str,
    fast_encode: bool,
) -> Result<()> {
    info!("Building BinBuffers for spectral sweep ({NUM_BINS} bins)...");
    let bin_buffers = BinBuffers::new(accum_spd, width as usize, height as usize);

    info!("Encoding spectral sweep video...");
    let total_frames = constants::CYCLE_TOTAL_FRAMES;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let options = if fast_encode {
        VideoEncodingOptions::fast_encode()
    } else {
        VideoEncodingOptions::default()
    };

    create_video_from_frames_singlepass(
        width,
        height,
        fps,
        |out| {
            let mut frame_buf: Vec<[u16; 3]> = Vec::new();

            for frame in 0..total_frames {
                let bin_f = f64::from(frame) * NUM_BINS as f64 / f64::from(total_frames);
                lerp_bins_frame(&bin_buffers, bin_f, &mut frame_buf);

                let bytes: &[u8] = bytemuck::cast_slice(&frame_buf);
                out.write_all(bytes).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            }
            Ok(())
        },
        output_path,
        &options,
    )?;

    info!("   Spectral sweep video complete => {output_path}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_W: usize = 4;
    const TEST_H: usize = 4;

    fn make_test_spd() -> Vec<[f64; NUM_BINS]> {
        let pixel_count = TEST_W * TEST_H;
        let mut spd = vec![[0.0f64; NUM_BINS]; pixel_count];
        for (i, pixel) in spd.iter_mut().enumerate() {
            let bin = i % NUM_BINS;
            pixel[bin] = 1.0 + i as f64 * 0.1;
        }
        spd
    }

    fn make_single_bin_spd(bin: usize, energy: f64) -> Vec<[f64; NUM_BINS]> {
        let pixel_count = TEST_W * TEST_H;
        let mut spd = vec![[0.0f64; NUM_BINS]; pixel_count];
        for pixel in &mut spd {
            pixel[bin] = energy;
        }
        spd
    }

    // -- BinBuffers ---------------------------------------------------------

    #[test]
    fn test_bin_buffers_correct_dimensions() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        assert_eq!(bb.buffers.len(), NUM_BINS);
        for buf in &bb.buffers {
            assert_eq!(buf.len(), TEST_W * TEST_H);
        }
        assert_eq!(bb.pixel_count(), TEST_W * TEST_H);
        assert_eq!(bb.width, TEST_W);
        assert_eq!(bb.height, TEST_H);
    }

    #[test]
    fn test_bin_buffers_all_values_in_unit_range() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        for (bin, buf) in bb.buffers.iter().enumerate() {
            for (i, pixel) in buf.iter().enumerate() {
                assert!(
                    pixel[0] >= 0.0 && pixel[0] <= 1.0,
                    "bin {bin} pixel {i} R={} out of range",
                    pixel[0]
                );
                assert!(
                    pixel[1] >= 0.0 && pixel[1] <= 1.0,
                    "bin {bin} pixel {i} G={} out of range",
                    pixel[1]
                );
                assert!(
                    pixel[2] >= 0.0 && pixel[2] <= 1.0,
                    "bin {bin} pixel {i} B={} out of range",
                    pixel[2]
                );
            }
        }
    }

    #[test]
    fn test_bin_buffers_zero_energy_produces_black() {
        let spd = vec![[0.0f64; NUM_BINS]; TEST_W * TEST_H];
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        for buf in &bb.buffers {
            for pixel in buf {
                assert_eq!(pixel[0], 0.0);
                assert_eq!(pixel[1], 0.0);
                assert_eq!(pixel[2], 0.0);
            }
        }
    }

    #[test]
    fn test_bin_buffers_uniform_energy_max_is_1() {
        let spd = make_single_bin_spd(32, 5.0);
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let buf = &bb.buffers[32];
        for pixel in buf {
            let max_channel = pixel[0].max(pixel[1]).max(pixel[2]);
            assert!(
                (max_channel - 1.0).abs() < 1e-5 || max_channel < 1.0,
                "uniform energy should normalize to ~1.0, got {max_channel}"
            );
        }
    }

    #[test]
    fn test_bin_buffers_deterministic() {
        let spd = make_test_spd();
        let bb1 = BinBuffers::new(&spd, TEST_W, TEST_H);
        let bb2 = BinBuffers::new(&spd, TEST_W, TEST_H);
        for bin in 0..NUM_BINS {
            for i in 0..TEST_W * TEST_H {
                assert_eq!(
                    bb1.buffers[bin][i][0].to_bits(),
                    bb2.buffers[bin][i][0].to_bits(),
                    "bin {bin} pixel {i} R not deterministic"
                );
                assert_eq!(
                    bb1.buffers[bin][i][1].to_bits(),
                    bb2.buffers[bin][i][1].to_bits(),
                    "bin {bin} pixel {i} G not deterministic"
                );
                assert_eq!(
                    bb1.buffers[bin][i][2].to_bits(),
                    bb2.buffers[bin][i][2].to_bits(),
                    "bin {bin} pixel {i} B not deterministic"
                );
            }
        }
    }

    #[test]
    fn test_bin_buffers_non_square_image() {
        let w = 6;
        let h = 3;
        let spd = vec![[0.5f64; NUM_BINS]; w * h];
        let bb = BinBuffers::new(&spd, w, h);
        assert_eq!(bb.width, w);
        assert_eq!(bb.height, h);
        assert_eq!(bb.pixel_count(), w * h);
        for buf in &bb.buffers {
            assert_eq!(buf.len(), w * h);
        }
    }

    #[test]
    fn test_bin_buffers_normalization_respects_max_pixel() {
        let mut spd = vec![[0.0f64; NUM_BINS]; TEST_W * TEST_H];
        spd[0][10] = 10.0;
        spd[1][10] = 5.0;
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let p0_max = bb.buffers[10][0][0].max(bb.buffers[10][0][1]).max(bb.buffers[10][0][2]);
        let p1_max = bb.buffers[10][1][0].max(bb.buffers[10][1][1]).max(bb.buffers[10][1][2]);
        assert!(
            p0_max > p1_max,
            "pixel with more energy should be brighter after normalization: {p0_max} vs {p1_max}"
        );
    }

    #[test]
    fn test_bin_buffers_different_bins_get_different_tints() {
        let mut spd = vec![[0.0f64; NUM_BINS]; TEST_W * TEST_H];
        for pixel in &mut spd {
            pixel[2] = 1.0;
            pixel[NUM_BINS - 3] = 1.0;
        }
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let blue_pixel = bb.buffers[2][0];
        let red_pixel = bb.buffers[NUM_BINS - 3][0];
        assert!(
            blue_pixel[2] > blue_pixel[0],
            "bin 2 (~violet/blue) should be blue-dominant: {blue_pixel:?}"
        );
        assert!(
            red_pixel[0] > red_pixel[2],
            "bin {} (~red) should be red-dominant: {:?}",
            NUM_BINS - 3,
            red_pixel
        );
    }

    // -- lerp_bins_frame ----------------------------------------------------

    #[test]
    fn test_lerp_bins_at_integer_index() {
        let spd = make_single_bin_spd(10, 2.0);
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let mut output: Vec<[u16; 3]> = Vec::new();
        lerp_bins_frame(&bb, 10.0, &mut output);
        assert_eq!(output.len(), TEST_W * TEST_H);
    }

    #[test]
    fn test_lerp_bins_output_length() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let mut output: Vec<[u16; 3]> = Vec::new();
        lerp_bins_frame(&bb, 5.5, &mut output);
        assert_eq!(output.len(), TEST_W * TEST_H);
    }

    #[test]
    fn test_lerp_bins_wraps_negative() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let mut out_neg: Vec<[u16; 3]> = Vec::new();
        let mut out_pos: Vec<[u16; 3]> = Vec::new();
        lerp_bins_frame(&bb, -1.0, &mut out_neg);
        lerp_bins_frame(&bb, (NUM_BINS - 1) as f64, &mut out_pos);
        assert_eq!(out_neg, out_pos, "wrapping should map -1 to NUM_BINS-1");
    }

    #[test]
    fn test_lerp_bins_wraps_over_max() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let mut out_a: Vec<[u16; 3]> = Vec::new();
        let mut out_b: Vec<[u16; 3]> = Vec::new();
        lerp_bins_frame(&bb, 0.0, &mut out_a);
        lerp_bins_frame(&bb, NUM_BINS as f64, &mut out_b);
        assert_eq!(out_a, out_b, "wrapping should map NUM_BINS to 0");
    }

    #[test]
    fn test_lerp_bins_output_all_valid_u16() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let mut output: Vec<[u16; 3]> = Vec::new();
        for bin_f_int in 0..NUM_BINS {
            let bin_f = bin_f_int as f64 + 0.3;
            lerp_bins_frame(&bb, bin_f, &mut output);
            assert!(!output.is_empty(), "output should be non-empty at bin_f={bin_f}");
        }
    }

    #[test]
    fn test_lerp_bins_midpoint_is_average() {
        let mut spd = vec![[0.0f64; NUM_BINS]; TEST_W * TEST_H];
        for pixel in &mut spd {
            pixel[5] = 2.0;
            pixel[6] = 2.0;
        }
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);

        let mut out_lo: Vec<[u16; 3]> = Vec::new();
        let mut out_hi: Vec<[u16; 3]> = Vec::new();
        let mut out_mid: Vec<[u16; 3]> = Vec::new();
        lerp_bins_frame(&bb, 5.0, &mut out_lo);
        lerp_bins_frame(&bb, 6.0, &mut out_hi);
        lerp_bins_frame(&bb, 5.5, &mut out_mid);

        for i in 0..out_lo.len() {
            for c in 0..3 {
                let expected =
                    f64::midpoint(f64::from(out_lo[i][c]), f64::from(out_hi[i][c])).round() as u16;
                let diff = (i32::from(out_mid[i][c]) - i32::from(expected)).unsigned_abs();
                assert!(
                    diff <= 1,
                    "lerp at 0.5 should be average: pixel {i} ch {c}: mid={}, expected={expected}",
                    out_mid[i][c]
                );
            }
        }
    }

    #[test]
    fn test_lerp_bins_deterministic_under_parallelism() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let mut out1: Vec<[u16; 3]> = Vec::new();
        let mut out2: Vec<[u16; 3]> = Vec::new();
        for _ in 0..5 {
            lerp_bins_frame(&bb, 17.3, &mut out1);
            lerp_bins_frame(&bb, 17.3, &mut out2);
            assert_eq!(out1, out2, "par_iter_mut lerp should be deterministic");
        }
    }

    // -- Sweep math ---------------------------------------------------------

    #[test]
    fn test_sweep_bin_f_monotonic() {
        let total = constants::CYCLE_TOTAL_FRAMES;
        let mut prev = -1.0f64;
        for frame in 0..total {
            let val = f64::from(frame) * NUM_BINS as f64 / f64::from(total);
            assert!(val >= prev, "sweep should be monotonically non-decreasing");
            prev = val;
        }
    }

    #[test]
    fn test_sweep_bin_f_range() {
        let total = constants::CYCLE_TOTAL_FRAMES;
        let first = 0.0f64 * NUM_BINS as f64 / f64::from(total);
        let last = f64::from(total - 1) * NUM_BINS as f64 / f64::from(total);
        assert!(first < 1.0, "first frame should be near bin 0");
        assert!(last > (NUM_BINS - 2) as f64, "last frame should be near bin {}", NUM_BINS - 1);
    }

    // -- Gallery I/O --------------------------------------------------------

    #[test]
    fn test_gallery_writes_expected_files() {
        let tmp = tempfile::tempdir().expect("failed to create temp directory");
        let dir = tmp.path().to_str().expect("temp directory path must be valid UTF-8");

        let spd = make_single_bin_spd(20, 1.0);
        generate_spectral_gallery(&spd, TEST_W as u32, TEST_H as u32, dir)
            .expect("gallery should succeed");

        for bin in 0..NUM_BINS {
            let wl = wavelength_nm_for_bin(bin);
            let file = format!("{dir}/{bin:02}_{wl:.0}nm.png");
            assert!(std::path::Path::new(&file).exists(), "missing bin image: {file}");
        }

        let file_count = std::fs::read_dir(dir).expect("failed to read gallery directory").count();
        assert_eq!(file_count, NUM_BINS, "expected exactly {NUM_BINS} PNG files");
    }

    #[test]
    fn test_gallery_zero_energy_does_not_crash() {
        let tmp = tempfile::tempdir().expect("failed to create temp directory");
        let dir = tmp.path().to_str().expect("temp directory path must be valid UTF-8");

        let spd = vec![[0.0f64; NUM_BINS]; TEST_W * TEST_H];
        let result = generate_spectral_gallery(&spd, TEST_W as u32, TEST_H as u32, dir);
        assert!(result.is_ok(), "gallery with zero energy should succeed: {result:?}");
    }

    #[test]
    fn test_gallery_extreme_energy_does_not_crash() {
        let tmp = tempfile::tempdir().expect("failed to create temp directory");
        let dir = tmp.path().to_str().expect("temp directory path must be valid UTF-8");

        let spd = vec![[1e6; NUM_BINS]; TEST_W * TEST_H];
        let result = generate_spectral_gallery(&spd, TEST_W as u32, TEST_H as u32, dir);
        assert!(result.is_ok(), "gallery with extreme energy should succeed: {result:?}");
    }

    #[test]
    fn test_gallery_non_square_image() {
        let tmp = tempfile::tempdir().expect("failed to create temp directory");
        let dir = tmp.path().to_str().expect("temp directory path must be valid UTF-8");

        let w = 6u32;
        let h = 3u32;
        let spd = vec![[0.5f64; NUM_BINS]; (w * h) as usize];
        let result = generate_spectral_gallery(&spd, w, h, dir);
        assert!(result.is_ok(), "gallery with non-square image should succeed: {result:?}");
    }

    #[test]
    fn test_gallery_single_pixel() {
        let tmp = tempfile::tempdir().expect("failed to create temp directory");
        let dir = tmp.path().to_str().expect("temp directory path must be valid UTF-8");

        let spd = vec![[1.0f64; NUM_BINS]; 1];
        let result = generate_spectral_gallery(&spd, 1, 1, dir);
        assert!(result.is_ok(), "gallery with 1x1 image should succeed: {result:?}");
    }

    // -- FramePixel ---------------------------------------------------------

    #[test]
    fn test_frame_pixel_clamps_above_one() {
        let px = <[u16; 3]>::from_rgb(1.5, 2.0, 999.0);
        assert_eq!(px, [65535, 65535, 65535]);
    }

    #[test]
    fn test_frame_pixel_clamps_below_zero() {
        let px = <[u16; 3]>::from_rgb(-0.5, -1.0, -0.001);
        assert_eq!(px, [0, 0, 0]);
    }

    #[test]
    fn test_frame_pixel_midpoint() {
        let px = <[u16; 3]>::from_rgb(0.5, 0.5, 0.5);
        for ch in px {
            let expected = (0.5f32 * 65535.0).round() as u16;
            assert_eq!(ch, expected);
        }
    }

    // -- BinBuffers Debug ---------------------------------------------------

    #[test]
    fn test_bin_buffers_debug_format() {
        let spd = make_single_bin_spd(0, 1.0);
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let dbg = format!("{bb:?}");
        assert!(dbg.contains("BinBuffers"));
        assert!(dbg.contains(&TEST_W.to_string()));
        assert!(dbg.contains(&TEST_H.to_string()));
        assert!(dbg.contains(&NUM_BINS.to_string()));
    }

    // -- Gallery determinism ------------------------------------------------

    #[test]
    fn test_gallery_deterministic() {
        let tmp1 = tempfile::tempdir().expect("tmpdir1");
        let tmp2 = tempfile::tempdir().expect("tmpdir2");
        let dir1 = tmp1.path().to_str().expect("tmpdir1 path should be valid utf-8");
        let dir2 = tmp2.path().to_str().expect("tmpdir2 path should be valid utf-8");

        let spd = make_single_bin_spd(30, 2.5);
        generate_spectral_gallery(&spd, TEST_W as u32, TEST_H as u32, dir1)
            .expect("gallery generation for dir1 should succeed");
        generate_spectral_gallery(&spd, TEST_W as u32, TEST_H as u32, dir2)
            .expect("gallery generation for dir2 should succeed");

        for bin in 0..NUM_BINS {
            let wl = wavelength_nm_for_bin(bin);
            let f1 = format!("{dir1}/{bin:02}_{wl:.0}nm.png");
            let f2 = format!("{dir2}/{bin:02}_{wl:.0}nm.png");
            let bytes1 = std::fs::read(&f1).expect("failed to read gallery file from dir1");
            let bytes2 = std::fs::read(&f2).expect("failed to read gallery file from dir2");
            assert_eq!(bytes1, bytes2, "bin {bin} PNG should be identical between two runs");
        }
    }

    // -- Gallery PNG validity -----------------------------------------------

    #[test]
    fn test_gallery_pngs_are_valid_16bit_images() {
        let tmp = tempfile::tempdir().expect("failed to create temp directory");
        let dir = tmp.path().to_str().expect("temp path should be valid utf-8");

        let spd = make_single_bin_spd(20, 1.0);
        generate_spectral_gallery(&spd, TEST_W as u32, TEST_H as u32, dir)
            .expect("gallery generation should succeed");

        let wl = wavelength_nm_for_bin(20);
        let path = format!("{dir}/20_{wl:.0}nm.png");
        let img = image::open(&path).expect("should open as valid image");
        assert_eq!(img.width(), TEST_W as u32);
        assert_eq!(img.height(), TEST_H as u32);
        let rgb16 = img.into_rgb16();
        assert_eq!(rgb16.as_raw().len(), TEST_W * TEST_H * 3);
    }

    // -- Gallery file contents differ per bin -------------------------------

    #[test]
    fn test_gallery_distinct_bins_produce_distinct_files() {
        let tmp = tempfile::tempdir().expect("failed to create temp directory");
        let dir = tmp.path().to_str().expect("temp path should be valid utf-8");

        let mut spd = vec![[0.0f64; NUM_BINS]; TEST_W * TEST_H];
        for pixel in &mut spd {
            pixel[5] = 1.0;
            pixel[50] = 1.0;
        }
        generate_spectral_gallery(&spd, TEST_W as u32, TEST_H as u32, dir)
            .expect("gallery generation should succeed");

        let wl5 = wavelength_nm_for_bin(5);
        let wl50 = wavelength_nm_for_bin(50);
        let bytes5 =
            std::fs::read(format!("{dir}/05_{wl5:.0}nm.png")).expect("failed to read bin 5 png");
        let bytes50 =
            std::fs::read(format!("{dir}/50_{wl50:.0}nm.png")).expect("failed to read bin 50 png");
        assert_ne!(
            bytes5, bytes50,
            "bins with different wavelengths should produce different PNGs"
        );
    }

    // -- Sweep: all frames finite -------------------------------------------

    #[test]
    fn test_sweep_all_frames_produce_finite_bin_f() {
        let total = constants::CYCLE_TOTAL_FRAMES;
        for frame in 0..total {
            let bin_f = f64::from(frame) * NUM_BINS as f64 / f64::from(total);
            assert!(bin_f.is_finite(), "bin_f should be finite at frame {frame}");
            assert!(bin_f >= 0.0, "bin_f should be non-negative at frame {frame}");
        }
    }

    // -- Sweep: covers all bins ---------------------------------------------

    #[test]
    fn test_sweep_visits_all_bins() {
        let total = constants::CYCLE_TOTAL_FRAMES;
        let mut visited = [false; NUM_BINS];
        for frame in 0..total {
            let bin_f = f64::from(frame) * NUM_BINS as f64 / f64::from(total);
            let nearest = bin_f.round() as usize;
            if nearest < NUM_BINS {
                visited[nearest] = true;
            }
        }
        for (bin, &was_visited) in visited.iter().enumerate() {
            assert!(was_visited, "bin {bin} should be visited during sweep");
        }
    }

    // -- lerp at exact bin matches the bin buffer ---------------------------

    #[test]
    fn test_lerp_at_exact_bin_matches_buffer() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);

        for test_bin in [0, 10, 32, NUM_BINS - 1] {
            let mut output: Vec<[u16; 3]> = Vec::new();
            lerp_bins_frame(&bb, test_bin as f64, &mut output);

            for (i, out_pixel) in output.iter().enumerate() {
                let expected = <[u16; 3]>::from_rgb(
                    bb.buffers[test_bin][i][0],
                    bb.buffers[test_bin][i][1],
                    bb.buffers[test_bin][i][2],
                );
                assert_eq!(
                    *out_pixel, expected,
                    "lerp at exact bin {test_bin} pixel {i} should match buffer"
                );
            }
        }
    }

    // -- Constants ----------------------------------------------------------

    #[test]
    fn test_cycle_constants_are_consistent() {
        let expected_frames = (constants::CYCLE_DURATION_SECONDS
            * f64::from(constants::DEFAULT_VIDEO_FPS))
        .round() as u32;
        assert_eq!(
            constants::CYCLE_TOTAL_FRAMES,
            expected_frames,
            "CYCLE_TOTAL_FRAMES should equal CYCLE_DURATION_SECONDS * FPS"
        );
    }

    #[test]
    fn test_display_gamma_is_srgb() {
        assert!(
            (constants::DISPLAY_GAMMA - 2.2).abs() < 0.01,
            "display gamma should be 2.2 for sRGB"
        );
    }

    #[test]
    fn test_cycle_total_frames_positive() {
        const { assert!(constants::CYCLE_TOTAL_FRAMES > 0) };
    }

    #[test]
    fn test_cycle_duration_positive() {
        const { assert!(constants::CYCLE_DURATION_SECONDS > 0.0) };
    }
}
