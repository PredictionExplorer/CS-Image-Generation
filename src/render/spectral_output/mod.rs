//! Spectral gallery and sweep video generation.
//!
//! After the main rendering pass accumulates the full SPD buffer, this module
//! produces per-bin wavelength images (the "spectral gallery") and a single
//! spectral sweep video that cycles through all 64 bins from violet to red.

#[cfg(test)]
use super::constants;
#[cfg(test)]
use super::context::PixelBuffer;
#[cfg(test)]
use crate::spectrum::NUM_BINS;
#[cfg(test)]
use crate::spectrum::wavelength_nm_for_bin;
#[cfg(test)]
use rayon::prelude::*;

mod bin_buffers;
mod gallery;
mod sweep;

pub use bin_buffers::BinBuffers;
pub use gallery::generate_spectral_gallery;
pub use sweep::generate_spectral_sweep_video;
pub(crate) use sweep::generate_spectral_sweep_video_with_encoder;
#[cfg(test)]
use sweep::{BlendWeights, gaussian_blend_to_pixelbuffer, quantize_to_u16_rgb};

#[cfg(test)]
mod tests {
    use super::*;

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

    fn gaussian_blend_frame(
        bin_buffers: &BinBuffers,
        center: f64,
        sigma: f64,
        output: &mut Vec<[u16; 3]>,
    ) {
        let pixel_count = bin_buffers.pixel_count();
        output.resize(pixel_count, [0u16; 3]);
        let bw = BlendWeights::compute(center, sigma);

        let bufs = &bin_buffers.buffers;
        output.par_iter_mut().enumerate().for_each(|(i, pixel)| {
            let (mut r, mut g, mut b_ch) = (0.0f32, 0.0f32, 0.0f32);
            for (j, bin) in (bw.lo..=bw.hi).enumerate() {
                let w = bw.weights[j];
                r += bufs[bin][i][0] * w;
                g += bufs[bin][i][1] * w;
                b_ch += bufs[bin][i][2] * w;
            }
            *pixel = <[u16; 3]>::from_rgb(r, g, b_ch);
        });
    }

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
        let bb = BinBuffers::try_new(&spd, TEST_W as u32, TEST_H as u32)
            .expect("valid image shape should build bin buffers");
        assert_eq!(bb.buffers.len(), NUM_BINS);
        for buf in &bb.buffers {
            assert_eq!(buf.len(), TEST_W * TEST_H);
        }
        assert_eq!(bb.pixel_count(), TEST_W * TEST_H);
        assert_eq!(bb.width, TEST_W);
        assert_eq!(bb.height, TEST_H);
    }

    #[test]
    fn test_bin_buffers_try_new_rejects_length_mismatch() {
        let mut spd = make_test_spd();
        spd.pop();

        let err = BinBuffers::try_new(&spd, TEST_W as u32, TEST_H as u32)
            .expect_err("mismatched SPD length should fail validation");

        assert!(matches!(err, crate::render::error::RenderError::InvalidScene { .. }));
        assert!(err.to_string().contains("accumulated SPD length"));
    }

    #[test]
    fn test_bin_buffers_try_new_rejects_zero_dimensions() {
        let err = BinBuffers::try_new(&[], 0, TEST_H as u32)
            .expect_err("zero width should fail validation");

        assert!(matches!(
            err,
            crate::render::error::RenderError::InvalidDimensions { width: 0, height }
                if height == TEST_H as u32
        ));
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

    // -- gaussian_blend_frame -----------------------------------------------

    #[test]
    fn test_gaussian_blend_at_integer_index() {
        let spd = make_single_bin_spd(10, 2.0);
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let mut output: Vec<[u16; 3]> = Vec::new();
        gaussian_blend_frame(&bb, 10.0, constants::SWEEP_GAUSSIAN_SIGMA, &mut output);
        assert_eq!(output.len(), TEST_W * TEST_H);
    }

    #[test]
    fn test_gaussian_blend_output_length() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let mut output: Vec<[u16; 3]> = Vec::new();
        gaussian_blend_frame(&bb, 5.5, constants::SWEEP_GAUSSIAN_SIGMA, &mut output);
        assert_eq!(output.len(), TEST_W * TEST_H);
    }

    #[test]
    fn test_gaussian_blend_clamps_low_center() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let mut output: Vec<[u16; 3]> = Vec::new();
        gaussian_blend_frame(&bb, 0.0, constants::SWEEP_GAUSSIAN_SIGMA, &mut output);
        assert_eq!(output.len(), TEST_W * TEST_H);
    }

    #[test]
    fn test_gaussian_blend_clamps_high_center() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let mut output: Vec<[u16; 3]> = Vec::new();
        gaussian_blend_frame(
            &bb,
            (NUM_BINS - 1) as f64,
            constants::SWEEP_GAUSSIAN_SIGMA,
            &mut output,
        );
        assert_eq!(output.len(), TEST_W * TEST_H);
    }

    #[test]
    fn test_gaussian_blend_output_all_valid_u16() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let mut output: Vec<[u16; 3]> = Vec::new();
        for bin_f_int in 0..NUM_BINS {
            let bin_f = bin_f_int as f64 + 0.3;
            gaussian_blend_frame(&bb, bin_f, constants::SWEEP_GAUSSIAN_SIGMA, &mut output);
            assert!(!output.is_empty(), "output should be non-empty at bin_f={bin_f}");
        }
    }

    #[test]
    fn test_gaussian_blend_center_bin_dominates() {
        let mut spd = vec![[0.0f64; NUM_BINS]; TEST_W * TEST_H];
        for pixel in &mut spd {
            pixel[30] = 10.0;
            pixel[31] = 1.0;
            pixel[29] = 1.0;
        }
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);

        let mut out_center: Vec<[u16; 3]> = Vec::new();
        let mut out_off: Vec<[u16; 3]> = Vec::new();
        gaussian_blend_frame(&bb, 30.0, constants::SWEEP_GAUSSIAN_SIGMA, &mut out_center);
        gaussian_blend_frame(&bb, 35.0, constants::SWEEP_GAUSSIAN_SIGMA, &mut out_off);

        let center_brightness: u64 =
            out_center.iter().map(|p| p.iter().map(|&c| u64::from(c)).sum::<u64>()).sum();
        let off_brightness: u64 =
            out_off.iter().map(|p| p.iter().map(|&c| u64::from(c)).sum::<u64>()).sum();
        assert!(
            center_brightness > off_brightness,
            "frame centred on the bright bin should be brighter"
        );
    }

    #[test]
    fn test_gaussian_blend_deterministic_under_parallelism() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let mut out1: Vec<[u16; 3]> = Vec::new();
        let mut out2: Vec<[u16; 3]> = Vec::new();
        for _ in 0..5 {
            gaussian_blend_frame(&bb, 17.3, constants::SWEEP_GAUSSIAN_SIGMA, &mut out1);
            gaussian_blend_frame(&bb, 17.3, constants::SWEEP_GAUSSIAN_SIGMA, &mut out2);
            assert_eq!(out1, out2, "par_iter_mut gaussian blend should be deterministic");
        }
    }

    #[test]
    fn test_gaussian_blend_narrow_sigma_approximates_single_bin() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let mut out_narrow: Vec<[u16; 3]> = Vec::new();
        gaussian_blend_frame(&bb, 20.0, 0.01, &mut out_narrow);

        for (i, pixel) in out_narrow.iter().enumerate() {
            let expected = <[u16; 3]>::from_rgb(
                bb.buffers[20][i][0],
                bb.buffers[20][i][1],
                bb.buffers[20][i][2],
            );
            for c in 0..3 {
                let diff = (i32::from(pixel[c]) - i32::from(expected[c])).unsigned_abs();
                assert!(
                    diff <= 1,
                    "very narrow sigma should approximate single bin: pixel {i} ch {c}"
                );
            }
        }
    }

    // -- Sweep math ---------------------------------------------------------

    fn sweep_bin_f(frame: u32) -> f64 {
        let total = constants::CYCLE_TOTAL_FRAMES;
        let start = constants::SWEEP_BIN_START as f64;
        let end = constants::SWEEP_BIN_END as f64;
        let t = f64::from(frame) / f64::from(total - 1);
        start + t * (end - start)
    }

    #[test]
    fn test_sweep_bin_f_monotonic() {
        let total = constants::CYCLE_TOTAL_FRAMES;
        let mut prev = -1.0f64;
        for frame in 0..total {
            let val = sweep_bin_f(frame);
            assert!(val >= prev, "sweep should be monotonically non-decreasing");
            prev = val;
        }
    }

    #[test]
    fn test_sweep_bin_f_range() {
        let first = sweep_bin_f(0);
        let last = sweep_bin_f(constants::CYCLE_TOTAL_FRAMES - 1);
        let start = constants::SWEEP_BIN_START as f64;
        let end = constants::SWEEP_BIN_END as f64;
        assert!(
            (first - start).abs() < 1e-10,
            "first frame should be at SWEEP_BIN_START ({start}), got {first}"
        );
        assert!(
            (last - end).abs() < 1e-10,
            "last frame should be at SWEEP_BIN_END ({end}), got {last}"
        );
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
    fn test_gallery_creates_missing_output_dir() {
        let tmp = tempfile::tempdir().expect("failed to create temp directory");
        let dir = tmp.path().join("nested").join("gallery");

        let spd = make_single_bin_spd(20, 1.0);
        generate_spectral_gallery(&spd, TEST_W as u32, TEST_H as u32, &dir)
            .expect("gallery should create missing output directory");

        assert!(dir.is_dir());
        assert_eq!(
            std::fs::read_dir(&dir).expect("gallery directory should be readable").count(),
            NUM_BINS
        );
    }

    #[test]
    fn test_gallery_rejects_spd_shape_before_creating_output_dir() {
        let tmp = tempfile::tempdir().expect("failed to create temp directory");
        let dir = tmp.path().join("nested").join("gallery");

        let mut spd = make_single_bin_spd(20, 1.0);
        spd.pop();
        let err = generate_spectral_gallery(&spd, TEST_W as u32, TEST_H as u32, &dir)
            .expect_err("mismatched SPD length should fail");

        assert!(matches!(err, crate::render::error::RenderError::InvalidScene { .. }));
        assert!(!dir.exists(), "invalid input should not create output directories");
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
            let bin_f = sweep_bin_f(frame);
            assert!(bin_f.is_finite(), "bin_f should be finite at frame {frame}");
            assert!(bin_f >= 0.0, "bin_f should be non-negative at frame {frame}");
        }
    }

    // -- Sweep: covers all bins in the narrowed range -----------------------

    #[test]
    fn test_sweep_visits_all_bins_in_range() {
        let total = constants::CYCLE_TOTAL_FRAMES;
        let start = constants::SWEEP_BIN_START;
        let end = constants::SWEEP_BIN_END;
        let mut visited = [false; NUM_BINS];
        for frame in 0..total {
            let bin_f = sweep_bin_f(frame);
            let nearest = bin_f.round() as usize;
            if nearest < NUM_BINS {
                visited[nearest] = true;
            }
        }
        for (bin, &was_visited) in visited.iter().enumerate().take(end + 1).skip(start) {
            assert!(was_visited, "bin {bin} in sweep range should be visited");
        }
    }

    // -- gaussian blend at exact bin with tiny sigma matches buffer ---------

    #[test]
    fn test_gaussian_blend_tiny_sigma_matches_buffer() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);

        for test_bin in [0, 10, 32, NUM_BINS - 1] {
            let mut output: Vec<[u16; 3]> = Vec::new();
            gaussian_blend_frame(&bb, test_bin as f64, 0.01, &mut output);

            for (i, out_pixel) in output.iter().enumerate() {
                let expected = <[u16; 3]>::from_rgb(
                    bb.buffers[test_bin][i][0],
                    bb.buffers[test_bin][i][1],
                    bb.buffers[test_bin][i][2],
                );
                for c in 0..3 {
                    let diff = (i32::from(out_pixel[c]) - i32::from(expected[c])).unsigned_abs();
                    assert!(
                        diff <= 1,
                        "tiny-sigma gaussian at exact bin {test_bin} pixel {i} ch {c} should match buffer"
                    );
                }
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

    #[test]
    fn test_sweep_bin_range_valid() {
        const { assert!(constants::SWEEP_BIN_START < constants::SWEEP_BIN_END) };
        const { assert!(constants::SWEEP_BIN_END < NUM_BINS) };
    }

    #[test]
    fn test_sweep_gaussian_sigma_positive() {
        const { assert!(constants::SWEEP_GAUSSIAN_SIGMA > 0.0) };
    }

    #[test]
    fn test_sweep_bloom_constants_positive() {
        const { assert!(constants::SWEEP_BLOOM_RADIUS > 0) };
        const { assert!(constants::SWEEP_BLOOM_STRENGTH > 0.0) };
        const { assert!(constants::SWEEP_BLOOM_CORE_BRIGHTNESS > 0.0) };
    }

    #[test]
    fn test_sweep_vignette_constants_positive() {
        const { assert!(constants::SWEEP_VIGNETTE_STRENGTH > 0.0) };
        const { assert!(constants::SWEEP_VIGNETTE_SOFTNESS > 1.0) };
    }

    #[test]
    fn test_sweep_vibrance_above_one() {
        const { assert!(constants::SWEEP_VIBRANCE >= 1.0) };
    }

    // -- BlendWeights -------------------------------------------------------

    #[test]
    fn test_blend_weights_sum_to_one() {
        let bw = BlendWeights::compute(30.0, 2.5);
        let sum: f32 = bw.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "weights should sum to ~1.0, got {sum}");
    }

    #[test]
    fn test_blend_weights_symmetric_at_integer() {
        let bw = BlendWeights::compute(30.0, 2.5);
        let center_idx = 30 - bw.lo;
        if center_idx > 0 && center_idx < bw.weights.len() - 1 {
            let diff = (bw.weights[center_idx - 1] - bw.weights[center_idx + 1]).abs();
            assert!(diff < 1e-6, "weights should be symmetric about the integer centre");
        }
    }

    #[test]
    fn test_blend_weights_peak_at_center() {
        let bw = BlendWeights::compute(30.0, 2.5);
        let center_idx = 30 - bw.lo;
        for (j, &w) in bw.weights.iter().enumerate() {
            if j != center_idx {
                assert!(w <= bw.weights[center_idx], "centre weight should be the maximum");
            }
        }
    }

    // -- BinBuffers::composite ----------------------------------------------

    #[test]
    fn test_composite_output_length() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let comp = bb.composite();
        assert_eq!(comp.len(), TEST_W * TEST_H);
    }

    #[test]
    fn test_composite_nonzero_for_nonzero_input() {
        let spd = make_single_bin_spd(20, 1.0);
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let comp = bb.composite();
        let any_nonzero = comp.iter().any(|&(r, g, b, _)| r > 0.0 || g > 0.0 || b > 0.0);
        assert!(any_nonzero, "composite should have non-zero pixels");
    }

    #[test]
    fn test_composite_alpha_is_one() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let comp = bb.composite();
        for &(_, _, _, a) in &comp {
            assert!((a - 1.0).abs() < 1e-10, "alpha should be 1.0");
        }
    }

    #[test]
    fn test_composite_zero_energy_produces_black() {
        let spd = vec![[0.0f64; NUM_BINS]; TEST_W * TEST_H];
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let comp = bb.composite();
        for &(r, g, b, _) in &comp {
            assert_eq!(r, 0.0);
            assert_eq!(g, 0.0);
            assert_eq!(b, 0.0);
        }
    }

    // -- active_bin_range ---------------------------------------------------

    #[test]
    fn test_active_bin_range_single_bin() {
        let spd = make_single_bin_spd(30, 1.0);
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let (lo, hi) = bb.active_bin_range();
        assert!(lo <= 30, "lo should be at or before the active bin");
        assert!(hi >= 30, "hi should be at or after the active bin");
        assert!(lo >= 28, "padding should not extend more than 2 bins below");
        assert!(hi <= 32, "padding should not extend more than 2 bins above");
    }

    #[test]
    fn test_active_bin_range_zero_energy_returns_fallback() {
        let spd = vec![[0.0f64; NUM_BINS]; TEST_W * TEST_H];
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let (lo, hi) = bb.active_bin_range();
        assert_eq!(lo, constants::SWEEP_BIN_START);
        assert_eq!(hi, constants::SWEEP_BIN_END);
    }

    #[test]
    fn test_active_bin_range_full_spectrum() {
        let spd = vec![[1.0f64; NUM_BINS]; TEST_W * TEST_H];
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let (lo, hi) = bb.active_bin_range();
        assert!(lo <= 2, "all bins active should start near 0");
        assert!(hi >= NUM_BINS - 3, "all bins active should end near last bin");
    }

    #[test]
    fn test_active_bin_range_edge_bins() {
        let spd = make_single_bin_spd(0, 1.0);
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let (lo, _) = bb.active_bin_range();
        assert_eq!(lo, 0, "should clamp to 0, not underflow");

        let spd = make_single_bin_spd(NUM_BINS - 1, 1.0);
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let (_, hi) = bb.active_bin_range();
        assert_eq!(hi, NUM_BINS - 1, "should clamp to last bin");
    }

    // -- quantize_to_u16_rgb ------------------------------------------------

    #[test]
    fn test_quantize_midpoint() {
        let pixels: PixelBuffer = vec![(0.5, 0.5, 0.5, 1.0)];
        let q = quantize_to_u16_rgb(&pixels);
        assert_eq!(q.len(), 3);
        let expected = (0.5 * 65535.0f64).round() as u16;
        assert_eq!(q[0], expected);
        assert_eq!(q[1], expected);
        assert_eq!(q[2], expected);
    }

    #[test]
    fn test_quantize_clamps() {
        let pixels: PixelBuffer = vec![(-0.5, 1.5, 0.0, 1.0)];
        let q = quantize_to_u16_rgb(&pixels);
        assert_eq!(q[0], 0);
        assert_eq!(q[1], 65535);
        assert_eq!(q[2], 0);
    }

    #[test]
    fn test_quantize_output_length() {
        let pixels: PixelBuffer = vec![(0.0, 0.0, 0.0, 1.0); 16];
        let q = quantize_to_u16_rgb(&pixels);
        assert_eq!(q.len(), 48);
    }

    // -- gaussian_blend_to_pixelbuffer --------------------------------------

    #[test]
    fn test_gaussian_blend_to_pixelbuffer_length() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let mut output: PixelBuffer = Vec::new();
        gaussian_blend_to_pixelbuffer(&bb, 30.0, constants::SWEEP_GAUSSIAN_SIGMA, &mut output);
        assert_eq!(output.len(), TEST_W * TEST_H);
    }

    #[test]
    fn test_gaussian_blend_to_pixelbuffer_alpha_is_one() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let mut output: PixelBuffer = Vec::new();
        gaussian_blend_to_pixelbuffer(&bb, 30.0, constants::SWEEP_GAUSSIAN_SIGMA, &mut output);
        for &(_, _, _, a) in &output {
            assert!((a - 1.0).abs() < 1e-10, "alpha should be 1.0");
        }
    }

    #[test]
    fn test_gaussian_blend_to_pixelbuffer_matches_u16_variant() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);

        let mut u16_out: Vec<[u16; 3]> = Vec::new();
        gaussian_blend_frame(&bb, 20.0, constants::SWEEP_GAUSSIAN_SIGMA, &mut u16_out);

        let mut pb_out: PixelBuffer = Vec::new();
        gaussian_blend_to_pixelbuffer(&bb, 20.0, constants::SWEEP_GAUSSIAN_SIGMA, &mut pb_out);

        for (i, (&u16_px, &(r, g, b, _))) in u16_out.iter().zip(pb_out.iter()).enumerate() {
            let expected_r = (r.clamp(0.0, 1.0) * 65535.0).round() as u16;
            let expected_g = (g.clamp(0.0, 1.0) * 65535.0).round() as u16;
            let expected_b = (b.clamp(0.0, 1.0) * 65535.0).round() as u16;
            for (c, (a, b)) in
                [(u16_px[0], expected_r), (u16_px[1], expected_g), (u16_px[2], expected_b)]
                    .iter()
                    .enumerate()
            {
                let diff = (i32::from(*a) - i32::from(*b)).unsigned_abs();
                assert!(diff <= 1, "pixel {i} ch {c}: u16={a} vs pixelbuf-quantized={b}");
            }
        }
    }

    // -- cosine easing math -------------------------------------------------

    #[test]
    fn test_cosine_easing_endpoints() {
        let t0 = 0.0f64;
        let t1 = 1.0f64;
        let eased_0 = (1.0 - (t0 * std::f64::consts::PI).cos()) * 0.5;
        let eased_1 = (1.0 - (t1 * std::f64::consts::PI).cos()) * 0.5;
        assert!(eased_0.abs() < 1e-10, "eased(0) should be 0, got {eased_0}");
        assert!((eased_1 - 1.0).abs() < 1e-10, "eased(1) should be 1, got {eased_1}");
    }

    #[test]
    fn test_cosine_easing_midpoint() {
        let t = 0.5f64;
        let eased = (1.0 - (t * std::f64::consts::PI).cos()) * 0.5;
        assert!((eased - 0.5).abs() < 1e-10, "eased(0.5) should be 0.5, got {eased}");
    }

    #[test]
    fn test_cosine_easing_monotonic() {
        let steps = 100;
        let mut prev = -1.0f64;
        for i in 0..=steps {
            let t = f64::from(i) / f64::from(steps);
            let eased = (1.0 - (t * std::f64::consts::PI).cos()) * 0.5;
            assert!(eased >= prev - 1e-15, "cosine easing should be monotonic at step {i}");
            prev = eased;
        }
    }
}
