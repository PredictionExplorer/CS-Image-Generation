//! Spectral gallery and sweep video generation.
//!
//! After the main rendering pass accumulates the full SPD buffer, this module
//! produces per-bin wavelength images (the "spectral gallery") and a single
//! spectral sweep video that cycles through all 64 bins from violet to red.

use super::constants;
use super::context::PixelBuffer;
use super::error::{RenderError, Result};
use super::video::{VideoEncodingOptions, create_video_from_frames_singlepass};
use crate::post_effects::{CinematicColorGrade, ColorGradeParams, GaussianBloom, PostEffect};
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

    /// Average all bin images into a single full-spectrum composite `PixelBuffer`.
    fn composite(&self) -> PixelBuffer {
        let pixel_count = self.pixel_count();
        let inv_bins = 1.0 / NUM_BINS as f64;
        (0..pixel_count)
            .into_par_iter()
            .map(|i| {
                let (mut r, mut g, mut b) = (0.0f64, 0.0f64, 0.0f64);
                for bin in &self.buffers {
                    r += f64::from(bin[i][0]);
                    g += f64::from(bin[i][1]);
                    b += f64::from(bin[i][2]);
                }
                (r * inv_bins, g * inv_bins, b * inv_bins, 1.0)
            })
            .collect()
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

/// Pre-computed, normalised Gaussian weights for a bin blend.
struct BlendWeights {
    lo: usize,
    hi: usize,
    weights: Vec<f32>,
}

impl BlendWeights {
    /// Compute normalised Gaussian weights centred at `center` with the given
    /// `sigma` (in bin-units).  Bins are clamped to `[0, NUM_BINS-1]`.
    fn compute(center: f64, sigma: f64) -> Self {
        let radius = (3.0 * sigma).ceil() as isize;
        let center_floor = center.floor() as isize;
        let lo = (center_floor - radius).max(0) as usize;
        let hi = ((center_floor + radius + 1) as usize).min(NUM_BINS - 1);

        let inv_2sig2 = 0.5 / (sigma * sigma);
        let mut weights: Vec<f32> = Vec::with_capacity(hi - lo + 1);
        let mut total: f64 = 0.0;
        for b in lo..=hi {
            let d = b as f64 - center;
            let w = (-d * d * inv_2sig2).exp();
            weights.push(w as f32);
            total += w;
        }
        let inv_total = 1.0 / total as f32;
        for w in &mut weights {
            *w *= inv_total;
        }

        Self { lo, hi, weights }
    }
}

/// Blend multiple bin images with a Gaussian kernel centred at `center`,
/// producing a `PixelBuffer` (f64 RGBA, alpha = 1.0) suitable for post-effects.
fn gaussian_blend_to_pixelbuffer(
    bin_buffers: &BinBuffers,
    center: f64,
    sigma: f64,
    output: &mut PixelBuffer,
) {
    let pixel_count = bin_buffers.pixel_count();
    output.resize(pixel_count, (0.0, 0.0, 0.0, 1.0));
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
        *pixel = (f64::from(r), f64::from(g), f64::from(b_ch), 1.0);
    });
}

/// Hermite smoothstep: 0 at `edge0`, 1 at `edge1`, smooth in between.
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Linearly blend two `PixelBuffer`s: returns `a*(1-t) + b*t`.
fn blend_pixelbuffers(a: &PixelBuffer, b: &PixelBuffer, t: f64) -> PixelBuffer {
    let one_minus_t = 1.0 - t;
    a.par_iter()
        .zip(b.par_iter())
        .map(|(&(ar, ag, ab, aa), &(br, bg, bb, ba))| {
            (
                ar * one_minus_t + br * t,
                ag * one_minus_t + bg * t,
                ab * one_minus_t + bb * t,
                aa * one_minus_t + ba * t,
            )
        })
        .collect()
}

/// Convert a `PixelBuffer` to packed 16-bit RGB for the ffmpeg `rgb48le` pipe.
fn quantize_to_u16_rgb(pixels: &PixelBuffer) -> Vec<u16> {
    let mut buf = vec![0u16; pixels.len() * 3];
    buf.par_chunks_mut(3).zip(pixels.par_iter()).for_each(|(chunk, &(r, g, b, _a))| {
        chunk[0] = (r.clamp(0.0, 1.0) * constants::U16_MAX_F64).round() as u16;
        chunk[1] = (g.clamp(0.0, 1.0) * constants::U16_MAX_F64).round() as u16;
        chunk[2] = (b.clamp(0.0, 1.0) * constants::U16_MAX_F64).round() as u16;
    });
    buf
}

/// Generate a single spectral sweep video (violet to red) at the given path.
///
/// The sweep applies cosine easing for smooth pacing, Gaussian bloom and
/// subtle colour grading (vignette + vibrance) per frame, and fades in from
/// / out to a full-spectrum composite at each end.
pub fn generate_spectral_sweep_video(
    accum_spd: &[[f64; NUM_BINS]],
    width: u32,
    height: u32,
    output_path: &str,
    fast_encode: bool,
) -> Result<()> {
    info!("Building BinBuffers for spectral sweep ({NUM_BINS} bins)...");
    let bin_buffers = BinBuffers::new(accum_spd, width as usize, height as usize);

    let w = width as usize;
    let h = height as usize;

    info!("Preparing sweep post-effects (bloom + colour grade)...");
    let bloom = GaussianBloom::new(
        constants::SWEEP_BLOOM_RADIUS,
        constants::SWEEP_BLOOM_STRENGTH,
        constants::SWEEP_BLOOM_CORE_BRIGHTNESS,
    );
    let color_grade = CinematicColorGrade::new(ColorGradeParams {
        strength: 1.0,
        vignette_strength: constants::SWEEP_VIGNETTE_STRENGTH,
        vignette_softness: constants::SWEEP_VIGNETTE_SOFTNESS,
        vibrance: constants::SWEEP_VIBRANCE,
        clarity_strength: 0.0,
        clarity_radius: 1,
        tone_curve: 0.0,
        shadow_tint: [0.0; 3],
        highlight_tint: [0.0; 3],
        palette_wave_strength: 0.0,
    });

    let composite = bin_buffers.composite();
    let processed_composite = apply_sweep_effects(&bloom, &color_grade, &composite, w, h)?;

    info!("Encoding spectral sweep video...");
    let total_frames = constants::CYCLE_TOTAL_FRAMES;
    let fade_frames = constants::SWEEP_FADE_FRAMES;
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
            let mut frame_buf: PixelBuffer = Vec::new();
            let start = constants::SWEEP_BIN_START as f64;
            let end = constants::SWEEP_BIN_END as f64;
            let sigma = constants::SWEEP_GAUSSIAN_SIGMA;

            for frame in 0..total_frames {
                let t_linear = f64::from(frame) / f64::from(total_frames - 1);
                let t_eased = (1.0 - (t_linear * std::f64::consts::PI).cos()) * 0.5;
                let bin_f = start + t_eased * (end - start);

                gaussian_blend_to_pixelbuffer(&bin_buffers, bin_f, sigma, &mut frame_buf);

                let processed = apply_sweep_effects(&bloom, &color_grade, &frame_buf, w, h)
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

                let final_frame = if frame < fade_frames {
                    let fade_t = smoothstep(0.0, f64::from(fade_frames), f64::from(frame));
                    blend_pixelbuffers(&processed_composite, &processed, fade_t)
                } else if frame >= total_frames - fade_frames {
                    let fade_t = smoothstep(
                        0.0,
                        f64::from(fade_frames),
                        f64::from(total_frames - 1 - frame),
                    );
                    blend_pixelbuffers(&processed_composite, &processed, fade_t)
                } else {
                    processed
                };

                let quantized = quantize_to_u16_rgb(&final_frame);
                let bytes: &[u8] = bytemuck::cast_slice(&quantized);
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

/// Run bloom then colour-grade on a single `PixelBuffer`.
fn apply_sweep_effects(
    bloom: &GaussianBloom,
    color_grade: &CinematicColorGrade,
    buf: &PixelBuffer,
    w: usize,
    h: usize,
) -> Result<PixelBuffer> {
    let bloomed =
        bloom.process(buf, w, h).map_err(|e| RenderError::InvalidConfig(e.to_string()))?;
    color_grade.process(&bloomed, w, h).map_err(|e| RenderError::InvalidConfig(e.to_string()))
}

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
    fn test_sweep_fade_frames_fits_in_total() {
        const { assert!(constants::SWEEP_FADE_FRAMES < constants::CYCLE_TOTAL_FRAMES / 2) };
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

    // -- smoothstep ---------------------------------------------------------

    #[test]
    fn test_smoothstep_boundaries() {
        assert!((smoothstep(0.0, 1.0, 0.0)).abs() < 1e-10, "should be 0 at edge0");
        assert!((smoothstep(0.0, 1.0, 1.0) - 1.0).abs() < 1e-10, "should be 1 at edge1");
    }

    #[test]
    fn test_smoothstep_midpoint() {
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-10, "should be 0.5 at midpoint");
    }

    #[test]
    fn test_smoothstep_clamps_outside_range() {
        assert!(smoothstep(0.0, 1.0, -1.0).abs() < 1e-10);
        assert!((smoothstep(0.0, 1.0, 2.0) - 1.0).abs() < 1e-10);
    }

    // -- blend_pixelbuffers -------------------------------------------------

    #[test]
    fn test_blend_pixelbuffers_identity_at_zero() {
        let a: PixelBuffer = vec![(1.0, 0.5, 0.0, 1.0); 4];
        let b: PixelBuffer = vec![(0.0, 0.5, 1.0, 1.0); 4];
        let blended = blend_pixelbuffers(&a, &b, 0.0);
        for (ap, bp) in a.iter().zip(blended.iter()) {
            assert!((ap.0 - bp.0).abs() < 1e-10);
            assert!((ap.1 - bp.1).abs() < 1e-10);
            assert!((ap.2 - bp.2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_blend_pixelbuffers_identity_at_one() {
        let a: PixelBuffer = vec![(1.0, 0.5, 0.0, 1.0); 4];
        let b: PixelBuffer = vec![(0.0, 0.5, 1.0, 1.0); 4];
        let blended = blend_pixelbuffers(&a, &b, 1.0);
        for (bp_orig, bp) in b.iter().zip(blended.iter()) {
            assert!((bp_orig.0 - bp.0).abs() < 1e-10);
            assert!((bp_orig.1 - bp.1).abs() < 1e-10);
            assert!((bp_orig.2 - bp.2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_blend_pixelbuffers_midpoint() {
        let a: PixelBuffer = vec![(1.0, 0.0, 0.0, 1.0); 4];
        let b: PixelBuffer = vec![(0.0, 1.0, 0.0, 1.0); 4];
        let blended = blend_pixelbuffers(&a, &b, 0.5);
        for &(r, g, _b, _a) in &blended {
            assert!((r - 0.5).abs() < 1e-10);
            assert!((g - 0.5).abs() < 1e-10);
        }
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
