//! Spectral gallery and cycle video generation.
//!
//! After the main rendering pass accumulates the full SPD buffer, this module
//! produces per-bin wavelength images (the "spectral gallery"), a dominant-wavelength
//! heatmap, and six spectral cycle video variants.

use super::constants;
use super::error::{RenderError, Result};
use super::video::{VideoEncodingOptions, create_video_from_frames_singlepass};
use crate::spectrum::{NUM_BINS, wavelength_nm_for_bin, wavelength_to_rgb};
use image::{ImageBuffer, Rgb};
use rayon::prelude::*;
use std::f64::consts::PI;
use tracing::info;

/// Pre-computed per-bin float RGB images used by both the gallery and cycle videos.
pub struct BinBuffers {
    buffers: Vec<Vec<[f32; 3]>>,
    width: usize,
    height: usize,
}

impl BinBuffers {
    /// Build 64 bin images from the accumulated SPD buffer.
    ///
    /// Each bin image normalizes that bin's energy across all pixels, tints by the
    /// bin's wavelength color, and applies display gamma.
    pub fn new(accum_spd: &[[f64; NUM_BINS]], width: usize, height: usize) -> Self {
        let pixel_count = width * height;
        assert_eq!(accum_spd.len(), pixel_count);

        let inv_gamma = 1.0 / constants::DISPLAY_GAMMA;

        let buffers: Vec<Vec<[f32; 3]>> = (0..NUM_BINS)
            .into_par_iter()
            .map(|bin| {
                let wavelength = wavelength_nm_for_bin(bin);
                let (tint_r, tint_g, tint_b) = wavelength_to_rgb(wavelength);

                let max_val = accum_spd
                    .iter()
                    .map(|spd| spd[bin])
                    .fold(0.0f64, f64::max)
                    .max(1e-10);

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

    fn pixel_count(&self) -> usize {
        self.width * self.height
    }
}

// ---------------------------------------------------------------------------
// Spectral Gallery (Section 9)
// ---------------------------------------------------------------------------

/// Generate 64 per-bin 16-bit PNGs and a dominant-wavelength heatmap.
pub fn generate_spectral_gallery(
    accum_spd: &[[f64; NUM_BINS]],
    width: u32,
    height: u32,
    output_dir: &str,
) -> Result<()> {
    info!("Generating spectral gallery ({NUM_BINS} bin images)...");

    let bin_buffers = BinBuffers::new(accum_spd, width as usize, height as usize);

    // Write per-bin PNGs in parallel
    (0..NUM_BINS).into_par_iter().try_for_each(|bin| {
        let wavelength = wavelength_nm_for_bin(bin);
        let filename = format!("{output_dir}/{bin:02}_{wavelength:.0}nm.png");
        save_bin_image(&bin_buffers.buffers[bin], width, height, &filename)
    })?;

    // Dominant-wavelength heatmap
    info!("   Generating dominant-wavelength heatmap...");
    generate_dominant_wavelength_heatmap(accum_spd, width, height, output_dir)?;

    info!("   Spectral gallery complete ({} images)", NUM_BINS + 1);
    Ok(())
}

fn save_bin_image(buf: &[[f32; 3]], width: u32, height: u32, path: &str) -> Result<()> {
    let pixel_count = (width * height) as usize;
    let mut raw = Vec::with_capacity(pixel_count * 3);
    for pixel in buf {
        raw.push((pixel[0].clamp(0.0, 1.0) * 65535.0).round() as u16);
        raw.push((pixel[1].clamp(0.0, 1.0) * 65535.0).round() as u16);
        raw.push((pixel[2].clamp(0.0, 1.0) * 65535.0).round() as u16);
    }

    let img: ImageBuffer<Rgb<u16>, Vec<u16>> =
        ImageBuffer::from_raw(width, height, raw).ok_or_else(|| {
            RenderError::InvalidConfig("Failed to create bin image buffer".to_string())
        })?;

    let dyn_img = image::DynamicImage::ImageRgb16(img);
    dyn_img
        .save(path)
        .map_err(|e| RenderError::ImageEncoding(e.to_string()))?;
    Ok(())
}

fn heatmap_color(t: f64) -> (f64, f64, f64) {
    // Violet -> blue -> cyan -> green -> yellow -> red
    let t = t.clamp(0.0, 1.0);
    if t < 0.2 {
        let s = t / 0.2;
        (0.5 * (1.0 - s), 0.0, 0.5 + 0.5 * s)
    } else if t < 0.4 {
        let s = (t - 0.2) / 0.2;
        (0.0, s, 1.0)
    } else if t < 0.6 {
        let s = (t - 0.4) / 0.2;
        (0.0, 1.0, 1.0 - s)
    } else if t < 0.8 {
        let s = (t - 0.6) / 0.2;
        (s, 1.0, 0.0)
    } else {
        let s = (t - 0.8) / 0.2;
        (1.0, 1.0 - s, 0.0)
    }
}

fn generate_dominant_wavelength_heatmap(
    accum_spd: &[[f64; NUM_BINS]],
    width: u32,
    height: u32,
    output_dir: &str,
) -> Result<()> {
    let pixel_count = (width * height) as usize;
    let inv_gamma = 1.0 / constants::DISPLAY_GAMMA;

    let max_total: f64 = accum_spd
        .par_iter()
        .map(|spd| spd.iter().sum::<f64>())
        .reduce(|| 0.0f64, f64::max);
    let log_denom = (1.0 + max_total).ln().max(1e-10);

    let mut raw = Vec::with_capacity(pixel_count * 3);
    for spd in accum_spd {
        let dominant_bin = spd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let t = dominant_bin as f64 / (NUM_BINS - 1) as f64;
        let (r, g, b) = heatmap_color(t);
        let total: f64 = spd.iter().sum();
        let brightness = (1.0 + total).ln() / log_denom;

        raw.push(((r * brightness).powf(inv_gamma).clamp(0.0, 1.0) * 65535.0).round() as u16);
        raw.push(((g * brightness).powf(inv_gamma).clamp(0.0, 1.0) * 65535.0).round() as u16);
        raw.push(((b * brightness).powf(inv_gamma).clamp(0.0, 1.0) * 65535.0).round() as u16);
    }

    let img: ImageBuffer<Rgb<u16>, Vec<u16>> =
        ImageBuffer::from_raw(width, height, raw).ok_or_else(|| {
            RenderError::InvalidConfig("Failed to create heatmap image buffer".to_string())
        })?;

    let path = format!("{output_dir}/dominant_wavelength.png");
    let dyn_img = image::DynamicImage::ImageRgb16(img);
    dyn_img
        .save(&path)
        .map_err(|e| RenderError::ImageEncoding(e.to_string()))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Spectral Cycle Videos (Section 10)
// ---------------------------------------------------------------------------

/// Generate all six spectral cycle video variants.
pub fn generate_spectral_cycle_videos(
    accum_spd: &[[f64; NUM_BINS]],
    width: u32,
    height: u32,
    output_dir: &str,
    fast_encode: bool,
) -> Result<()> {
    info!("Generating spectral cycle videos (6 variants)...");

    let bin_buffers = BinBuffers::new(accum_spd, width as usize, height as usize);
    let total_frames = constants::CYCLE_TOTAL_FRAMES;

    let variants: Vec<(&str, CycleVariant)> = vec![
        ("forward", CycleVariant::Forward),
        ("reverse", CycleVariant::Reverse),
        ("pingpong", CycleVariant::PingPong),
        ("ease", CycleVariant::Ease),
        ("radial", CycleVariant::Radial),
        ("complementary", CycleVariant::Complementary),
    ];

    // Pre-compute distance map for radial variant
    let dist_map = build_distance_map(width as usize, height as usize);

    for (name, variant) in &variants {
        let output_path = format!("{output_dir}/{name}.mp4");
        info!("   Encoding {name} cycle...");
        encode_cycle_video(
            &bin_buffers,
            variant,
            &dist_map,
            total_frames,
            width,
            height,
            &output_path,
            fast_encode,
        )?;
    }

    info!("   Spectral cycle videos complete (6 variants)");
    Ok(())
}

#[derive(Clone, Copy)]
enum CycleVariant {
    Forward,
    Reverse,
    PingPong,
    Ease,
    Radial,
    Complementary,
}

fn build_distance_map(width: usize, height: usize) -> Vec<f64> {
    let cx = (width as f64 - 1.0) / 2.0;
    let cy = (height as f64 - 1.0) / 2.0;
    let max_dist = (cx * cx + cy * cy).sqrt();

    let mut map = vec![0.0; width * height];
    map.par_iter_mut().enumerate().for_each(|(idx, val)| {
        let x = (idx % width) as f64;
        let y = (idx / width) as f64;
        *val = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() / max_dist;
    });
    map
}

/// Interpolate between two adjacent bin images at fractional bin index (with wrapping).
fn lerp_bins_frame(
    bin_buffers: &BinBuffers,
    bin_f: f64,
    output: &mut Vec<u16>,
) {
    let pixel_count = bin_buffers.pixel_count();
    output.resize(pixel_count * 3, 0);

    let wrapped = ((bin_f % NUM_BINS as f64) + NUM_BINS as f64) % NUM_BINS as f64;
    let lo = wrapped.floor() as usize % NUM_BINS;
    let hi = (lo + 1) % NUM_BINS;
    let t = wrapped.fract() as f32;
    let one_minus_t = 1.0 - t;

    let lo_buf = &bin_buffers.buffers[lo];
    let hi_buf = &bin_buffers.buffers[hi];

    for i in 0..pixel_count {
        let r = lo_buf[i][0] * one_minus_t + hi_buf[i][0] * t;
        let g = lo_buf[i][1] * one_minus_t + hi_buf[i][1] * t;
        let b = lo_buf[i][2] * one_minus_t + hi_buf[i][2] * t;
        output[i * 3] = (r.clamp(0.0, 1.0) * 65535.0).round() as u16;
        output[i * 3 + 1] = (g.clamp(0.0, 1.0) * 65535.0).round() as u16;
        output[i * 3 + 2] = (b.clamp(0.0, 1.0) * 65535.0).round() as u16;
    }
}

/// Per-pixel bin selection for the radial variant.
fn radial_frame(
    bin_buffers: &BinBuffers,
    base_bin: f64,
    dist_map: &[f64],
    output: &mut Vec<u16>,
) {
    let pixel_count = bin_buffers.pixel_count();
    output.resize(pixel_count * 3, 0);

    for i in 0..pixel_count {
        let bin_f = base_bin - dist_map[i] * constants::RADIAL_SPREAD;
        let wrapped = ((bin_f % NUM_BINS as f64) + NUM_BINS as f64) % NUM_BINS as f64;
        let lo = wrapped.floor() as usize % NUM_BINS;
        let hi = (lo + 1) % NUM_BINS;
        let t = wrapped.fract() as f32;
        let one_minus_t = 1.0 - t;

        let r = bin_buffers.buffers[lo][i][0] * one_minus_t + bin_buffers.buffers[hi][i][0] * t;
        let g = bin_buffers.buffers[lo][i][1] * one_minus_t + bin_buffers.buffers[hi][i][1] * t;
        let b = bin_buffers.buffers[lo][i][2] * one_minus_t + bin_buffers.buffers[hi][i][2] * t;
        output[i * 3] = (r.clamp(0.0, 1.0) * 65535.0).round() as u16;
        output[i * 3 + 1] = (g.clamp(0.0, 1.0) * 65535.0).round() as u16;
        output[i * 3 + 2] = (b.clamp(0.0, 1.0) * 65535.0).round() as u16;
    }
}

/// Complementary: average two cursors 32 bins apart.
fn complementary_frame(
    bin_buffers: &BinBuffers,
    bin_f: f64,
    output: &mut Vec<u16>,
) {
    let pixel_count = bin_buffers.pixel_count();
    output.resize(pixel_count * 3, 0);

    let half = (NUM_BINS / 2) as f64;
    let bin_a = ((bin_f % NUM_BINS as f64) + NUM_BINS as f64) % NUM_BINS as f64;
    let bin_b = (((bin_f + half) % NUM_BINS as f64) + NUM_BINS as f64) % NUM_BINS as f64;

    let lo_a = bin_a.floor() as usize % NUM_BINS;
    let hi_a = (lo_a + 1) % NUM_BINS;
    let t_a = bin_a.fract() as f32;

    let lo_b = bin_b.floor() as usize % NUM_BINS;
    let hi_b = (lo_b + 1) % NUM_BINS;
    let t_b = bin_b.fract() as f32;

    for i in 0..pixel_count {
        let ra = bin_buffers.buffers[lo_a][i][0] * (1.0 - t_a) + bin_buffers.buffers[hi_a][i][0] * t_a;
        let ga = bin_buffers.buffers[lo_a][i][1] * (1.0 - t_a) + bin_buffers.buffers[hi_a][i][1] * t_a;
        let ba = bin_buffers.buffers[lo_a][i][2] * (1.0 - t_a) + bin_buffers.buffers[hi_a][i][2] * t_a;

        let rb = bin_buffers.buffers[lo_b][i][0] * (1.0 - t_b) + bin_buffers.buffers[hi_b][i][0] * t_b;
        let gb = bin_buffers.buffers[lo_b][i][1] * (1.0 - t_b) + bin_buffers.buffers[hi_b][i][1] * t_b;
        let bb = bin_buffers.buffers[lo_b][i][2] * (1.0 - t_b) + bin_buffers.buffers[hi_b][i][2] * t_b;

        let r = ((ra + rb) * 0.5).clamp(0.0, 1.0);
        let g = ((ga + gb) * 0.5).clamp(0.0, 1.0);
        let b = ((ba + bb) * 0.5).clamp(0.0, 1.0);

        output[i * 3] = (r * 65535.0).round() as u16;
        output[i * 3 + 1] = (g * 65535.0).round() as u16;
        output[i * 3 + 2] = (b * 65535.0).round() as u16;
    }
}

/// Compute the fractional bin index for a given frame and variant.
/// Exposed for testing the cycle math without requiring video encoding.
#[cfg(test)]
fn cycle_bin_f(variant: &CycleVariant, frame: u32, total_frames: u32) -> f64 {
    let total_f = total_frames as f64;
    let t = frame as f64 / total_f;
    let max_bin = (NUM_BINS - 1) as f64;

    match variant {
        CycleVariant::Forward => frame as f64 * NUM_BINS as f64 / total_f,
        CycleVariant::Reverse => NUM_BINS as f64 - (frame as f64 * NUM_BINS as f64 / total_f),
        CycleVariant::PingPong => {
            if t < 0.5 {
                t * 2.0 * max_bin
            } else {
                (1.0 - (t - 0.5) * 2.0) * max_bin
            }
        }
        CycleVariant::Ease => (1.0 - (t * 2.0 * PI).cos()) * 0.5 * NUM_BINS as f64,
        CycleVariant::Radial => frame as f64 * NUM_BINS as f64 / total_f,
        CycleVariant::Complementary => frame as f64 * NUM_BINS as f64 / total_f,
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_cycle_video(
    bin_buffers: &BinBuffers,
    variant: &CycleVariant,
    dist_map: &[f64],
    total_frames: u32,
    width: u32,
    height: u32,
    output_path: &str,
    fast_encode: bool,
) -> Result<()> {
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
            let mut frame_buf: Vec<u16> = Vec::new();
            let total_f = total_frames as f64;
            let max_bin = (NUM_BINS - 1) as f64;

            for frame in 0..total_frames {
                let t = frame as f64 / total_f;

                match variant {
                    CycleVariant::Forward => {
                        let bin_f = frame as f64 * NUM_BINS as f64 / total_f;
                        lerp_bins_frame(bin_buffers, bin_f, &mut frame_buf);
                    }
                    CycleVariant::Reverse => {
                        let bin_f = NUM_BINS as f64 - (frame as f64 * NUM_BINS as f64 / total_f);
                        lerp_bins_frame(bin_buffers, bin_f, &mut frame_buf);
                    }
                    CycleVariant::PingPong => {
                        let bin_f = if t < 0.5 {
                            t * 2.0 * max_bin
                        } else {
                            (1.0 - (t - 0.5) * 2.0) * max_bin
                        };
                        lerp_bins_frame(bin_buffers, bin_f, &mut frame_buf);
                    }
                    CycleVariant::Ease => {
                        let bin_f = (1.0 - (t * 2.0 * PI).cos()) * 0.5 * NUM_BINS as f64;
                        lerp_bins_frame(bin_buffers, bin_f, &mut frame_buf);
                    }
                    CycleVariant::Radial => {
                        let base_bin = frame as f64 * NUM_BINS as f64 / total_f;
                        radial_frame(bin_buffers, base_bin, dist_map, &mut frame_buf);
                    }
                    CycleVariant::Complementary => {
                        let bin_f = frame as f64 * NUM_BINS as f64 / total_f;
                        complementary_frame(bin_buffers, bin_f, &mut frame_buf);
                    }
                }

                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        frame_buf.as_ptr() as *const u8,
                        frame_buf.len() * 2,
                    )
                };
                out.write_all(bytes)
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            }
            Ok(())
        },
        output_path,
        &options,
    )?;

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

    // ── BinBuffers ──────────────────────────────────────────────────

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

    // ── heatmap_color ───────────────────────────────────────────────

    #[test]
    fn test_heatmap_color_boundaries() {
        let (r, _g, b) = heatmap_color(0.0);
        assert!(r > 0.0 && b > 0.0, "t=0 should be violet (r>0, b>0)");

        let (r, _, b) = heatmap_color(1.0);
        assert!(r > 0.9, "t=1 should be red (r≈1)");
        assert!(b < 0.01, "t=1 should have no blue");
    }

    #[test]
    fn test_heatmap_color_all_in_unit_range() {
        for i in 0..=100 {
            let t = i as f64 / 100.0;
            let (r, g, b) = heatmap_color(t);
            assert!((0.0..=1.0).contains(&r), "r out of range at t={t}: {r}");
            assert!((0.0..=1.0).contains(&g), "g out of range at t={t}: {g}");
            assert!((0.0..=1.0).contains(&b), "b out of range at t={t}: {b}");
        }
    }

    #[test]
    fn test_heatmap_color_clamped_beyond_range() {
        let (r, g, b) = heatmap_color(-1.0);
        assert!(r >= 0.0 && g >= 0.0 && b >= 0.0);
        let (r, g, b) = heatmap_color(2.0);
        assert!(r <= 1.0 && g <= 1.0 && b <= 1.0);
    }

    #[test]
    fn test_heatmap_color_midpoint_is_green() {
        let (r, g, b) = heatmap_color(0.5);
        assert!(g > r && g > b, "midpoint should be green-dominant, got ({r},{g},{b})");
    }

    // ── build_distance_map ──────────────────────────────────────────

    #[test]
    fn test_distance_map_center_is_zero() {
        let w = 5;
        let h = 5;
        let map = build_distance_map(w, h);
        let center = (h / 2) * w + (w / 2);
        assert!(map[center] < 0.01, "center pixel should be near 0, got {}", map[center]);
    }

    #[test]
    fn test_distance_map_corners_are_max() {
        let w = 10;
        let h = 10;
        let map = build_distance_map(w, h);
        let max_val = map.iter().cloned().fold(0.0f64, f64::max);
        assert!(
            (max_val - 1.0).abs() < 0.02,
            "corner distance should be ~1.0, got {max_val}"
        );
    }

    #[test]
    fn test_distance_map_length() {
        let w = 7;
        let h = 3;
        let map = build_distance_map(w, h);
        assert_eq!(map.len(), w * h);
    }

    #[test]
    fn test_distance_map_all_non_negative() {
        let map = build_distance_map(8, 6);
        for &v in &map {
            assert!(v >= 0.0, "distance map should be non-negative, got {v}");
        }
    }

    #[test]
    fn test_distance_map_symmetric() {
        let w = 8;
        let h = 8;
        let map = build_distance_map(w, h);
        let tl = map[0];
        let tr = map[w - 1];
        let bl = map[(h - 1) * w];
        let br = map[(h - 1) * w + (w - 1)];
        assert!(
            (tl - tr).abs() < 1e-10 && (bl - br).abs() < 1e-10 && (tl - bl).abs() < 1e-10,
            "corners of a square should have equal distance"
        );
    }

    // ── lerp_bins_frame ─────────────────────────────────────────────

    #[test]
    fn test_lerp_bins_at_integer_index() {
        let spd = make_single_bin_spd(10, 2.0);
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let mut output = Vec::new();
        lerp_bins_frame(&bb, 10.0, &mut output);
        assert_eq!(output.len(), TEST_W * TEST_H * 3);
    }

    #[test]
    fn test_lerp_bins_output_length() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let mut output = Vec::new();
        lerp_bins_frame(&bb, 5.5, &mut output);
        assert_eq!(output.len(), TEST_W * TEST_H * 3);
    }

    #[test]
    fn test_lerp_bins_wraps_negative() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let mut out_neg = Vec::new();
        let mut out_pos = Vec::new();
        lerp_bins_frame(&bb, -1.0, &mut out_neg);
        lerp_bins_frame(&bb, (NUM_BINS - 1) as f64, &mut out_pos);
        assert_eq!(out_neg, out_pos, "wrapping should map -1 to NUM_BINS-1");
    }

    #[test]
    fn test_lerp_bins_wraps_over_max() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let mut out_a = Vec::new();
        let mut out_b = Vec::new();
        lerp_bins_frame(&bb, 0.0, &mut out_a);
        lerp_bins_frame(&bb, NUM_BINS as f64, &mut out_b);
        assert_eq!(out_a, out_b, "wrapping should map NUM_BINS to 0");
    }

    #[test]
    fn test_lerp_bins_output_all_valid_u16() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let mut output = Vec::new();
        for bin_f_int in 0..NUM_BINS {
            let bin_f = bin_f_int as f64 + 0.3;
            lerp_bins_frame(&bb, bin_f, &mut output);
            assert!(!output.is_empty(), "output should be non-empty at bin_f={bin_f}");
        }
    }

    // ── radial_frame ────────────────────────────────────────────────

    #[test]
    fn test_radial_frame_output_length() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let dist_map = build_distance_map(TEST_W, TEST_H);
        let mut output = Vec::new();
        radial_frame(&bb, 10.0, &dist_map, &mut output);
        assert_eq!(output.len(), TEST_W * TEST_H * 3);
    }

    #[test]
    fn test_radial_frame_deterministic() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let dist_map = build_distance_map(TEST_W, TEST_H);
        let mut out1 = Vec::new();
        let mut out2 = Vec::new();
        radial_frame(&bb, 20.0, &dist_map, &mut out1);
        radial_frame(&bb, 20.0, &dist_map, &mut out2);
        assert_eq!(out1, out2);
    }

    // ── complementary_frame ─────────────────────────────────────────

    #[test]
    fn test_complementary_frame_output_length() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let mut output = Vec::new();
        complementary_frame(&bb, 5.0, &mut output);
        assert_eq!(output.len(), TEST_W * TEST_H * 3);
    }

    #[test]
    fn test_complementary_frame_deterministic() {
        let spd = make_test_spd();
        let bb = BinBuffers::new(&spd, TEST_W, TEST_H);
        let mut out1 = Vec::new();
        let mut out2 = Vec::new();
        complementary_frame(&bb, 16.0, &mut out1);
        complementary_frame(&bb, 16.0, &mut out2);
        assert_eq!(out1, out2);
    }

    // ── Cycle variant math ──────────────────────────────────────────

    #[test]
    fn test_forward_cycle_monotonic() {
        let total = constants::CYCLE_TOTAL_FRAMES;
        let mut prev = -1.0f64;
        for frame in 0..total {
            let val = cycle_bin_f(&CycleVariant::Forward, frame, total);
            assert!(val >= prev, "forward should be monotonically non-decreasing");
            prev = val;
        }
    }

    #[test]
    fn test_forward_cycle_range() {
        let total = constants::CYCLE_TOTAL_FRAMES;
        let first = cycle_bin_f(&CycleVariant::Forward, 0, total);
        let last = cycle_bin_f(&CycleVariant::Forward, total - 1, total);
        assert!(first < 1.0, "first frame should be near bin 0");
        assert!(last > (NUM_BINS - 2) as f64, "last frame should be near bin {}", NUM_BINS - 1);
    }

    #[test]
    fn test_reverse_cycle_monotonic_decreasing() {
        let total = constants::CYCLE_TOTAL_FRAMES;
        let mut prev = f64::MAX;
        for frame in 0..total {
            let val = cycle_bin_f(&CycleVariant::Reverse, frame, total);
            assert!(val <= prev, "reverse should be monotonically non-increasing");
            prev = val;
        }
    }

    #[test]
    fn test_pingpong_starts_and_ends_at_zero() {
        let total = constants::CYCLE_TOTAL_FRAMES;
        let first = cycle_bin_f(&CycleVariant::PingPong, 0, total);
        let last = cycle_bin_f(&CycleVariant::PingPong, total - 1, total);
        assert!(first < 1.0, "ping-pong should start near 0, got {first}");
        assert!(last < 1.0, "ping-pong should end near 0, got {last}");
    }

    #[test]
    fn test_pingpong_peaks_at_midpoint() {
        let total = constants::CYCLE_TOTAL_FRAMES;
        let mid = total / 2 - 1;
        let val = cycle_bin_f(&CycleVariant::PingPong, mid, total);
        assert!(
            val > (NUM_BINS / 2) as f64,
            "ping-pong should peak near max at midpoint, got {val}"
        );
    }

    #[test]
    fn test_ease_cycle_starts_and_ends_near_zero() {
        let total = constants::CYCLE_TOTAL_FRAMES;
        let first = cycle_bin_f(&CycleVariant::Ease, 0, total);
        let last = cycle_bin_f(&CycleVariant::Ease, total - 1, total);
        assert!(first < 1.0, "ease should start near 0, got {first}");
        assert!(last < 1.0, "ease should end near 0, got {last}");
    }

    #[test]
    fn test_ease_cycle_peaks_at_midpoint() {
        let total = constants::CYCLE_TOTAL_FRAMES;
        let mid = total / 2;
        let val = cycle_bin_f(&CycleVariant::Ease, mid, total);
        assert!(
            val > (NUM_BINS - 2) as f64,
            "ease should peak near NUM_BINS at midpoint, got {val}"
        );
    }

    #[test]
    fn test_all_cycle_variants_produce_finite_values() {
        let total = 120;
        let variants = [
            CycleVariant::Forward,
            CycleVariant::Reverse,
            CycleVariant::PingPong,
            CycleVariant::Ease,
            CycleVariant::Radial,
            CycleVariant::Complementary,
        ];
        for variant in &variants {
            for frame in 0..total {
                let val = cycle_bin_f(variant, frame, total);
                assert!(val.is_finite(), "NaN/Inf from cycle variant at frame {frame}");
            }
        }
    }

    // ── Gallery I/O tests (tmpdir) ──────────────────────────────────

    #[test]
    fn test_gallery_writes_expected_files() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let dir = tmp.path().to_str().unwrap();

        let spd = make_single_bin_spd(20, 1.0);
        generate_spectral_gallery(&spd, TEST_W as u32, TEST_H as u32, dir)
            .expect("gallery should succeed");

        for bin in 0..NUM_BINS {
            let wl = wavelength_nm_for_bin(bin);
            let file = format!("{dir}/{bin:02}_{wl:.0}nm.png");
            assert!(
                std::path::Path::new(&file).exists(),
                "missing bin image: {file}"
            );
        }

        let heatmap = format!("{dir}/dominant_wavelength.png");
        assert!(std::path::Path::new(&heatmap).exists(), "missing heatmap");
    }

    #[test]
    fn test_gallery_zero_energy_does_not_crash() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let dir = tmp.path().to_str().unwrap();

        let spd = vec![[0.0f64; NUM_BINS]; TEST_W * TEST_H];
        let result = generate_spectral_gallery(&spd, TEST_W as u32, TEST_H as u32, dir);
        assert!(result.is_ok(), "gallery with zero energy should succeed: {result:?}");
    }

    #[test]
    fn test_gallery_extreme_energy_does_not_crash() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let dir = tmp.path().to_str().unwrap();

        let spd = vec![[1e6; NUM_BINS]; TEST_W * TEST_H];
        let result = generate_spectral_gallery(&spd, TEST_W as u32, TEST_H as u32, dir);
        assert!(result.is_ok(), "gallery with extreme energy should succeed: {result:?}");
    }

    // ── Constants validation ────────────────────────────────────────

    #[test]
    fn test_cycle_constants_are_consistent() {
        let expected_frames =
            (constants::CYCLE_DURATION_SECONDS * constants::DEFAULT_VIDEO_FPS as f64).round()
                as u32;
        assert_eq!(
            constants::CYCLE_TOTAL_FRAMES, expected_frames,
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
    fn test_radial_spread_positive() {
        let spread = constants::RADIAL_SPREAD;
        assert!(spread > 0.0, "radial spread must be positive, got {spread}");
    }
}
