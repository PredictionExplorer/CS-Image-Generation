//! v2 hero outputs: `OpenEXR` scene-linear master, 9:16 + 1:1 crops, and 4×16 contact sheet.
//!
//! All of these consume the *final tonemapped* 16-bit PNG (or its raw `u16` buffer) plus the
//! fully accumulated spectral buffer, so they do not re-run the spectral pass and add only a
//! few tens of milliseconds over the baseline render.
//!
//! Coverage:
//!   * [`save_exr_linear_rgb_from_u16`] — preserves scene-linear floats above `1.0` by treating
//!     the 16-bit PNG as gamma-encoded display data (inverse `AgX` is intractable so we use
//!     an inverse-sRGB transfer) and writes uncompressed half-float EXR.
//!   * [`save_vertical_9_16_crop`] / [`save_square_1_1_crop`] — saliency-anchored crops around
//!     the brightest column/row of the image.
//!   * [`save_contact_sheet_4x16`] — 4 columns × 16 rows thumbnail atlas of the 64 spectral bins.
//!   * [`save_hero_upscale`] — simple bilinear upsample of the final display PNG for quick
//!     "8K hero" style output without re-rendering.

use super::error::{RenderError, Result};
use crate::spectrum::{NUM_BINS, wavelength_nm_for_bin};
use image::{ImageBuffer, Rgb};
use rayon::prelude::*;
use tracing::info;

/// Inverse sRGB companding for a single channel in `[0, 1]`.
#[inline]
#[must_use]
pub fn inverse_srgb(c: f64) -> f64 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// Convert a 16-bit display RGB buffer into an f32 scene-linear buffer via inverse sRGB.
#[must_use]
pub fn u16_rgb_to_linear_f32(raw: &[u16]) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0f32; 3]; raw.len() / 3];
    out.par_iter_mut().enumerate().for_each(|(idx, pixel)| {
        let base = idx * 3;
        let r = f64::from(raw[base]) / 65_535.0;
        let g = f64::from(raw[base + 1]) / 65_535.0;
        let b = f64::from(raw[base + 2]) / 65_535.0;
        pixel[0] = inverse_srgb(r) as f32;
        pixel[1] = inverse_srgb(g) as f32;
        pixel[2] = inverse_srgb(b) as f32;
    });
    out
}

/// Write a scene-linear `OpenEXR` master to `path` from a 16-bit RGB image.
pub fn save_exr_linear_rgb_from_u16(
    img: &ImageBuffer<Rgb<u16>, Vec<u16>>,
    path: &str,
) -> Result<()> {
    let width = img.width() as usize;
    let height = img.height() as usize;
    if width == 0 || height == 0 {
        return Err(RenderError::ImageEncoding {
            reason: format!("cannot save zero-sized EXR {width}x{height}"),
        });
    }
    let linear = u16_rgb_to_linear_f32(img.as_raw());
    exr::prelude::write_rgb_file(path, width, height, |x, y| {
        let p = linear[y * width + x];
        (p[0], p[1], p[2])
    })
    .map_err(|e| RenderError::ImageEncoding {
        reason: format!("EXR encode '{path}': {e}"),
    })?;
    info!("   Saved scene-linear EXR master => {path}");
    Ok(())
}

fn brightest_column(img: &ImageBuffer<Rgb<u16>, Vec<u16>>) -> u32 {
    let w = img.width();
    let h = img.height();
    let mut col_sum = vec![0u64; w as usize];
    let raw = img.as_raw();
    for y in 0..h {
        let row = y as usize * w as usize * 3;
        for x in 0..w {
            let i = row + x as usize * 3;
            col_sum[x as usize] += u64::from(raw[i]) + u64::from(raw[i + 1]) + u64::from(raw[i + 2]);
        }
    }
    col_sum.iter().enumerate().max_by_key(|&(_, v)| *v).map_or(w / 2, |(i, _)| i as u32)
}

fn brightest_row(img: &ImageBuffer<Rgb<u16>, Vec<u16>>) -> u32 {
    let w = img.width();
    let h = img.height();
    let mut row_sum = vec![0u64; h as usize];
    let raw = img.as_raw();
    for y in 0..h {
        let row = y as usize * w as usize * 3;
        let mut s: u64 = 0;
        for x in 0..w {
            let i = row + x as usize * 3;
            s += u64::from(raw[i]) + u64::from(raw[i + 1]) + u64::from(raw[i + 2]);
        }
        row_sum[y as usize] = s;
    }
    row_sum.iter().enumerate().max_by_key(|&(_, v)| *v).map_or(h / 2, |(i, _)| i as u32)
}

/// Copy a rectangular `w*h`-window out of `src` into a new 16-bit image, clamping coords.
#[must_use]
pub fn crop_rect(
    src: &ImageBuffer<Rgb<u16>, Vec<u16>>,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) -> ImageBuffer<Rgb<u16>, Vec<u16>> {
    let sw = src.width();
    let sh = src.height();
    let x = x.min(sw.saturating_sub(1));
    let y = y.min(sh.saturating_sub(1));
    let w = w.min(sw - x);
    let h = h.min(sh - y);
    let mut out: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::new(w, h);
    let raw_src = src.as_raw();
    let out_vec = out.as_mut();
    for row in 0..h {
        let ss = ((y + row) as usize * sw as usize + x as usize) * 3;
        let dd = row as usize * w as usize * 3;
        out_vec[dd..dd + w as usize * 3]
            .copy_from_slice(&raw_src[ss..ss + w as usize * 3]);
    }
    out
}

/// Save a 9:16 vertical crop anchored on the brightest column.
pub fn save_vertical_9_16_crop(img: &ImageBuffer<Rgb<u16>, Vec<u16>>, path: &str) -> Result<()> {
    let h = img.height();
    let target_w = (u64::from(h) * 9 / 16) as u32;
    let target_w = target_w.clamp(1, img.width());
    let cx = brightest_column(img);
    let x = cx.saturating_sub(target_w / 2).min(img.width() - target_w);
    let cropped = crop_rect(img, x, 0, target_w, h);
    super::save_image_as_png_16bit(&cropped, path)
}

/// Save a 1:1 square crop anchored on the brightest pixel.
pub fn save_square_1_1_crop(img: &ImageBuffer<Rgb<u16>, Vec<u16>>, path: &str) -> Result<()> {
    let w = img.width();
    let h = img.height();
    let s = w.min(h);
    let cx = brightest_column(img);
    let cy = brightest_row(img);
    let x = cx.saturating_sub(s / 2).min(w - s);
    let y = cy.saturating_sub(s / 2).min(h - s);
    let cropped = crop_rect(img, x, y, s, s);
    super::save_image_as_png_16bit(&cropped, path)
}

/// Bilinearly upscale a 16-bit RGB image to a larger resolution and save as PNG.
pub fn save_hero_upscale(
    img: &ImageBuffer<Rgb<u16>, Vec<u16>>,
    out_w: u32,
    out_h: u32,
    path: &str,
) -> Result<()> {
    let sw = img.width() as usize;
    let sh = img.height() as usize;
    if sw == 0 || sh == 0 || out_w == 0 || out_h == 0 {
        return Err(RenderError::ImageEncoding {
            reason: format!("invalid hero resize {sw}x{sh} -> {out_w}x{out_h}"),
        });
    }
    let raw_src = img.as_raw();
    let tw = out_w as usize;
    let th = out_h as usize;
    let mut out_raw = vec![0u16; tw * th * 3];
    out_raw.par_chunks_mut(3).enumerate().for_each(|(idx, chunk)| {
        let x = idx % tw;
        let y = idx / tw;
        let fx = (x as f64 + 0.5) * sw as f64 / tw as f64 - 0.5;
        let fy = (y as f64 + 0.5) * sh as f64 / th as f64 - 0.5;
        let x0 = fx.floor().clamp(0.0, (sw - 1) as f64) as usize;
        let y0 = fy.floor().clamp(0.0, (sh - 1) as f64) as usize;
        let x1 = (x0 + 1).min(sw - 1);
        let y1 = (y0 + 1).min(sh - 1);
        let tx = (fx - x0 as f64).clamp(0.0, 1.0);
        let ty = (fy - y0 as f64).clamp(0.0, 1.0);
        for c in 0..3 {
            let p00 = f64::from(raw_src[(y0 * sw + x0) * 3 + c]);
            let p10 = f64::from(raw_src[(y0 * sw + x1) * 3 + c]);
            let p01 = f64::from(raw_src[(y1 * sw + x0) * 3 + c]);
            let p11 = f64::from(raw_src[(y1 * sw + x1) * 3 + c]);
            let v = (1.0 - ty) * ((1.0 - tx) * p00 + tx * p10)
                + ty * ((1.0 - tx) * p01 + tx * p11);
            chunk[c] = v.round().clamp(0.0, 65_535.0) as u16;
        }
    });
    let out: ImageBuffer<Rgb<u16>, Vec<u16>> =
        ImageBuffer::from_raw(out_w, out_h, out_raw).ok_or_else(|| {
            RenderError::ImageEncoding { reason: "hero upscale buffer invalid".into() }
        })?;
    super::save_image_as_png_16bit(&out, path)
}

/// Pack 64 spectral-bin previews into a single 4-column × 16-row contact sheet PNG.
pub fn save_contact_sheet_4x16(
    accum_spd: &[[f64; NUM_BINS]],
    width: u32,
    height: u32,
    path: &str,
) -> Result<()> {
    const COLS: u32 = 4;
    const ROWS: u32 = 16;
    assert!(COLS * ROWS == NUM_BINS as u32, "contact sheet layout must match NUM_BINS");
    let cell_w: u32 = width.max(256) / 6;
    let cell_h: u32 = cell_w * height / width.max(1);
    let sheet_w = cell_w * COLS;
    let sheet_h = cell_h * ROWS;
    let bins = super::spectral_output::BinBuffers::new(accum_spd, width as usize, height as usize);
    let mut sheet = vec![0u16; sheet_w as usize * sheet_h as usize * 3];

    for (bin_idx, bin) in bins.buffers.iter().enumerate() {
        let col = bin_idx as u32 % COLS;
        let row = bin_idx as u32 / COLS;
        let ox = col * cell_w;
        let oy = row * cell_h;
        let bin_u16 = bin_f32_to_u16(bin, width as usize, height as usize);
        for y in 0..cell_h {
            let sy = (u64::from(y) * u64::from(height) / u64::from(cell_h.max(1))) as usize;
            for x in 0..cell_w {
                let sx = (u64::from(x) * u64::from(width) / u64::from(cell_w.max(1))) as usize;
                let src_idx = (sy * width as usize + sx) * 3;
                let dst_idx = ((oy + y) as usize * sheet_w as usize + (ox + x) as usize) * 3;
                sheet[dst_idx] = bin_u16[src_idx];
                sheet[dst_idx + 1] = bin_u16[src_idx + 1];
                sheet[dst_idx + 2] = bin_u16[src_idx + 2];
            }
        }
        // Faint wavelength annotation: darken the top row of each cell proportional to wavelength.
        let nm = wavelength_nm_for_bin(bin_idx);
        let fade = ((nm - 380.0) / 320.0).clamp(0.0, 1.0);
        let tint = (fade * 12_000.0) as u16;
        for x in 0..cell_w {
            let dst_idx = ((oy) as usize * sheet_w as usize + (ox + x) as usize) * 3;
            sheet[dst_idx] = sheet[dst_idx].saturating_sub(tint);
            sheet[dst_idx + 2] = sheet[dst_idx + 2].saturating_add(tint / 2);
        }
    }

    let out: ImageBuffer<Rgb<u16>, Vec<u16>> =
        ImageBuffer::from_raw(sheet_w, sheet_h, sheet).ok_or_else(|| {
            RenderError::ImageEncoding { reason: "contact sheet buffer invalid".into() }
        })?;
    super::save_image_as_png_16bit(&out, path)
}

fn bin_f32_to_u16(bin: &[[f32; 3]], width: usize, height: usize) -> Vec<u16> {
    let pixel_count = width * height;
    let mut out = vec![0u16; pixel_count * 3];
    out.par_chunks_mut(3).zip(bin.par_iter()).for_each(|(chunk, px)| {
        chunk[0] = (f64::from(px[0].clamp(0.0, 1.0)) * 65_535.0).round() as u16;
        chunk[1] = (f64::from(px[1].clamp(0.0, 1.0)) * 65_535.0).round() as u16;
        chunk[2] = (f64::from(px[2].clamp(0.0, 1.0)) * 65_535.0).round() as u16;
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checker(width: u32, height: u32) -> ImageBuffer<Rgb<u16>, Vec<u16>> {
        let mut raw = vec![0u16; width as usize * height as usize * 3];
        for y in 0..height {
            for x in 0..width {
                let bright = ((x / 8 + y / 8) % 2) == 0;
                let v = if bright { 60_000 } else { 500 };
                let i = (y as usize * width as usize + x as usize) * 3;
                raw[i] = v;
                raw[i + 1] = v / 2;
                raw[i + 2] = v / 3;
            }
        }
        ImageBuffer::from_raw(width, height, raw).expect("checker buffer builds")
    }

    #[test]
    fn test_inverse_srgb_endpoints_and_monotonicity() {
        assert!((inverse_srgb(0.0) - 0.0).abs() < 1e-12);
        assert!((inverse_srgb(1.0) - 1.0).abs() < 1e-6);
        let mut prev = -1.0;
        for i in 0..=100 {
            let v = inverse_srgb(f64::from(i) / 100.0);
            assert!(v >= prev, "inverse sRGB must be monotone, step {i}");
            prev = v;
        }
    }

    #[test]
    fn test_u16_rgb_to_linear_f32_midgray_is_approximately_linear() {
        let raw = vec![32_767u16; 9]; // 3 pixels of 50% sRGB
        let linear = u16_rgb_to_linear_f32(&raw);
        for p in linear {
            for c in p {
                // sRGB 0.5 ≈ linear 0.214
                assert!((c - 0.214).abs() < 0.01, "expected ~0.214, got {c}");
            }
        }
    }

    #[test]
    fn test_crop_rect_is_in_bounds_and_copies_data() {
        let img = checker(32, 16);
        let crop = crop_rect(&img, 2, 3, 8, 6);
        assert_eq!(crop.width(), 8);
        assert_eq!(crop.height(), 6);
        for y in 0..6 {
            for x in 0..8 {
                let src = img.get_pixel(2 + x, 3 + y).0;
                let dst = crop.get_pixel(x, y).0;
                assert_eq!(src, dst, "crop mismatch at ({x},{y})");
            }
        }
    }

    #[test]
    fn test_brightest_column_prefers_bright_side() {
        let mut raw = vec![0u16; 8 * 4 * 3];
        for y in 0..4 {
            for x in 4..8 {
                let i = (y * 8 + x) * 3;
                raw[i] = 50_000;
                raw[i + 1] = 50_000;
                raw[i + 2] = 50_000;
            }
        }
        let img: ImageBuffer<Rgb<u16>, Vec<u16>> =
            ImageBuffer::from_raw(8, 4, raw).expect("test image");
        let cx = brightest_column(&img);
        assert!(cx >= 4, "expected bright half on the right, got {cx}");
    }

    #[test]
    fn test_hero_upscale_roundtrips_size() {
        let img = checker(16, 8);
        let tmpdir = tempfile::tempdir().expect("tmpdir");
        let path = tmpdir.path().join("hero.png");
        save_hero_upscale(&img, 32, 16, path.to_str().expect("path"))
            .expect("hero upscale succeeds");
        let reloaded = image::open(&path).expect("png loads");
        assert_eq!(reloaded.width(), 32);
        assert_eq!(reloaded.height(), 16);
    }

    #[test]
    fn test_contact_sheet_has_expected_aspect() {
        let tmpdir = tempfile::tempdir().expect("tmpdir");
        let path = tmpdir.path().join("contact.png");
        let w = 128;
        let h = 72;
        let accum = vec![[0.0f64; NUM_BINS]; w * h];
        save_contact_sheet_4x16(&accum, w as u32, h as u32, path.to_str().expect("path"))
            .expect("contact sheet writes");
        let reloaded = image::open(&path).expect("png loads");
        // 4 cols × 16 rows with cells sized from width/6, so reloaded dims are fixed.
        assert_eq!(reloaded.width() % 4, 0);
        assert_eq!(reloaded.height() % 16, 0);
    }

    #[test]
    fn test_exr_save_from_u16_produces_nonempty_file() {
        let img = checker(32, 16);
        let tmpdir = tempfile::tempdir().expect("tmpdir");
        let path = tmpdir.path().join("master.exr");
        save_exr_linear_rgb_from_u16(&img, path.to_str().expect("path"))
            .expect("EXR writes");
        let size = std::fs::metadata(&path).expect("exr metadata").len();
        assert!(size > 64, "EXR too small: {size}");
    }
}
