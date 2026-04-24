//! Per-bin spectral gallery PNG generation.

use super::BinBuffers;
use crate::render::constants;
use crate::render::error::{RenderError, Result};
use crate::spectrum::{NUM_BINS, wavelength_nm_for_bin};
use image::{ImageBuffer, Rgb};
use rayon::prelude::*;
use std::path::Path;
use tracing::info;

/// Generate 64 per-bin 16-bit PNGs into `output_dir`.
///
/// # Errors
///
/// Returns an error if `output_dir` cannot be created or any PNG cannot be
/// encoded.
pub fn generate_spectral_gallery(
    accum_spd: &[[f64; NUM_BINS]],
    width: u32,
    height: u32,
    output_dir: impl AsRef<Path>,
) -> Result<()> {
    let output_dir = output_dir.as_ref();
    std::fs::create_dir_all(output_dir).map_err(|e| RenderError::ImageEncoding {
        reason: format!(
            "Failed to create spectral gallery directory {}: {e}",
            output_dir.display()
        ),
    })?;

    info!("Building BinBuffers ({NUM_BINS} bins)...");
    let bin_buffers = BinBuffers::new(accum_spd, width as usize, height as usize);

    info!("Generating spectral gallery ({NUM_BINS} bin images)...");
    (0..NUM_BINS).into_par_iter().try_for_each(|bin| {
        let wavelength = wavelength_nm_for_bin(bin);
        let filename = output_dir.join(format!("{bin:02}_{wavelength:.0}nm.png"));
        save_bin_image(&bin_buffers.buffers[bin], width, height, &filename)
    })?;

    info!("   Spectral gallery complete ({NUM_BINS} images)");
    Ok(())
}

fn save_bin_image(buf: &[[f32; 3]], width: u32, height: u32, path: &Path) -> Result<()> {
    use crate::utils::f64_to_u16_saturating;
    let pixel_count = (width * height) as usize;
    let mut raw = Vec::with_capacity(pixel_count * 3);
    for pixel in buf {
        raw.push(f64_to_u16_saturating(
            f64::from(pixel[0].clamp(0.0, 1.0)) * constants::U16_MAX_F64,
        ));
        raw.push(f64_to_u16_saturating(
            f64::from(pixel[1].clamp(0.0, 1.0)) * constants::U16_MAX_F64,
        ));
        raw.push(f64_to_u16_saturating(
            f64::from(pixel[2].clamp(0.0, 1.0)) * constants::U16_MAX_F64,
        ));
    }

    let img: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::from_raw(width, height, raw)
        .ok_or_else(|| RenderError::ImageEncoding {
            reason: "Failed to create bin image buffer".into(),
        })?;

    let dyn_img = image::DynamicImage::ImageRgb16(img);
    dyn_img.save(path).map_err(|e| RenderError::ImageEncoding {
        reason: format!("Failed to save {}: {e}", path.display()),
    })?;
    Ok(())
}
