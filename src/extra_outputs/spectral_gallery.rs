//! Spectral decomposition gallery — individual images per spectral bin,
//! revealing the hidden wavelength structure of the artwork.

use crate::render::context::RenderContext;
use crate::render::velocity_hdr;
use crate::render::{
    SpectralRenderSettings, SpectralScene, accumulate_spectral_steps,
    apply_energy_density_shift, constants, default_accumulation_backend, save_image_as_png_16bit,
};
use crate::spectrum::{NUM_BINS, wavelength_nm_for_bin, wavelength_to_rgb};
use image::ImageBuffer;
use rayon::prelude::*;
use tracing::info;

pub fn render_spectral_gallery(
    scene: SpectralScene<'_>,
    settings: SpectralRenderSettings<'_>,
    output_dir: &str,
) -> crate::render::error::Result<()> {
    info!("Rendering spectral decomposition gallery ({} bins, full accumulation)...", NUM_BINS);

    let resolved = settings.resolved_config;
    let width = resolved.width;
    let height = resolved.height;
    let ctx = RenderContext::new(width, height, scene.positions, settings.aspect_correction);
    let mut accum_spd = vec![[0.0f64; NUM_BINS]; ctx.pixel_count()];

    let total_steps = scene.step_count();
    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);

    accumulate_spectral_steps(
        &mut accum_spd,
        None,
        scene,
        &ctx,
        &velocity_calc,
        0,
        total_steps,
        settings.render_config.hdr_scale,
        default_accumulation_backend(),
    );

    render_spectral_gallery_from_buffer(accum_spd, width, height, output_dir)
}

/// Write spectral gallery images from a pre-computed accumulation buffer.
///
/// Applies energy-density shift, then writes one 16-bit PNG per spectral bin
/// plus a dominant-wavelength heatmap. Accepts the raw (pre-shift) buffer
/// produced by the main video render pass, avoiding a redundant accumulation.
pub fn render_spectral_gallery_from_buffer(
    mut accum_spd: Vec<[f64; NUM_BINS]>,
    width: u32,
    height: u32,
    output_dir: &str,
) -> crate::render::error::Result<()> {
    info!("Writing spectral decomposition gallery ({} bins)...", NUM_BINS);

    let pixel_count = (width * height) as usize;
    apply_energy_density_shift(&mut accum_spd);

    for bin in 0..NUM_BINS {
        let wavelength = wavelength_nm_for_bin(bin);
        let (tint_r, tint_g, tint_b) = wavelength_to_rgb(wavelength);

        let max_val: f64 = accum_spd
            .par_iter()
            .map(|spd| spd[bin])
            .reduce(|| 0.0f64, f64::max)
            .max(1e-10);

        let mut buf_16bit = vec![0u16; pixel_count * 3];
        buf_16bit.par_chunks_mut(3).zip(accum_spd.par_iter()).for_each(|(chunk, spd)| {
            let normalized = (spd[bin] / max_val).clamp(0.0, 1.0);
            chunk[0] = ((normalized * tint_r).powf(1.0 / 2.2) * 65535.0).round() as u16;
            chunk[1] = ((normalized * tint_g).powf(1.0 / 2.2) * 65535.0).round() as u16;
            chunk[2] = ((normalized * tint_b).powf(1.0 / 2.2) * 65535.0).round() as u16;
        });

        let image = ImageBuffer::from_raw(width, height, buf_16bit).ok_or_else(|| {
            crate::render::error::RenderError::ImageEncoding(format!(
                "Failed to create spectral bin {} image",
                bin
            ))
        })?;

        let path = format!("{}/{:02}_{:.0}nm.png", output_dir, bin, wavelength);
        save_image_as_png_16bit(&image, &path)?;
    }

    render_dominant_wavelength_map(&accum_spd, width, height, output_dir)?;

    info!("   Saved {} spectral bin images + heatmap => {}/", NUM_BINS, output_dir);
    Ok(())
}

fn render_dominant_wavelength_map(
    accum_spd: &[[f64; NUM_BINS]],
    width: u32,
    height: u32,
    output_dir: &str,
) -> crate::render::error::Result<()> {
    let pixel_count = (width * height) as usize;
    let mut buf_16bit = vec![0u16; pixel_count * 3];

    let max_total: f64 = accum_spd
        .par_iter()
        .map(|spd| spd.iter().sum::<f64>())
        .reduce(|| 0.0f64, f64::max)
        .max(1e-10);
    let log_max = (1.0 + max_total).ln();

    buf_16bit.par_chunks_mut(3).zip(accum_spd.par_iter()).for_each(|(chunk, spd)| {
        let total: f64 = spd.iter().sum();
        if total < 1e-10 {
            return;
        }

        let dominant_bin = spd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let t = dominant_bin as f64 / (NUM_BINS - 1) as f64;
        let (r, g, b) = heatmap_color(t);
        let brightness = ((1.0 + total).ln() / log_max).clamp(0.0, 1.0);
        chunk[0] = ((r * brightness).powf(1.0 / 2.2) * 65535.0).round() as u16;
        chunk[1] = ((g * brightness).powf(1.0 / 2.2) * 65535.0).round() as u16;
        chunk[2] = ((b * brightness).powf(1.0 / 2.2) * 65535.0).round() as u16;
    });

    let image = ImageBuffer::from_raw(width, height, buf_16bit).ok_or_else(|| {
        crate::render::error::RenderError::ImageEncoding(
            "Failed to create spectral heatmap".to_string(),
        )
    })?;

    let path = format!("{}/heatmap.png", output_dir);
    save_image_as_png_16bit(&image, &path)
}

fn heatmap_color(t: f64) -> (f64, f64, f64) {
    let t = t.clamp(0.0, 1.0);
    if t < 0.25 {
        let s = t / 0.25;
        (0.5 * (1.0 - s), 0.0, 0.5 + 0.5 * s)
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (0.0, s, 1.0 - 0.5 * s)
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (s, 1.0 - 0.5 * s, 0.5 * (1.0 - s))
    } else {
        let s = (t - 0.75) / 0.25;
        (1.0, 0.5 * (1.0 - s), 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heatmap_color_violet_at_zero() {
        let (r, _g, b) = heatmap_color(0.0);
        assert!(b >= r, "t=0 should be violet-ish (blue >= red)");
        assert_eq!(r, 0.5);
        assert_eq!(b, 0.5);
    }

    #[test]
    fn test_heatmap_color_green_at_half() {
        let (_r, g, _b) = heatmap_color(0.5);
        assert!(g >= 0.9, "t=0.5 should have high green, got {g}");
    }

    #[test]
    fn test_heatmap_color_red_at_one() {
        let (r, g, b) = heatmap_color(1.0);
        assert_eq!(r, 1.0);
        assert_eq!(g, 0.0);
        assert_eq!(b, 0.0);
    }

    #[test]
    fn test_heatmap_color_all_in_unit_range() {
        for i in 0..=100 {
            let t = i as f64 / 100.0;
            let (r, g, b) = heatmap_color(t);
            assert!((0.0..=1.0).contains(&r), "r={r} out of range at t={t}");
            assert!((0.0..=1.0).contains(&g), "g={g} out of range at t={t}");
            assert!((0.0..=1.0).contains(&b), "b={b} out of range at t={t}");
        }
    }

    #[test]
    fn test_heatmap_color_red_channel_monotonic() {
        let mut prev_r = 0.0f64;
        for i in 25..=100 {
            let t = i as f64 / 100.0;
            let (r, _, _) = heatmap_color(t);
            assert!(r >= prev_r - 1e-10, "red not monotonic at t={t}: {r} < {prev_r}");
            prev_r = r;
        }
    }

    #[test]
    fn test_heatmap_color_clamps_out_of_range() {
        let neg = heatmap_color(-1.0);
        let zero = heatmap_color(0.0);
        assert_eq!(neg.0, zero.0);
        assert_eq!(neg.1, zero.1);
        assert_eq!(neg.2, zero.2);

        let over = heatmap_color(2.0);
        let one = heatmap_color(1.0);
        assert_eq!(over.0, one.0);
        assert_eq!(over.1, one.1);
        assert_eq!(over.2, one.2);
    }

    #[test]
    fn test_heatmap_nonblack_for_sub_unit_energy() {
        let mut spd_data: Vec<[f64; NUM_BINS]> = vec![[0.0; NUM_BINS]; 4];
        spd_data[0][3] = 0.05;
        spd_data[0][4] = 0.02;
        spd_data[1][10] = 0.3;
        spd_data[2][0] = 0.001;
        // spd_data[3] stays all-zero (background pixel)

        let tmp = std::env::temp_dir().join("cosmic_sig_test_heatmap");
        let _ = std::fs::create_dir_all(&tmp);
        let dir = tmp.to_str().unwrap();

        render_dominant_wavelength_map(&spd_data, 2, 2, dir).unwrap();

        let path = format!("{}/heatmap.png", dir);
        let img = image::open(&path).expect("heatmap PNG should exist").to_rgb16();
        let non_black = img.pixels().any(|px| px.0[0] > 0 || px.0[1] > 0 || px.0[2] > 0);
        assert!(non_black, "heatmap must contain non-black pixels for sub-1.0 energy data");

        let bg = img.get_pixel(1, 1);
        assert_eq!(bg.0, [0, 0, 0], "zero-energy pixel should be black");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_heatmap_brightness_monotonic() {
        let max_total = 0.5f64;
        let log_max = (1.0 + max_total).ln();

        let mut prev = 0.0f64;
        for i in 0..=100 {
            let total = max_total * (i as f64 / 100.0);
            let brightness = ((1.0 + total).ln() / log_max).clamp(0.0, 1.0);
            assert!(
                brightness >= prev - 1e-15,
                "brightness not monotonic at total={total}: {brightness} < {prev}"
            );
            prev = brightness;
        }
        let full = ((1.0 + max_total).ln() / log_max).clamp(0.0, 1.0);
        assert!((full - 1.0).abs() < 1e-10, "brightness at max_total should be 1.0, got {full}");
    }
}
