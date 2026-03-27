//! Gallery variants — renders the same orbit with alternative tonemapping
//! for light-background display (the "white cube" gallery aesthetic).

use crate::render::{
    ChannelLevels, SpectralRenderSettings, SpectralScene, render_final_frame_spectral,
    save_image_as_png_16bit,
};
use image::{ImageBuffer, Rgb};
use tracing::info;

pub fn render_light_variant(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    output_path: &str,
) -> crate::render::error::Result<()> {
    info!("Rendering light gallery variant...");

    let image = render_final_frame_spectral(scene, levels, settings)?;
    let light = invert_to_light_background(&image);
    save_image_as_png_16bit(&light, output_path)?;

    info!("   Saved light variant => {}", output_path);
    Ok(())
}

fn invert_to_light_background(
    image: &ImageBuffer<Rgb<u16>, Vec<u16>>,
) -> ImageBuffer<Rgb<u16>, Vec<u16>> {
    let (w, h) = image.dimensions();
    let mut output = ImageBuffer::new(w, h);

    let bg_r = 63000u16;
    let bg_g = 62000u16;
    let bg_b = 61000u16;

    for y in 0..h {
        for x in 0..w {
            let px = image.get_pixel(x, y);
            let r = px[0] as f64 / 65535.0;
            let g = px[1] as f64 / 65535.0;
            let b = px[2] as f64 / 65535.0;

            let luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let trail_alpha = luminance.clamp(0.0, 1.0);
            let boost = 1.0 + trail_alpha * 0.4;

            let tr = (r * boost).clamp(0.0, 1.0);
            let tg = (g * boost).clamp(0.0, 1.0);
            let tb = (b * boost).clamp(0.0, 1.0);

            let sat_boost = 1.3;
            let avg = (tr + tg + tb) / 3.0;
            let sr = (avg + (tr - avg) * sat_boost).clamp(0.0, 1.0);
            let sg = (avg + (tg - avg) * sat_boost).clamp(0.0, 1.0);
            let sb = (avg + (tb - avg) * sat_boost).clamp(0.0, 1.0);

            let alpha = trail_alpha.powf(0.7);
            let inv_alpha = 1.0 - alpha;

            let bg_norm_r = bg_r as f64 / 65535.0;
            let bg_norm_g = bg_g as f64 / 65535.0;
            let bg_norm_b = bg_b as f64 / 65535.0;

            let out_r = sr * alpha + bg_norm_r * inv_alpha;
            let out_g = sg * alpha + bg_norm_g * inv_alpha;
            let out_b = sb * alpha + bg_norm_b * inv_alpha;

            let dx = (x as f64 / w as f64) - 0.5;
            let dy = (y as f64 / h as f64) - 0.5;
            let dist = (dx * dx + dy * dy).sqrt() * 2.0;
            let vignette = 1.0 - 0.08 * dist * dist;

            output.put_pixel(x, y, Rgb([
                ((out_r * vignette).clamp(0.0, 1.0) * 65535.0) as u16,
                ((out_g * vignette).clamp(0.0, 1.0) * 65535.0) as u16,
                ((out_b * vignette).clamp(0.0, 1.0) * 65535.0) as u16,
            ]));
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invert_black_image_produces_light_background() {
        let img: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::from_fn(4, 4, |_, _| Rgb([0u16, 0, 0]));
        let result = invert_to_light_background(&img);
        let center = result.get_pixel(2, 2);
        assert!(center[0] > 50000, "background should be light, got R={}", center[0]);
    }

    #[test]
    fn test_invert_preserves_dimensions() {
        let img: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::new(16, 8);
        let result = invert_to_light_background(&img);
        assert_eq!(result.dimensions(), (16, 8));
    }

    #[test]
    fn test_invert_bright_pixels_remain_visible() {
        let img: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::from_fn(4, 4, |_, _| Rgb([60000u16, 20000, 40000]));
        let result = invert_to_light_background(&img);
        let px = result.get_pixel(2, 2);
        assert!(px[0] > 0);
        assert!(px[0] < 65535);
    }
}
