//! Cinematic zoom — a slow Ken Burns push into the densest region of the
//! orbital pattern, revealing fractal-like micro-detail invisible at
//! normal viewing distance.

use crate::render::{
    ChannelLevels, SpectralRenderSettings, SpectralScene, VideoEncodingOptions,
    constants, create_video_from_frames_singlepass, render_final_frame_spectral,
};
use tracing::info;

const ZOOM_SECONDS: f64 = 10.0;
const MAX_ZOOM: f64 = 5.0;

pub fn render_cinematic_zoom(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    output_path: &str,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
    info!("Rendering cinematic zoom video ({:.0}s, {:.0}x zoom)...", ZOOM_SECONDS, MAX_ZOOM);

    let resolved = settings.resolved_config;
    let width = resolved.width;
    let height = resolved.height;

    let hi_res_scale = 4u32;
    let hi_w = width * hi_res_scale;
    let hi_h = height * hi_res_scale;

    let mut hi_resolved = settings.resolved_config.clone();
    hi_resolved.width = hi_w;
    hi_resolved.height = hi_h;

    let hi_settings = SpectralRenderSettings::new(
        &hi_resolved,
        settings.render_config,
        settings.aspect_correction,
    );

    info!("   Rendering {}x{} base frame for zoom...", hi_w, hi_h);
    let hi_image = render_final_frame_spectral(scene, levels, hi_settings)?;

    let (hotspot_x, hotspot_y) = find_energy_hotspot(&hi_image, hi_w, hi_h);
    info!("   Energy hotspot at ({}, {})", hotspot_x, hotspot_y);

    let fps = constants::DEFAULT_VIDEO_FPS;
    let total_frames = (ZOOM_SECONDS * fps as f64) as usize;

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
            for frame_idx in 0..total_frames {
                let t = frame_idx as f64 / (total_frames - 1).max(1) as f64;
                let eased = t * t * (3.0 - 2.0 * t);
                let zoom = 1.0 + (MAX_ZOOM - 1.0) * eased;

                let crop_w = (hi_w as f64 / zoom) as u32;
                let crop_h = (hi_h as f64 / zoom) as u32;
                let crop_x = hotspot_x.saturating_sub(crop_w / 2).min(hi_w - crop_w);
                let crop_y = hotspot_y.saturating_sub(crop_h / 2).min(hi_h - crop_h);

                let frame = crop_and_resize(
                    &hi_image, hi_w, hi_h,
                    crop_x, crop_y, crop_w, crop_h,
                    width, height,
                );

                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        frame.as_ptr() as *const u8,
                        frame.len() * 2,
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

    info!("   Saved cinematic zoom => {}", output_path);
    Ok(())
}

fn find_energy_hotspot(
    image: &image::ImageBuffer<image::Rgb<u16>, Vec<u16>>,
    w: u32,
    h: u32,
) -> (u32, u32) {
    let block = 64u32;
    let bw = (w / block).max(1);
    let bh = (h / block).max(1);
    let mut best_energy = 0.0f64;
    let mut best_bx = bw / 2;
    let mut best_by = bh / 2;

    for by in 0..bh {
        for bx in 0..bw {
            let mut energy = 0.0f64;
            let x0 = bx * block;
            let y0 = by * block;
            for dy in (0..block.min(h - y0)).step_by(4) {
                for dx in (0..block.min(w - x0)).step_by(4) {
                    let px = image.get_pixel(x0 + dx, y0 + dy);
                    let lum = px[0] as f64 * 0.2126 + px[1] as f64 * 0.7152 + px[2] as f64 * 0.0722;
                    energy += lum;
                }
            }
            if energy > best_energy {
                best_energy = energy;
                best_bx = bx;
                best_by = by;
            }
        }
    }

    (best_bx * block + block / 2, best_by * block + block / 2)
}

#[allow(clippy::too_many_arguments)]
fn crop_and_resize(
    src: &image::ImageBuffer<image::Rgb<u16>, Vec<u16>>,
    _src_w: u32,
    _src_h: u32,
    cx: u32,
    cy: u32,
    cw: u32,
    ch: u32,
    dst_w: u32,
    dst_h: u32,
) -> Vec<u16> {
    let mut out = vec![0u16; (dst_w * dst_h * 3) as usize];

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = cx + (dx as f64 / dst_w as f64 * cw as f64) as u32;
            let sy = cy + (dy as f64 / dst_h as f64 * ch as f64) as u32;
            let sx = sx.min(src.width() - 1);
            let sy = sy.min(src.height() - 1);
            let px = src.get_pixel(sx, sy);
            let idx = ((dy * dst_w + dx) * 3) as usize;
            out[idx] = px[0];
            out[idx + 1] = px[1];
            out[idx + 2] = px[2];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn test_find_energy_hotspot_bright_corner() {
        let mut img: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::new(128, 128);
        for y in 96..128 {
            for x in 96..128 {
                img.put_pixel(x, y, Rgb([65535, 65535, 65535]));
            }
        }
        let (hx, hy) = find_energy_hotspot(&img, 128, 128);
        assert!(hx >= 64, "hotspot X should be in right half, got {}", hx);
        assert!(hy >= 64, "hotspot Y should be in bottom half, got {}", hy);
    }

    #[test]
    fn test_crop_and_resize_dimensions() {
        let img: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::from_fn(64, 64, |x, y| {
            Rgb([(x * 1000) as u16, (y * 1000) as u16, 0])
        });
        let out = crop_and_resize(&img, 64, 64, 10, 10, 32, 32, 16, 16);
        assert_eq!(out.len(), 16 * 16 * 3);
    }
}
