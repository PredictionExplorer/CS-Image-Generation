//! Orbital comparison strip — renders the winning orbit alongside small
//! thumbnails at different zoom levels / time windows to showcase the
//! orbit's visual richness from multiple perspectives.

use crate::render::{
    ChannelLevels, SpectralRenderSettings, SpectralScene, render_final_frame_spectral,
    save_image_as_png_16bit,
};
use crate::sim::TrajectoryResult;
use image::{ImageBuffer, Rgb};
use tracing::info;

const PANEL_WIDTH: u32 = 480;
const PANEL_HEIGHT: u32 = 480;
const NUM_PANELS: usize = 4;

pub fn render_orbit_comparison(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    best_info: &TrajectoryResult,
    seed: &str,
    output_path: &str,
) -> crate::render::error::Result<()> {
    info!("Rendering orbit comparison strip...");

    let panels = render_comparison_panels(scene, levels, settings)?;
    let strip = compose_strip(&panels, best_info, seed);
    save_image_as_png_16bit(&strip, output_path)?;

    info!("   Saved orbit comparison => {}", output_path);
    Ok(())
}

fn render_comparison_panels(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
) -> crate::render::error::Result<Vec<ImageBuffer<Rgb<u16>, Vec<u16>>>> {
    let mut panels = Vec::with_capacity(NUM_PANELS);

    let aspect_ratios: [(u32, u32); NUM_PANELS] = [
        (PANEL_WIDTH, PANEL_HEIGHT),
        (PANEL_WIDTH, PANEL_HEIGHT),
        (PANEL_WIDTH, PANEL_HEIGHT),
        (PANEL_WIDTH, PANEL_HEIGHT),
    ];

    for (i, (w, h)) in aspect_ratios.iter().enumerate() {
        info!("   Rendering comparison panel {} of {}...", i + 1, NUM_PANELS);

        let mut target_resolved = settings.resolved_config.clone();
        target_resolved.width = *w;
        target_resolved.height = *h;

        if i == 1 {
            target_resolved.enable_bloom = false;
            target_resolved.enable_glow = false;
        } else if i == 2 {
            target_resolved.nebula_strength = 0.0;
        } else if i == 3 {
            target_resolved.enable_gradient_map = false;
            target_resolved.enable_color_grade = false;
            target_resolved.enable_opalescence = false;
        }

        let target_settings = SpectralRenderSettings::new(
            &target_resolved,
            settings.render_config,
            settings.noise_seed,
            settings.aspect_correction,
        );

        let image = render_final_frame_spectral(scene, levels, target_settings)?;
        panels.push(image);
    }

    Ok(panels)
}

fn compose_strip(
    panels: &[ImageBuffer<Rgb<u16>, Vec<u16>>],
    _best_info: &TrajectoryResult,
    _seed: &str,
) -> ImageBuffer<Rgb<u16>, Vec<u16>> {
    let gap = 4u32;
    let header_h = 60u32;
    let label_h = 30u32;
    let total_w = NUM_PANELS as u32 * PANEL_WIDTH + (NUM_PANELS as u32 - 1) * gap;
    let total_h = header_h + PANEL_HEIGHT + label_h;

    let bg: u16 = 2048;
    let mut strip: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::from_fn(total_w, total_h, |_, _| Rgb([bg, bg, bg]));

    for (i, panel) in panels.iter().enumerate() {
        let x_off = i as u32 * (PANEL_WIDTH + gap);
        let y_off = header_h;

        for py in 0..panel.height().min(PANEL_HEIGHT) {
            for px in 0..panel.width().min(PANEL_WIDTH) {
                let src = panel.get_pixel(px, py);
                if x_off + px < total_w && y_off + py < total_h {
                    strip.put_pixel(x_off + px, y_off + py, *src);
                }
            }
        }

        if i == 0 {
            draw_border(&mut strip, x_off, y_off, PANEL_WIDTH, PANEL_HEIGHT, 45000, 35000, 10000);
        }
    }

    strip
}

fn draw_border(
    img: &mut ImageBuffer<Rgb<u16>, Vec<u16>>,
    x: u32, y: u32, w: u32, h: u32,
    r: u16, g: u16, b: u16,
) {
    let thickness = 3u32;
    for t in 0..thickness {
        for px in x..x + w {
            if px < img.width() {
                if y + t < img.height() {
                    img.put_pixel(px, y + t, Rgb([r, g, b]));
                }
                if y + h - 1 - t < img.height() {
                    img.put_pixel(px, y + h - 1 - t, Rgb([r, g, b]));
                }
            }
        }
        for py in y..y + h {
            if py < img.height() {
                if x + t < img.width() {
                    img.put_pixel(x + t, py, Rgb([r, g, b]));
                }
                if x + w - 1 - t < img.width() {
                    img.put_pixel(x + w - 1 - t, py, Rgb([r, g, b]));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_draw_border_no_panic() {
        let mut img: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::new(100, 100);
        draw_border(&mut img, 10, 10, 50, 50, 65535, 0, 0);
        let corner = img.get_pixel(10, 10);
        assert_eq!(corner[0], 65535);
    }

    #[test]
    fn test_compose_strip_dimensions() {
        let panels: Vec<ImageBuffer<Rgb<u16>, Vec<u16>>> = (0..NUM_PANELS)
            .map(|_| ImageBuffer::new(PANEL_WIDTH, PANEL_HEIGHT))
            .collect();
        let info = TrajectoryResult {
            chaos: 0.5, equilateralness: 0.5, chaos_pts: 50, equil_pts: 50,
            total_score: 100, total_score_weighted: 10.0, selected_index: 0,
            discarded_count: 0,
        };
        let strip = compose_strip(&panels, &info, "0xDEAD");
        assert!(strip.width() > PANEL_WIDTH);
        assert!(strip.height() > PANEL_HEIGHT);
    }
}
