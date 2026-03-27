//! Wallpaper pack — renders the final accumulated frame at multiple
//! common device resolutions from a single simulation.

use crate::render::{
    ChannelLevels, SpectralRenderSettings, SpectralScene, render_final_frame_spectral,
    save_image_as_png_16bit,
};
use std::fs;
use tracing::info;

struct WallpaperTarget {
    name: &'static str,
    width: u32,
    height: u32,
}

const TARGETS: &[WallpaperTarget] = &[
    WallpaperTarget { name: "4k", width: 3840, height: 2160 },
    WallpaperTarget { name: "ultrawide", width: 3440, height: 1440 },
    WallpaperTarget { name: "iphone", width: 1170, height: 2532 },
    WallpaperTarget { name: "ipad", width: 2048, height: 2732 },
    WallpaperTarget { name: "watch", width: 396, height: 484 },
];

pub fn render_wallpaper_pack(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    output_dir: &str,
    name: &str,
) -> crate::render::error::Result<()> {
    info!("Rendering wallpaper pack ({} resolutions)...", TARGETS.len());

    fs::create_dir_all(output_dir).map_err(|e| {
        crate::render::error::RenderError::InvalidConfig(format!(
            "Failed to create wallpaper directory: {}",
            e
        ))
    })?;

    for target in TARGETS {
        info!("   Rendering {}x{} ({})...", target.width, target.height, target.name);

        let mut target_resolved = settings.resolved_config.clone();
        target_resolved.width = target.width;
        target_resolved.height = target.height;

        let target_settings = SpectralRenderSettings::new(
            &target_resolved,
            settings.render_config,
            settings.noise_seed,
            settings.aspect_correction,
        );

        let image = render_final_frame_spectral(scene, levels, target_settings)?;
        let path = format!("{}/{}_{}.png", output_dir, name, target.name);
        save_image_as_png_16bit(&image, &path)?;
    }

    info!("   Saved {} wallpapers => {}/", TARGETS.len(), output_dir);
    Ok(())
}
