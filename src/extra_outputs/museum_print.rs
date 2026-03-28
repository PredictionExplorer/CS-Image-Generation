//! Museum-grade print files — ultra-high-resolution renders in multiple
//! aspect ratios suitable for large-format gallery printing at 300+ DPI.

use crate::render::{
    ChannelLevels, SpectralRenderSettings, SpectralScene, render_final_frame_spectral,
    save_image_as_png_16bit,
};
use std::fs;
use tracing::info;

struct PrintTarget {
    name: &'static str,
    width: u32,
    height: u32,
}

const TARGETS: &[PrintTarget] = &[
    PrintTarget { name: "8k_landscape", width: 7680, height: 4320 },
    PrintTarget { name: "square", width: 4320, height: 4320 },
    PrintTarget { name: "portrait", width: 4320, height: 7680 },
];

pub fn render_museum_prints(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    output_dir: &str,
) -> crate::render::error::Result<()> {
    info!("Rendering museum-grade prints ({} formats)...", TARGETS.len());

    fs::create_dir_all(output_dir).map_err(|e| {
        crate::render::error::RenderError::InvalidConfig(format!(
            "Failed to create museum print directory: {}", e
        ))
    })?;

    for target in TARGETS {
        info!(
            "   Rendering {}x{} ({}) — this may take several minutes...",
            target.width, target.height, target.name
        );

        let mut target_resolved = settings.resolved_config.clone();
        target_resolved.width = target.width;
        target_resolved.height = target.height;

        let target_settings = SpectralRenderSettings::new(
            &target_resolved,
            settings.render_config,
            settings.aspect_correction,
        );

        let image = render_final_frame_spectral(scene, levels, target_settings)?;
        let path = format!("{}/{}.png", output_dir, target.name);
        save_image_as_png_16bit(&image, &path)?;
        info!("   Saved {} => {}", target.name, path);
    }

    info!("   Museum prints complete => {}/", output_dir);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_count() {
        assert_eq!(TARGETS.len(), 3);
    }

    #[test]
    fn test_8k_dimensions() {
        let t = &TARGETS[0];
        assert_eq!(t.width, 7680);
        assert_eq!(t.height, 4320);
    }

    #[test]
    fn test_square_is_square() {
        let t = &TARGETS[1];
        assert_eq!(t.width, t.height);
    }

    #[test]
    fn test_portrait_is_taller_than_wide() {
        let t = &TARGETS[2];
        assert!(t.height > t.width);
    }
}
