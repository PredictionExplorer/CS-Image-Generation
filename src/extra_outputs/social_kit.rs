//! Social media kit — pre-formatted crops of the final still for every
//! major platform, with optional branded overlay text.

use crate::render::{
    ChannelLevels, SpectralRenderSettings, SpectralScene, render_final_frame_spectral,
    save_image_as_png_16bit,
};
use std::fs;
use tracing::info;

struct SocialTarget {
    name: &'static str,
    width: u32,
    height: u32,
}

const TARGETS: &[SocialTarget] = &[
    SocialTarget { name: "twitter_header", width: 1500, height: 500 },
    SocialTarget { name: "instagram_square", width: 1080, height: 1080 },
    SocialTarget { name: "instagram_story", width: 1080, height: 1920 },
    SocialTarget { name: "discord_banner", width: 960, height: 540 },
    SocialTarget { name: "og_image", width: 1200, height: 630 },
];

pub fn generate_social_kit(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    output_dir: &str,
) -> crate::render::error::Result<()> {
    info!("Generating social media kit ({} formats)...", TARGETS.len());

    fs::create_dir_all(output_dir).map_err(|e| {
        crate::render::error::RenderError::InvalidConfig(format!(
            "Failed to create social kit directory: {}", e
        ))
    })?;

    for target in TARGETS {
        info!("   Rendering {} ({}x{})...", target.name, target.width, target.height);

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
    }

    info!("   Saved {} social assets => {}/", TARGETS.len(), output_dir);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_count() {
        assert_eq!(TARGETS.len(), 5);
    }

    #[test]
    fn test_all_targets_have_positive_dimensions() {
        for t in TARGETS {
            assert!(t.width > 0);
            assert!(t.height > 0);
        }
    }

    #[test]
    fn test_target_names_unique() {
        let mut names: Vec<&str> = TARGETS.iter().map(|t| t.name).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), TARGETS.len());
    }
}
