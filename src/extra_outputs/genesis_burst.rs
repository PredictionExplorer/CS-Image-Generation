//! "Genesis Burst" still — the first ~1% of the trajectory, capturing the
//! explosive initial divergence of the three bodies.

use crate::render::context::RenderContext;
use crate::render::effects::{FinishEffectPipeline, FrameParams, convert_spd_buffer_to_rgba};
use crate::render::velocity_hdr;
use crate::render::{
    ChannelLevels, FinishOutputMode, SpectralRenderSettings, SpectralScene,
    accumulate_spectral_steps, apply_energy_density_shift, build_effect_config_from_resolved,
    constants, default_accumulation_backend,
    quantize_display_buffer_to_16bit, save_image_as_png_16bit, tonemap_to_display_buffer,
};
use crate::spectrum::NUM_BINS;
use image::ImageBuffer;
use tracing::info;

const GENESIS_FRACTION: f64 = 0.01;

pub fn render_genesis_burst(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    output_path: &str,
) -> crate::render::error::Result<()> {
    info!("Rendering genesis burst still (first 1% of trajectory)...");

    let resolved = settings.resolved_config;
    let width = resolved.width;
    let height = resolved.height;
    let ctx = RenderContext::new(width, height, scene.positions, settings.aspect_correction);
    let mut accum_spd = vec![[0.0f64; NUM_BINS]; ctx.pixel_count()];
    let mut accum_rgba = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];

    let effect_config =
        build_effect_config_from_resolved(resolved, settings.render_config, FinishOutputMode::Still);
    let finish_pipeline = FinishEffectPipeline::new(effect_config);

    let total_steps = scene.step_count();
    let genesis_steps = ((total_steps as f64 * GENESIS_FRACTION) as usize).max(2);
    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);

    accumulate_spectral_steps(
        &mut accum_spd,
        scene,
        &ctx,
        &velocity_calc,
        0,
        genesis_steps,
        settings.render_config.hdr_scale,
        default_accumulation_backend(),
    );

    apply_energy_density_shift(&mut accum_spd);
    convert_spd_buffer_to_rgba(&accum_spd, &mut accum_rgba, width as usize, height as usize);

    let frame_params = FrameParams { frame_number: 0, density: None };
    let trajectory_pixels = finish_pipeline
        .process_trajectory(accum_rgba, width as usize, height as usize, &frame_params)
        .map_err(|e| crate::render::error::RenderError::EffectChain(e.to_string()))?;

    let display_buffer = tonemap_to_display_buffer(&trajectory_pixels, levels);
    let final_display = finish_pipeline
        .process_image(display_buffer, width as usize, height as usize, &frame_params)
        .map_err(|e| crate::render::error::RenderError::EffectChain(e.to_string()))?;
    let buf_16bit = quantize_display_buffer_to_16bit(&final_display);

    let image = ImageBuffer::from_raw(width, height, buf_16bit).ok_or_else(|| {
        crate::render::error::RenderError::ImageEncoding(
            "Failed to create genesis burst image".to_string(),
        )
    })?;

    save_image_as_png_16bit(&image, output_path)?;
    info!("   Saved genesis burst => {}", output_path);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{BloomMode, RenderConfig, SpectralRenderSettings, SpectralScene};
    use crate::render::randomizable_config::ResolvedEffectConfig;
    use nalgebra::Vector3;

    fn minimal_resolved_config(w: u32, h: u32) -> ResolvedEffectConfig {
        ResolvedEffectConfig {
            width: w,
            height: h,
            enable_bloom: false, enable_glow: false, enable_chromatic_bloom: false,
            enable_perceptual_blur: false, enable_micro_contrast: false,
            enable_gradient_map: false, enable_color_grade: false,
            enable_champleve: false, enable_aether: false, enable_opalescence: false,
            enable_edge_luminance: false, enable_atmospheric_depth: false,
            enable_fine_texture: false,
            blur_strength: 0.0, blur_radius_scale: 0.0, blur_core_brightness: 0.0,
            dog_strength: 0.3, dog_sigma_scale: 0.005, dog_ratio: 2.6,
            glow_strength: 0.0, glow_threshold: 0.0, glow_radius_scale: 0.0,
            glow_sharpness: 0.0, glow_saturation_boost: 0.0,
            chromatic_bloom_strength: 0.0, chromatic_bloom_radius_scale: 0.0,
            chromatic_bloom_separation_scale: 0.0, chromatic_bloom_threshold: 0.0,
            perceptual_blur_strength: 0.0, color_grade_strength: 0.0,
            vignette_strength: 0.0, vignette_softness: 0.0, vibrance: 0.0,
            clarity_strength: 0.0, tone_curve_strength: 0.0,
            gradient_map_strength: 0.0, gradient_map_hue_preservation: 0.0, gradient_map_palette: 0,
            opalescence_strength: 0.0, opalescence_scale: 0.0, opalescence_layers: 0,
            champleve_flow_alignment: 0.0, champleve_interference_amplitude: 0.0,
            champleve_rim_intensity: 0.0, champleve_rim_warmth: 0.0, champleve_interior_lift: 0.0,
            aether_flow_alignment: 0.0, aether_scattering_strength: 0.0,
            aether_iridescence_amplitude: 0.0, aether_caustic_strength: 0.0,
            micro_contrast_strength: 0.0, micro_contrast_radius: 0,
            edge_luminance_strength: 0.0, edge_luminance_threshold: 0.0,
            edge_luminance_brightness_boost: 0.0,
            atmospheric_depth_strength: 0.0, atmospheric_desaturation: 0.0,
            atmospheric_darkening: 0.0,
            atmospheric_fog_color_r: 0.0, atmospheric_fog_color_g: 0.0, atmospheric_fog_color_b: 0.0,
            fine_texture_strength: 0.0, fine_texture_scale: 0.0, fine_texture_contrast: 0.0,
            hdr_scale: 0.12, clip_black: 0.01, clip_white: 0.99,
        }
    }

    #[test]
    fn test_genesis_burst_produces_nonzero_image() {
        let positions = vec![
            vec![
                Vector3::new(0.10, 0.10, 0.0),
                Vector3::new(0.16, 0.14, 0.0),
                Vector3::new(0.24, 0.22, 0.0),
                Vector3::new(0.32, 0.28, 0.0),
                Vector3::new(0.38, 0.32, 0.0),
            ],
            vec![
                Vector3::new(0.86, 0.12, 0.0),
                Vector3::new(0.80, 0.18, 0.0),
                Vector3::new(0.72, 0.26, 0.0),
                Vector3::new(0.64, 0.34, 0.0),
                Vector3::new(0.58, 0.42, 0.0),
            ],
            vec![
                Vector3::new(0.45, 0.88, 0.0),
                Vector3::new(0.48, 0.80, 0.0),
                Vector3::new(0.52, 0.72, 0.0),
                Vector3::new(0.56, 0.64, 0.0),
                Vector3::new(0.60, 0.56, 0.0),
            ],
        ];
        let colors = vec![
            vec![(0.72, 0.22, 0.08); 5],
            vec![(0.70, -0.18, 0.18); 5],
            vec![(0.68, 0.04, -0.20); 5],
        ];
        let body_alphas = vec![0.65, 0.85, 0.95];

        let w = 32u32;
        let h = 18u32;
        let resolved = minimal_resolved_config(w, h);
        let render_config = RenderConfig { hdr_scale: resolved.hdr_scale, bloom_mode: BloomMode::None };
        let levels = crate::render::ChannelLevels::new(0.0, 0.1, 0.0, 0.1, 0.0, 0.1);
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let settings = SpectralRenderSettings::new(&resolved, &render_config, false);

        let tmp_dir = std::env::temp_dir().join("cosmic_sig_test_genesis");
        let _ = std::fs::create_dir_all(&tmp_dir);
        let output = tmp_dir.join("test_genesis.png");
        let output_str = output.to_str().unwrap();

        let result = render_genesis_burst(scene, &levels, settings, output_str);
        assert!(result.is_ok(), "genesis burst should succeed: {:?}", result.err());

        let img = image::open(output_str).expect("should read the output PNG");
        let energy: u64 = img.to_rgb16().as_raw().iter().map(|&v| v as u64).sum();
        assert!(energy > 0, "genesis burst image should have non-zero energy");
        assert_eq!(img.width(), w);
        assert_eq!(img.height(), h);

        let _ = std::fs::remove_file(output_str);
    }
}
