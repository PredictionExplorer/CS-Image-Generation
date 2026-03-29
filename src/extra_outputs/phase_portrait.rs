//! Phase space portrait — plots velocity vs. position to reveal the
//! mathematical structure of the orbit as an alternate artwork.

use crate::render::context::RenderContext;
use crate::render::effects::{FinishEffectPipeline, FrameParams, convert_spd_buffer_to_rgba};
use crate::render::velocity_hdr;
use crate::render::{
    ChannelLevels, FinishOutputMode, OklabColor, SpectralRenderSettings, SpectralScene,
    accumulate_spectral_steps, apply_energy_density_shift, build_effect_config_from_resolved,
    constants, default_accumulation_backend, quantize_display_buffer_to_16bit,
    save_image_as_png_16bit, tonemap_to_display_buffer,
};
use crate::spectrum::NUM_BINS;
use image::ImageBuffer;
use nalgebra::Vector3;
use tracing::info;

/// Build phase-space positions from spatial positions and velocities.
/// Maps (x, vx) for each body: position's x-component vs. velocity's x-component.
pub fn build_phase_positions(
    positions: &[Vec<Vector3<f64>>],
    velocities: &[Vec<Vector3<f64>>],
) -> Vec<Vec<Vector3<f64>>> {
    positions
        .iter()
        .zip(velocities.iter())
        .map(|(pos, vel)| {
            pos.iter()
                .zip(vel.iter())
                .map(|(p, v)| Vector3::new(p.x, v.x, p.z))
                .collect()
        })
        .collect()
}

pub fn render_phase_portrait(
    positions: &[Vec<Vector3<f64>>],
    velocities: &[Vec<Vector3<f64>>],
    colors: &[Vec<OklabColor>],
    body_alphas: &[f64],
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    output_path: &str,
) -> crate::render::error::Result<()> {
    info!("Rendering phase space portrait...");

    let phase_positions = build_phase_positions(positions, velocities);
    let scene = SpectralScene::new(&phase_positions, colors, body_alphas);

    let resolved = settings.resolved_config;
    let width = resolved.width;
    let height = resolved.height;
    let ctx = RenderContext::new(width, height, &phase_positions, settings.aspect_correction);
    let mut accum_spd = vec![[0.0f64; NUM_BINS]; ctx.pixel_count()];
    let mut accum_rgba = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];

    let effect_config =
        build_effect_config_from_resolved(resolved, settings.render_config, FinishOutputMode::Still);
    let finish_pipeline = FinishEffectPipeline::new(effect_config);

    let total_steps = scene.step_count();
    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(&phase_positions, dt);

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
            "Failed to create phase portrait image".to_string(),
        )
    })?;

    save_image_as_png_16bit(&image, output_path)?;
    info!("   Saved phase portrait => {}", output_path);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_phase_positions_length() {
        let positions = vec![
            vec![Vector3::new(1.0, 2.0, 3.0), Vector3::new(4.0, 5.0, 6.0)],
            vec![Vector3::new(7.0, 8.0, 9.0), Vector3::new(10.0, 11.0, 12.0)],
        ];
        let velocities = vec![
            vec![Vector3::new(0.1, 0.2, 0.3), Vector3::new(0.4, 0.5, 0.6)],
            vec![Vector3::new(0.7, 0.8, 0.9), Vector3::new(1.0, 1.1, 1.2)],
        ];

        let result = build_phase_positions(&positions, &velocities);
        assert_eq!(result.len(), positions.len());
        assert_eq!(result[0].len(), positions[0].len());
        assert_eq!(result[1].len(), positions[1].len());
    }

    #[test]
    fn test_build_phase_positions_x_equals_position_x() {
        let positions = vec![vec![Vector3::new(1.5, 2.5, 3.5)]];
        let velocities = vec![vec![Vector3::new(0.1, 0.2, 0.3)]];
        let result = build_phase_positions(&positions, &velocities);
        assert_eq!(result[0][0].x, 1.5);
    }

    #[test]
    fn test_build_phase_positions_y_equals_velocity_x() {
        let positions = vec![vec![Vector3::new(1.5, 2.5, 3.5)]];
        let velocities = vec![vec![Vector3::new(0.1, 0.2, 0.3)]];
        let result = build_phase_positions(&positions, &velocities);
        assert_eq!(result[0][0].y, 0.1);
    }

    #[test]
    fn test_build_phase_positions_z_equals_position_z() {
        let positions = vec![vec![Vector3::new(1.5, 2.5, 3.5)]];
        let velocities = vec![vec![Vector3::new(0.1, 0.2, 0.3)]];
        let result = build_phase_positions(&positions, &velocities);
        assert_eq!(result[0][0].z, 3.5);
    }

    #[test]
    fn test_build_phase_positions_multi_body_multi_step() {
        let positions = vec![
            vec![Vector3::new(1.0, 2.0, 3.0), Vector3::new(4.0, 5.0, 6.0)],
            vec![Vector3::new(7.0, 8.0, 9.0), Vector3::new(10.0, 11.0, 12.0)],
            vec![Vector3::new(13.0, 14.0, 15.0), Vector3::new(16.0, 17.0, 18.0)],
        ];
        let velocities = vec![
            vec![Vector3::new(-1.0, -2.0, -3.0), Vector3::new(-4.0, -5.0, -6.0)],
            vec![Vector3::new(-7.0, -8.0, -9.0), Vector3::new(-10.0, -11.0, -12.0)],
            vec![Vector3::new(-13.0, -14.0, -15.0), Vector3::new(-16.0, -17.0, -18.0)],
        ];
        let result = build_phase_positions(&positions, &velocities);

        assert_eq!(result[2][1].x, 16.0);
        assert_eq!(result[2][1].y, -16.0);
        assert_eq!(result[2][1].z, 18.0);
    }
}
