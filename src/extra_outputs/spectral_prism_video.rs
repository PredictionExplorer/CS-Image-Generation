//! Prism sweep video — a diagonal rainbow band sweeps across the image,
//! each column within the band showing a different spectral bin.

use crate::extra_outputs::spectral_video_utils::BinBuffers;
use crate::render::context::RenderContext;
use crate::render::velocity_hdr;
use crate::render::{
    SpectralRenderSettings, SpectralScene, VideoEncodingOptions,
    accumulate_spectral_steps, constants, create_video_from_frames_singlepass,
    default_accumulation_backend,
};
use crate::spectrum::NUM_BINS;
use rayon::prelude::*;
use tracing::info;

const PRISM_DURATION_SECONDS: f64 = 8.0;
const BAND_WIDTH_FRACTION: f64 = 0.3;
const BLEND_ZONE: f64 = 0.05;

#[inline]
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    if edge1 <= edge0 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub fn render_spectral_prism_video(
    scene: SpectralScene<'_>,
    settings: SpectralRenderSettings<'_>,
    output_path: &str,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
    info!(
        "Rendering spectral prism sweep video ({:.0}s, diagonal band)...",
        PRISM_DURATION_SECONDS
    );

    let resolved = settings.resolved_config;
    let width = resolved.width;
    let height = resolved.height;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let total_frames = (PRISM_DURATION_SECONDS * fps as f64).max(1.0) as usize;

    let ctx = RenderContext::new(width, height, scene.positions, settings.aspect_correction);
    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);

    let mut accum_spd = vec![[0.0f64; NUM_BINS]; ctx.pixel_count()];
    let total_steps = scene.step_count();
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

    let bins = BinBuffers::from_spd(accum_spd, width, height);
    let mut composite_buf = Vec::<u16>::new();
    bins.full_composite(&mut composite_buf);

    let band_half = BAND_WIDTH_FRACTION / 2.0;
    let pixel_count = bins.pixel_count;
    let inv_w = 1.0 / width as f64;
    let inv_h = 1.0 / height as f64;

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
            let mut frame_buf = vec![0u16; pixel_count * 3];
            for frame in 0..total_frames {
                let sweep_center =
                    -band_half + (1.0 + BAND_WIDTH_FRACTION) * frame as f64 / total_frames as f64;

                frame_buf
                    .par_chunks_mut(3)
                    .enumerate()
                    .for_each(|(i, chunk)| {
                        let x = (i as u32 % width) as f64;
                        let y = (i as u32 / width) as f64;
                        let d = (x * inv_w + y * inv_h) / 2.0;
                        let dist = d - sweep_center;

                        let cr = composite_buf[i * 3] as f64 / 65535.0;
                        let cg = composite_buf[i * 3 + 1] as f64 / 65535.0;
                        let cb = composite_buf[i * 3 + 2] as f64 / 65535.0;

                        let left_in = smoothstep(-band_half - BLEND_ZONE, -band_half + BLEND_ZONE, dist);
                        let right_out = smoothstep(band_half - BLEND_ZONE, band_half + BLEND_ZONE, dist);
                        let prism_alpha = left_in * (1.0 - right_out);

                        let dist_clamped = dist.clamp(-band_half, band_half);
                        let bin_t = (dist_clamped + band_half) / BAND_WIDTH_FRACTION;
                        let bin_f = bin_t * (NUM_BINS - 1) as f64;
                        let bin_f = bin_f.rem_euclid(NUM_BINS as f64);
                        let lo = bin_f.floor() as usize % NUM_BINS;
                        let hi = (lo + 1) % NUM_BINS;
                        let t = bin_f.fract();

                        let pr = bins.buffers[lo][i][0] * (1.0 - t) + bins.buffers[hi][i][0] * t;
                        let pg = bins.buffers[lo][i][1] * (1.0 - t) + bins.buffers[hi][i][1] * t;
                        let pb = bins.buffers[lo][i][2] * (1.0 - t) + bins.buffers[hi][i][2] * t;

                        let r = pr * prism_alpha + cr * (1.0 - prism_alpha);
                        let g = pg * prism_alpha + cg * (1.0 - prism_alpha);
                        let b = pb * prism_alpha + cb * (1.0 - prism_alpha);

                        chunk[0] = (r.clamp(0.0, 1.0) * 65535.0).round() as u16;
                        chunk[1] = (g.clamp(0.0, 1.0) * 65535.0).round() as u16;
                        chunk[2] = (b.clamp(0.0, 1.0) * 65535.0).round() as u16;
                    });

                let bytes = crate::utils::u16_slice_as_bytes(&frame_buf);
                out.write_all(bytes)
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            }
            Ok::<(), Box<dyn std::error::Error>>(())
        },
        output_path,
        &options,
    )?;

    info!("   Saved spectral prism video => {}", output_path);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smoothstep_below_edge0_is_zero() {
        assert!((smoothstep(0.0, 1.0, -0.5) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_smoothstep_above_edge1_is_one() {
        assert!((smoothstep(0.0, 1.0, 1.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_smoothstep_midpoint_is_half() {
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_smoothstep_degenerate_equal_edges() {
        assert!((smoothstep(0.5, 0.5, 0.3) - 0.0).abs() < 1e-10);
        assert!((smoothstep(0.5, 0.5, 0.5) - 1.0).abs() < 1e-10);
        assert!((smoothstep(0.5, 0.5, 0.7) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_smoothstep_degenerate_reversed_edges() {
        assert!((smoothstep(1.0, 0.0, -0.1) - 0.0).abs() < 1e-10);
        assert!((smoothstep(1.0, 0.0, 1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_smoothstep_output_in_unit_range() {
        for i in -20..=120 {
            let x = i as f64 / 100.0;
            let val = smoothstep(0.0, 1.0, x);
            assert!((0.0..=1.0).contains(&val), "smoothstep(0,1,{x}) = {val} out of [0,1]");
        }
    }
}
