//! Spectral decomposition video — starts with the full artwork, then peels
//! away wavelength groups from red to violet, revealing what each band hides.

use crate::extra_outputs::spectral_video_utils::{BinBuffers, NUM_GROUPS, BINS_PER_GROUP};
use crate::render::context::RenderContext;
use crate::render::velocity_hdr;
use crate::render::{
    SpectralRenderSettings, SpectralScene, VideoEncodingOptions,
    accumulate_spectral_steps, constants, create_video_from_frames_singlepass,
    default_accumulation_backend,
};
use crate::spectrum::NUM_BINS;
use tracing::info;

const DECOMP_DURATION_SECONDS: f64 = 15.0;
const HOLD_SECONDS: f64 = 2.0;
const FADE_OUT_SECONDS: f64 = 1.0;
const WAVE_FADE_SECONDS: f64 = 0.5;

#[inline]
fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn weights_for_decomp_t(
    weights: &mut [f64; NUM_BINS],
    decomp_t: f64,
    active_duration: f64,
    wave_interval: f64,
) {
    let decomp_t = decomp_t.clamp(0.0, active_duration);
    let mut current_removing_group = (decomp_t / wave_interval).floor() as usize;
    if current_removing_group >= NUM_GROUPS {
        current_removing_group = NUM_GROUPS - 1;
    }
    let reverse_group = NUM_GROUPS - 1 - current_removing_group;
    let group_start = current_removing_group as f64 * wave_interval;

    for bin in 0..NUM_BINS {
        let g = bin / BINS_PER_GROUP;
        weights[bin] = if g > reverse_group {
            0.0
        } else if g < reverse_group {
            1.0
        } else {
            let linear = ((decomp_t - group_start) / WAVE_FADE_SECONDS).clamp(0.0, 1.0);
            1.0 - smoothstep(linear)
        };
    }
}

pub fn render_spectral_decomposition_video(
    scene: SpectralScene<'_>,
    settings: SpectralRenderSettings<'_>,
    output_path: &str,
    fast_encode: bool,
) -> crate::render::error::Result<()> {
    info!(
        "Rendering spectral decomposition video ({:.0}s, {} groups × {} bins)...",
        DECOMP_DURATION_SECONDS,
        NUM_GROUPS,
        BINS_PER_GROUP
    );

    let resolved = settings.resolved_config;
    let width = resolved.width;
    let height = resolved.height;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let total_frames = (DECOMP_DURATION_SECONDS * fps as f64) as usize;

    let decomp_end = DECOMP_DURATION_SECONDS - FADE_OUT_SECONDS;
    let active_duration = DECOMP_DURATION_SECONDS - HOLD_SECONDS - FADE_OUT_SECONDS;
    let wave_interval = active_duration / NUM_GROUPS as f64;

    let ctx = RenderContext::new(width, height, scene.positions, settings.aspect_correction);
    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);

    let mut accum_spd = vec![[0.0f64; NUM_BINS]; ctx.pixel_count()];
    let total_steps = scene.step_count();
    accumulate_spectral_steps(
        &mut accum_spd,
        scene,
        &ctx,
        &velocity_calc,
        0,
        total_steps,
        settings.render_config.hdr_scale,
        default_accumulation_backend(),
    );

    let bins = BinBuffers::from_spd(accum_spd, width, height);

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
            let mut frame_buf = Vec::<u16>::new();
            let mut weights = [0.0f64; NUM_BINS];

            for frame in 0..total_frames {
                let t = frame as f64 / fps as f64;

                if t < HOLD_SECONDS {
                    weights.fill(1.0);
                } else if t >= decomp_end {
                    weights_for_decomp_t(
                        &mut weights,
                        active_duration,
                        active_duration,
                        wave_interval,
                    );
                    let fade_elapsed = t - decomp_end;
                    let fade_linear = (fade_elapsed / FADE_OUT_SECONDS).clamp(0.0, 1.0);
                    let fade_factor = 1.0 - smoothstep(fade_linear);
                    for w in &mut weights {
                        *w *= fade_factor;
                    }
                } else {
                    let decomp_t = t - HOLD_SECONDS;
                    weights_for_decomp_t(&mut weights, decomp_t, active_duration, wave_interval);
                }

                bins.weighted_blend(&weights, &mut frame_buf);
                let bytes = crate::utils::u16_slice_as_bytes(&frame_buf);
                out.write_all(bytes)
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            }
            Ok::<(), Box<dyn std::error::Error>>(())
        },
        output_path,
        &options,
    )?;

    info!("   Saved spectral decomposition video => {}", output_path);
    Ok(())
}
