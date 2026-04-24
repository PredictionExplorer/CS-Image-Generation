//! Spectral sweep video generation.

use super::BinBuffers;
use crate::post_effects::{CinematicColorGrade, ColorGradeParams, GaussianBloom, PostEffect};
use crate::render::constants;
use crate::render::context::PixelBuffer;
use crate::render::error::{RenderError, Result};
use crate::render::video::{
    FfmpegVideoEncoder, VideoEncoder, VideoEncodingOptions,
    create_video_from_frames_singlepass_with_encoder,
};
use crate::spectrum::NUM_BINS;
use rayon::prelude::*;
use tracing::info;

/// Pre-computed, normalised Gaussian weights for a bin blend.
pub(super) struct BlendWeights {
    pub(super) lo: usize,
    pub(super) hi: usize,
    pub(super) weights: Vec<f32>,
}

impl BlendWeights {
    /// Compute normalised Gaussian weights centred at `center` with the given
    /// `sigma` (in bin-units).  Bins are clamped to `[0, NUM_BINS-1]`.
    pub(super) fn compute(center: f64, sigma: f64) -> Self {
        let radius = (3.0 * sigma).ceil() as isize;
        let center_floor = center.floor() as isize;
        let lo = (center_floor - radius).max(0) as usize;
        let hi = ((center_floor + radius + 1) as usize).min(NUM_BINS - 1);

        let inv_2sig2 = 0.5 / (sigma * sigma);
        let mut weights: Vec<f32> = Vec::with_capacity(hi - lo + 1);
        let mut total: f64 = 0.0;
        for b in lo..=hi {
            let d = b as f64 - center;
            let w = (-d * d * inv_2sig2).exp();
            weights.push(w as f32);
            total += w;
        }
        let inv_total = 1.0 / total as f32;
        for w in &mut weights {
            *w *= inv_total;
        }

        Self { lo, hi, weights }
    }
}

/// Blend multiple bin images with a Gaussian kernel centred at `center`,
/// producing a `PixelBuffer` (f64 RGBA, alpha = 1.0) suitable for post-effects.
pub(super) fn gaussian_blend_to_pixelbuffer(
    bin_buffers: &BinBuffers,
    center: f64,
    sigma: f64,
    output: &mut PixelBuffer,
) {
    let pixel_count = bin_buffers.pixel_count();
    output.resize(pixel_count, (0.0, 0.0, 0.0, 1.0));
    let bw = BlendWeights::compute(center, sigma);

    let bufs = &bin_buffers.buffers;
    output.par_iter_mut().enumerate().for_each(|(i, pixel)| {
        let (mut r, mut g, mut b_ch) = (0.0f32, 0.0f32, 0.0f32);
        for (j, bin) in (bw.lo..=bw.hi).enumerate() {
            let w = bw.weights[j];
            r += bufs[bin][i][0] * w;
            g += bufs[bin][i][1] * w;
            b_ch += bufs[bin][i][2] * w;
        }
        *pixel = (f64::from(r), f64::from(g), f64::from(b_ch), 1.0);
    });
}

/// Convert a `PixelBuffer` to packed 16-bit RGB for the ffmpeg `rgb48le` pipe.
pub(super) fn quantize_to_u16_rgb(pixels: &PixelBuffer) -> Vec<u16> {
    let mut buf = vec![0u16; pixels.len() * 3];
    buf.par_chunks_mut(3).zip(pixels.par_iter()).for_each(|(chunk, &(r, g, b, _a))| {
        chunk[0] = (r.clamp(0.0, 1.0) * constants::U16_MAX_F64).round() as u16;
        chunk[1] = (g.clamp(0.0, 1.0) * constants::U16_MAX_F64).round() as u16;
        chunk[2] = (b.clamp(0.0, 1.0) * constants::U16_MAX_F64).round() as u16;
    });
    buf
}

/// Generate a single spectral sweep video (violet to red) at the given path.
///
/// The sweep applies cosine easing for smooth pacing, Gaussian bloom and
/// subtle colour grading (vignette + vibrance) per frame.  The bin range is
/// determined dynamically from actual scene energy so that only bins with
/// visible content are swept.
///
/// # Errors
///
/// Returns an error if sweep frame generation or video encoding fails.
pub fn generate_spectral_sweep_video(
    accum_spd: &[[f64; NUM_BINS]],
    width: u32,
    height: u32,
    output_path: &str,
    fast_encode: bool,
) -> Result<()> {
    generate_spectral_sweep_video_with_encoder(
        accum_spd,
        width,
        height,
        output_path,
        fast_encode,
        &FfmpegVideoEncoder,
    )
}

pub(crate) fn generate_spectral_sweep_video_with_encoder(
    accum_spd: &[[f64; NUM_BINS]],
    width: u32,
    height: u32,
    output_path: &str,
    fast_encode: bool,
    video_encoder: &dyn VideoEncoder,
) -> Result<()> {
    info!("Building BinBuffers for spectral sweep ({NUM_BINS} bins)...");
    let bin_buffers = BinBuffers::new(accum_spd, width as usize, height as usize);

    let (active_start, active_end) = bin_buffers.active_bin_range();
    info!("   Active bin range: {active_start}..={active_end}");

    let w = width as usize;
    let h = height as usize;

    info!("Preparing sweep post-effects (bloom + colour grade)...");
    let bloom = GaussianBloom::new(
        constants::SWEEP_BLOOM_RADIUS,
        constants::SWEEP_BLOOM_STRENGTH,
        constants::SWEEP_BLOOM_CORE_BRIGHTNESS,
    );
    let color_grade = CinematicColorGrade::new(ColorGradeParams {
        strength: 1.0,
        vignette_strength: constants::SWEEP_VIGNETTE_STRENGTH,
        vignette_softness: constants::SWEEP_VIGNETTE_SOFTNESS,
        vibrance: constants::SWEEP_VIBRANCE,
        clarity_strength: 0.0,
        clarity_radius: 1,
        tone_curve: 0.0,
        shadow_tint: [0.0; 3],
        highlight_tint: [0.0; 3],
        palette_wave_strength: 0.0,
    });

    info!("Encoding spectral sweep video...");
    let total_frames = constants::CYCLE_TOTAL_FRAMES;
    let fps = constants::DEFAULT_VIDEO_FPS;
    let options = if fast_encode {
        VideoEncodingOptions::fast_encode()
    } else {
        VideoEncodingOptions::default()
    };

    let mut write_frames = |out: &mut dyn std::io::Write| {
        let mut frame_buf: PixelBuffer = Vec::new();
        let start = active_start as f64;
        let end = active_end as f64;
        let sigma = constants::SWEEP_GAUSSIAN_SIGMA;

        for frame in 0..total_frames {
            let t_linear = f64::from(frame) / f64::from(total_frames - 1);
            let t_eased = (1.0 - (t_linear * std::f64::consts::PI).cos()) * 0.5;
            let bin_f = start + t_eased * (end - start);

            gaussian_blend_to_pixelbuffer(&bin_buffers, bin_f, sigma, &mut frame_buf);

            let processed = apply_sweep_effects(&bloom, &color_grade, &frame_buf, w, h)
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

            let quantized = quantize_to_u16_rgb(&processed);
            let bytes: &[u8] = bytemuck::cast_slice(&quantized);
            out.write_all(bytes).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        }
        Ok(())
    };

    create_video_from_frames_singlepass_with_encoder(
        width,
        height,
        fps,
        &mut write_frames,
        output_path,
        &options,
        video_encoder,
    )?;

    info!("   Spectral sweep video complete => {output_path}");
    Ok(())
}

/// Run bloom then colour-grade on a single `PixelBuffer`.
fn apply_sweep_effects(
    bloom: &GaussianBloom,
    color_grade: &CinematicColorGrade,
    buf: &PixelBuffer,
    w: usize,
    h: usize,
) -> Result<PixelBuffer> {
    let bloomed = bloom.process(buf, w, h).map_err(|e| RenderError::EffectChain {
        effect_name: "gaussian_bloom".into(),
        reason: e.to_string(),
    })?;
    color_grade.process(&bloomed, w, h).map_err(|e| RenderError::EffectChain {
        effect_name: "color_grade".into(),
        reason: e.to_string(),
    })
}
