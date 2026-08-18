//! Re-accumulation infrastructure: a thin owner around the production
//! spectral accumulator for transformed scenes, custom step schedules, and
//! decayed (comet-style) accumulation, plus shared helpers for per-scene
//! tonemap levels, resized render settings, and incremental video streaming.

use crate::error::Result;
use crate::render::constants::{
    DEFAULT_HISTOGRAM_SAMPLE_FRAMES, DEFAULT_TONEMAP_HIGHLIGHT_ROLLOFF, DEFAULT_TONEMAP_PAPER_WHITE,
};
use crate::render::context::{BoundingBox, PixelBuffer, RenderContext};
use crate::render::effects::convert_spd_buffer_to_rgba;
use crate::render::histogram::analyze_tonemapping;
use crate::render::randomizable_config::ResolvedEffectConfig;
use crate::render::velocity_hdr::VelocityHdrCalculator;
use crate::render::{
    AccumulationParams, ChannelLevels, OklabColor, SceneTraits, SpectralRenderSettings,
    SpectralScene, ToneMappingControls, VideoEncodingOptions, VideoOutputSpec,
    accumulate_spectral_steps, create_videos_from_frames_singlepass, default_accumulation_backend,
    pass_1_build_histogram_spectral, quantize_display_buffer_to_16bit, tonemap_to_display_buffer,
};
use crate::spectrum::NUM_BINS;
use nalgebra::Vector3;
use std::ops::Range;

/// Owns a (possibly transformed) scene and an SPD buffer, and runs the
/// production accumulator over it with the seed's scene traits.
pub struct Accumulator {
    positions: Vec<Vec<Vector3<f64>>>,
    colors: Vec<Vec<OklabColor>>,
    body_alphas: Vec<f64>,
    ctx: RenderContext,
    traits: SceneTraits,
    hdr_scale: f64,
    spd: Vec<[f64; NUM_BINS]>,
}

impl Accumulator {
    /// Build with automatic bounds from the given positions.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        positions: Vec<Vec<Vector3<f64>>>,
        colors: Vec<Vec<OklabColor>>,
        body_alphas: Vec<f64>,
        width: u32,
        height: u32,
        aspect_correction: bool,
        traits: SceneTraits,
        hdr_scale: f64,
    ) -> Self {
        let ctx = RenderContext::new(width, height, &positions, aspect_correction);
        Self::with_context(positions, colors, body_alphas, ctx, traits, hdr_scale)
    }

    /// Build with explicit world bounds (shared-scale multi-panel renders).
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn with_bounds(
        positions: Vec<Vec<Vector3<f64>>>,
        colors: Vec<Vec<OklabColor>>,
        body_alphas: Vec<f64>,
        width: u32,
        height: u32,
        bounds: BoundingBox,
        traits: SceneTraits,
        hdr_scale: f64,
    ) -> Self {
        let ctx = RenderContext::with_bounds(width, height, bounds);
        Self::with_context(positions, colors, body_alphas, ctx, traits, hdr_scale)
    }

    fn with_context(
        positions: Vec<Vec<Vector3<f64>>>,
        colors: Vec<Vec<OklabColor>>,
        body_alphas: Vec<f64>,
        ctx: RenderContext,
        traits: SceneTraits,
        hdr_scale: f64,
    ) -> Self {
        let spd = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
        Self { positions, colors, body_alphas, ctx, traits, hdr_scale, spd }
    }

    /// Borrow the scene view over the owned trajectory.
    #[must_use]
    pub fn scene(&self) -> SpectralScene<'_> {
        SpectralScene::new(&self.positions, &self.colors, &self.body_alphas)
    }

    /// Number of steps in the owned trajectory.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.positions.first().map_or(0, Vec::len)
    }

    /// Accumulate a step range with the production splatter.
    pub fn accumulate(&mut self, range: Range<usize>) {
        if range.is_empty() {
            return;
        }
        let step_end = range.end.min(self.step_count());
        let velocity_calc =
            VelocityHdrCalculator::new(&self.positions, crate::render::constants::DEFAULT_DT);
        accumulate_spectral_steps(
            &mut self.spd,
            &AccumulationParams {
                scene: SpectralScene::new(&self.positions, &self.colors, &self.body_alphas),
                ctx: &self.ctx,
                velocity_calc: &velocity_calc,
                step_start: range.start,
                step_end,
                hdr_scale: self.hdr_scale,
                traits: self.traits,
            },
            default_accumulation_backend(),
        );
    }

    /// Multiply the whole SPD by a decay factor (comet-style fading).
    pub fn decay(&mut self, factor: f64) {
        use rayon::prelude::*;
        self.spd.par_iter_mut().for_each(|bins| {
            for value in bins.iter_mut() {
                *value *= factor;
            }
        });
    }

    /// Zero the SPD buffer (window re-use).
    pub fn clear(&mut self) {
        use rayon::prelude::*;
        self.spd.par_iter_mut().for_each(|bins| *bins = [0.0; NUM_BINS]);
    }

    /// Convert the current SPD into a fresh linear RGBA buffer.
    #[must_use]
    pub fn convert(&self) -> PixelBuffer {
        let mut rgba = vec![(0.0, 0.0, 0.0, 0.0); self.ctx.pixel_count()];
        self.convert_into(&mut rgba);
        rgba
    }

    /// Convert the current SPD into an existing RGBA buffer.
    pub fn convert_into(&self, rgba: &mut PixelBuffer) {
        rgba.resize(self.ctx.pixel_count(), (0.0, 0.0, 0.0, 0.0));
        convert_spd_buffer_to_rgba(&self.spd, rgba, self.ctx.width_usize, self.ctx.height_usize);
    }

    /// Immutable access to the SPD (loop blending, snapshots).
    #[must_use]
    pub fn spd(&self) -> &[[f64; NUM_BINS]] {
        &self.spd
    }

    /// Snapshot the SPD buffer.
    #[must_use]
    pub fn snapshot(&self) -> Vec<[f64; NUM_BINS]> {
        self.spd.clone()
    }

    /// Render context (world-to-pixel mapping) for this accumulator.
    #[must_use]
    pub fn render_ctx(&self) -> &RenderContext {
        &self.ctx
    }
}

/// Production pass-1 levels for an arbitrary (transformed) scene.
#[must_use]
pub fn scene_levels(
    scene: SpectralScene<'_>,
    settings: SpectralRenderSettings<'_>,
    exposure_key: f64,
) -> ChannelLevels {
    let interval = (scene.step_count() / DEFAULT_HISTOGRAM_SAMPLE_FRAMES as usize).max(1);
    let histogram = pass_1_build_histogram_spectral(scene, interval, settings);
    let analysis = analyze_tonemapping(
        histogram.data(),
        settings.resolved_config.clip_black,
        settings.resolved_config.clip_white,
    );
    ChannelLevels::with_tone_mapping(
        analysis.black_r,
        analysis.white_r,
        analysis.black_g,
        analysis.white_g,
        analysis.black_b,
        analysis.white_b,
        ToneMappingControls {
            exposure_scale: analysis.exposure_scale * exposure_key.clamp(0.5, 1.5),
            paper_white: DEFAULT_TONEMAP_PAPER_WHITE,
            highlight_rolloff: DEFAULT_TONEMAP_HIGHLIGHT_ROLLOFF,
        },
    )
}

/// Clone the resolved effect config with new output dimensions (even-rounded).
#[must_use]
pub fn resized_config(
    base: &ResolvedEffectConfig,
    width: u32,
    height: u32,
) -> ResolvedEffectConfig {
    let mut resized = base.clone();
    resized.width = (width & !1).max(16);
    resized.height = (height & !1).max(16);
    resized
}

/// Stream an incremental video: the closure fills the linear RGBA buffer for
/// each frame; grading and quantization use the given fixed levels.
#[allow(clippy::too_many_arguments)]
pub fn stream_video(
    width: u32,
    height: u32,
    fps: u32,
    web_path: &str,
    hq_path: &str,
    fast_encode: bool,
    frame_count: usize,
    levels: &ChannelLevels,
    mut render_frame: impl FnMut(usize, &mut PixelBuffer),
) -> Result<()> {
    let even_w = (width & !1).max(16);
    let even_h = (height & !1).max(16);
    let outputs = [
        VideoOutputSpec {
            output_file: web_path.to_string(),
            options: VideoEncodingOptions::web_compatible(),
        },
        VideoOutputSpec {
            output_file: hq_path.to_string(),
            options: if fast_encode {
                VideoEncodingOptions::fast_encode()
            } else {
                VideoEncodingOptions::high_quality()
            },
        },
    ];
    create_videos_from_frames_singlepass(
        even_w,
        even_h,
        fps,
        |out| {
            let mut rgba: PixelBuffer =
                vec![(0.0, 0.0, 0.0, 0.0); width as usize * height as usize];
            let mut cropped: Vec<u16> = Vec::new();
            for frame in 0..frame_count {
                render_frame(frame, &mut rgba);
                let display = tonemap_to_display_buffer(&rgba, levels);
                let full = quantize_display_buffer_to_16bit(&display);
                cropped.clear();
                for y in 0..even_h as usize {
                    let row = y * width as usize * 3;
                    cropped.extend_from_slice(&full[row..row + even_w as usize * 3]);
                }
                out.write_all(bytemuck::cast_slice(&cropped))
                    .map_err(crate::render::error::RenderError::VideoEncoding)?;
            }
            Ok(())
        },
        &outputs,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_scales_all_bins() {
        let positions: Vec<Vec<Vector3<f64>>> = (0..3)
            .map(|body| {
                (0..64)
                    .map(|step| {
                        let t = f64::from(step) * 0.1 + f64::from(body);
                        Vector3::new(t.cos(), t.sin(), 0.0)
                    })
                    .collect()
            })
            .collect();
        let colors: Vec<Vec<OklabColor>> = (0..3).map(|_| vec![(0.7, 0.1, 0.05); 64]).collect();
        let mut accumulator = Accumulator::new(
            positions,
            colors,
            vec![0.5; 3],
            64,
            48,
            false,
            SceneTraits::default(),
            1.0,
        );
        accumulator.accumulate(0..63);
        let before: f64 = accumulator.spd().iter().flat_map(|bins| bins.iter()).sum();
        assert!(before > 0.0, "accumulation must deposit energy");
        accumulator.decay(0.5);
        let after: f64 = accumulator.spd().iter().flat_map(|bins| bins.iter()).sum();
        assert!((after - before * 0.5).abs() < before * 1e-9);
    }
}
