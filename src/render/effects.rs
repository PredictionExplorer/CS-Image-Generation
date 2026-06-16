//! Active finish-effect pipeline for the `CosmicSignature` renderer.

use super::constants;
use super::context::PixelBuffer;
use super::drawing::parallel_blur_2d_rgba;
use super::error::{RenderError, Result};
use crate::post_effects::{
    ChromaticBloom, ChromaticBloomConfig, DogBloom, GaussianBloom, PostEffectChain,
};
use crate::spectrum::{NUM_BINS, spd_to_rgba};
use rayon::prelude::*;

const LUMA_R: f64 = 0.299;
const LUMA_G: f64 = 0.587;
const LUMA_B: f64 = 0.114;

/// Configuration for active finish-chain creation.
#[derive(Clone, Debug)]
pub struct EffectConfig {
    /// Bloom mode selector (e.g., "gaussian", "dog", "none")
    pub bloom_mode: String,
    /// Gaussian blur radius in pixels
    pub blur_radius_px: usize,
    /// Bloom blur blend strength
    pub blur_strength: f64,
    /// Brightness preservation factor for bloom core
    pub blur_core_brightness: f64,
    /// Difference-of-Gaussians bloom configuration
    pub dog_config: DogBloomConfig,
    /// Whether chromatic bloom (prismatic separation) is enabled
    pub chromatic_bloom_enabled: bool,
    /// Chromatic bloom configuration
    pub chromatic_bloom_config: ChromaticBloomConfig,
}

/// Per-frame parameters that may vary
#[derive(Clone, Debug)]
pub struct FrameParams {
    /// Current animation frame index
    pub frame_number: usize,
    /// Optional density override for this frame
    pub density: Option<f64>,
}

/// Persistent finish pipeline with separate trajectory and image stages.
pub struct FinishEffectPipeline {
    trajectory_chain: PostEffectChain,
    image_chain: PostEffectChain,
}

impl FinishEffectPipeline {
    /// Create a new finish pipeline with given configuration.
    #[must_use]
    pub fn new(config: EffectConfig) -> Self {
        let trajectory_chain = Self::build_trajectory_chain(&config);
        let image_chain = Self::build_image_chain(&config);
        Self { trajectory_chain, image_chain }
    }

    /// Build the trajectory finish chain based on configuration.
    ///
    fn build_trajectory_chain(config: &EffectConfig) -> PostEffectChain {
        let mut chain = PostEffectChain::new();

        if config.blur_radius_px > 0 {
            chain.add(Box::new(GaussianBloom::new(
                config.blur_radius_px,
                config.blur_strength,
                config.blur_core_brightness,
            )));
        }

        if config.bloom_mode == "dog" {
            chain.add(Box::new(DogBloom::new(
                config.dog_config.clone(),
                config.blur_core_brightness,
            )));
        }

        if config.chromatic_bloom_enabled {
            chain.add(Box::new(ChromaticBloom::new(config.chromatic_bloom_config.clone())));
        }

        chain
    }

    fn build_image_chain(_config: &EffectConfig) -> PostEffectChain {
        PostEffectChain::new()
    }

    /// Process trajectory content through the persistent finish chain.
    pub fn process_trajectory(
        &self,
        buffer: PixelBuffer,
        width: usize,
        height: usize,
        _params: &FrameParams,
    ) -> Result<PixelBuffer> {
        self.trajectory_chain.process(buffer, width, height).map_err(|e| RenderError::EffectChain {
            effect_name: "trajectory_chain".into(),
            reason: e.to_string(),
        })
    }

    /// Process the fully composited display image through the final image chain.
    pub fn process_image(
        &self,
        buffer: PixelBuffer,
        width: usize,
        height: usize,
        _params: &FrameParams,
    ) -> Result<PixelBuffer> {
        self.image_chain.process(buffer, width, height).map_err(|e| RenderError::EffectChain {
            effect_name: "image_chain".into(),
            reason: e.to_string(),
        })
    }

    /// Number of effects in the trajectory processing chain
    #[cfg(test)]
    pub fn trajectory_len(&self) -> usize {
        self.trajectory_chain.len()
    }

    /// Number of effects in the image processing chain
    #[cfg(test)]
    pub fn image_len(&self) -> usize {
        self.image_chain.len()
    }
}

/// Configuration for Difference-of-Gaussians bloom
#[derive(Clone, Debug)]
pub struct DogBloomConfig {
    /// Base blur radius
    pub inner_sigma: f64,
    /// Outer sigma = inner * ratio (typically 2-3)
    pub outer_ratio: f64,
    /// `DoG` multiplier (0.2--0.8).
    pub strength: f64,
    /// Minimum value to include
    pub threshold: f64,
}

impl Default for DogBloomConfig {
    fn default() -> Self {
        Self { inner_sigma: 6.0, outer_ratio: 2.5, strength: 0.35, threshold: 0.01 }
    }
}

/// Mipmap pyramid for efficient multi-scale filtering
pub struct MipPyramid {
    levels: Vec<Vec<(f64, f64, f64, f64)>>,
    widths: Vec<usize>,
    heights: Vec<usize>,
}

impl MipPyramid {
    /// Build a mipmap pyramid with the given number of downsampled levels
    #[must_use]
    pub fn new(base: &[(f64, f64, f64, f64)], width: usize, height: usize, levels: usize) -> Self {
        let mut pyramid =
            MipPyramid { levels: vec![base.to_vec()], widths: vec![width], heights: vec![height] };

        for level in 1..levels {
            let prev_w = pyramid.widths[level - 1];
            let prev_h = pyramid.heights[level - 1];
            let new_w = prev_w.div_ceil(2);
            let new_h = prev_h.div_ceil(2);

            let mut downsampled = vec![(0.0, 0.0, 0.0, 0.0); new_w * new_h];

            // Box filter downsample (parallel)
            downsampled.par_iter_mut().enumerate().for_each(|(idx, pixel)| {
                let x = idx % new_w;
                let y = idx / new_w;

                // Sample 2x2 region from previous level
                let x0 = (x * 2).min(prev_w - 1);
                let x1 = ((x * 2) + 1).min(prev_w - 1);
                let y0 = (y * 2).min(prev_h - 1);
                let y1 = ((y * 2) + 1).min(prev_h - 1);

                let p00 = pyramid.levels[level - 1][y0 * prev_w + x0];
                let p01 = pyramid.levels[level - 1][y0 * prev_w + x1];
                let p10 = pyramid.levels[level - 1][y1 * prev_w + x0];
                let p11 = pyramid.levels[level - 1][y1 * prev_w + x1];

                *pixel = (
                    (p00.0 + p01.0 + p10.0 + p11.0) * constants::BILINEAR_AVG_FACTOR,
                    (p00.1 + p01.1 + p10.1 + p11.1) * constants::BILINEAR_AVG_FACTOR,
                    (p00.2 + p01.2 + p10.2 + p11.2) * constants::BILINEAR_AVG_FACTOR,
                    (p00.3 + p01.3 + p10.3 + p11.3) * constants::BILINEAR_AVG_FACTOR,
                );
            });

            pyramid.levels.push(downsampled);
            pyramid.widths.push(new_w);
            pyramid.heights.push(new_h);
        }

        pyramid
    }
}

/// Standalone bilinear upsampling function for arbitrary data
/// Handles premultiplied alpha values correctly
#[must_use]
pub fn upsample_bilinear(
    src: &[(f64, f64, f64, f64)],
    src_w: usize,
    src_h: usize,
    target_w: usize,
    target_h: usize,
) -> Vec<(f64, f64, f64, f64)> {
    let mut result = vec![(0.0, 0.0, 0.0, 0.0); target_w * target_h];

    result.par_iter_mut().enumerate().for_each(|(idx, pixel)| {
        let x = idx % target_w;
        let y = idx / target_w;

        // Map to source coordinates
        let sx = (x as f64 * src_w as f64 / target_w as f64).min((src_w - 1) as f64);
        let sy = (y as f64 * src_h as f64 / target_h as f64).min((src_h - 1) as f64);

        let x0 = sx.floor() as usize;
        let y0 = sy.floor() as usize;
        let x1 = (x0 + 1).min(src_w - 1);
        let y1 = (y0 + 1).min(src_h - 1);

        let fx = sx - x0 as f64;
        let fy = sy - y0 as f64;

        // Get source pixels (premultiplied RGBA)
        let p00 = src[y0 * src_w + x0];
        let p01 = src[y0 * src_w + x1];
        let p10 = src[y1 * src_w + x0];
        let p11 = src[y1 * src_w + x1];

        // Proper premultiplied alpha interpolation
        // Interpolate premultiplied values directly
        let top = (
            p00.0 * (1.0 - fx) + p01.0 * fx,
            p00.1 * (1.0 - fx) + p01.1 * fx,
            p00.2 * (1.0 - fx) + p01.2 * fx,
            p00.3 * (1.0 - fx) + p01.3 * fx,
        );

        let bottom = (
            p10.0 * (1.0 - fx) + p11.0 * fx,
            p10.1 * (1.0 - fx) + p11.1 * fx,
            p10.2 * (1.0 - fx) + p11.2 * fx,
            p10.3 * (1.0 - fx) + p11.3 * fx,
        );

        *pixel = (
            top.0 * (1.0 - fy) + bottom.0 * fy,
            top.1 * (1.0 - fy) + bottom.1 * fy,
            top.2 * (1.0 - fy) + bottom.2 * fy,
            top.3 * (1.0 - fy) + bottom.3 * fy,
        );

        // Renormalize for very low alpha to prevent color bleeding
        if pixel.3 > 0.0 && pixel.3 < 0.01 {
            let expected_alpha = p00.3 * (1.0 - fx) * (1.0 - fy)
                + p01.3 * fx * (1.0 - fy)
                + p10.3 * (1.0 - fx) * fy
                + p11.3 * fx * fy;
            if expected_alpha > 1e-10 {
                let scale = pixel.3 / expected_alpha;
                pixel.0 *= scale;
                pixel.1 *= scale;
                pixel.2 *= scale;
            }
        }
    });

    result
}

/// Apply Difference-of-Gaussians bloom effect
#[must_use]
pub fn apply_dog_bloom(
    input: &[(f64, f64, f64, f64)],
    width: usize,
    height: usize,
    config: &DogBloomConfig,
) -> Vec<(f64, f64, f64, f64)> {
    // Create mip pyramid (3 levels)
    let pyramid = MipPyramid::new(input, width, height, 3);

    // Blur at different mip levels for efficiency
    let inner_radius = config.inner_sigma.round() as usize;
    let outer_radius = (config.inner_sigma * config.outer_ratio).round() as usize;

    // Blur level 1 (half resolution) with inner sigma
    let mut blur_inner = pyramid.levels[1].clone();
    parallel_blur_2d_rgba(
        &mut blur_inner,
        pyramid.widths[1],
        pyramid.heights[1],
        inner_radius / 2, // Adjust for mip level
    );

    // Blur level 2 (quarter resolution) with outer sigma
    let mut blur_outer = pyramid.levels[2].clone();
    parallel_blur_2d_rgba(
        &mut blur_outer,
        pyramid.widths[2],
        pyramid.heights[2],
        outer_radius / 4, // Adjust for mip level
    );

    // Upsample both BLURRED data to original resolution
    let inner_upsampled =
        upsample_bilinear(&blur_inner, pyramid.widths[1], pyramid.heights[1], width, height);
    let outer_upsampled =
        upsample_bilinear(&blur_outer, pyramid.widths[2], pyramid.heights[2], width, height);

    // Compute DoG and apply threshold
    let mut dog_result = vec![(0.0, 0.0, 0.0, 0.0); width * height];

    dog_result
        .par_iter_mut()
        .zip(inner_upsampled.par_iter())
        .zip(outer_upsampled.par_iter())
        .for_each(|((dog, &inner), &outer)| {
            let diff = (inner.0 - outer.0, inner.1 - outer.1, inner.2 - outer.2, inner.3 - outer.3);

            // Compute luminance for thresholding
            let lum = LUMA_R * diff.0 + LUMA_G * diff.1 + LUMA_B * diff.2;

            if lum > config.threshold {
                *dog = (
                    diff.0 * config.strength,
                    diff.1 * config.strength,
                    diff.2 * config.strength,
                    diff.3 * config.strength,
                );
            }
            // Negative values are left as zero (clamped)
        });

    dog_result
}

/// Apply the rare diffraction-spike finish: thin astrophoto star-cross streaks
/// radiating from the brightest cores.
///
/// Deterministic and resolution-aware: the brightest pixels (relative to the
/// frame's own high percentile, so early dim video frames grow spikes as the
/// accumulation brightens) are marched along `arms` directions with an
/// exponential falloff. Sources are capped at the brightest
/// [`constants::SPIKE_MAX_SOURCES`] and processed in stable index order so the
/// pass is bit-reproducible.
pub fn apply_diffraction_spikes(
    buffer: &mut PixelBuffer,
    width: usize,
    height: usize,
    spikes: &crate::render::visual_profile::SpikeTraits,
) {
    if !spikes.enabled() || buffer.is_empty() || width == 0 || height == 0 {
        return;
    }

    let luminance =
        |p: &(f64, f64, f64, f64)| -> f64 { LUMA_R * p.0 + LUMA_G * p.1 + LUMA_B * p.2 };

    // Reference brightness: a high percentile of the lit pixels.
    let mut lit: Vec<f64> = buffer.iter().map(luminance).filter(|&lum| lum > 0.0).collect();
    if lit.len() < 16 {
        return;
    }
    let percentile_idx = ((lit.len() - 1) as f64 * constants::SPIKE_LUMINANCE_PERCENTILE) as usize;
    let (_, reference, _) =
        lit.select_nth_unstable_by(percentile_idx, |a, b| a.partial_cmp(b).expect("finite luma"));
    let reference = *reference;
    if reference <= 0.0 {
        return;
    }
    let threshold = reference * spikes.threshold_fraction;

    // Collect sources above threshold; cap at the brightest N, stable order.
    let mut sources: Vec<(usize, f64)> = buffer
        .iter()
        .enumerate()
        .filter_map(|(idx, p)| {
            let lum = luminance(p);
            (lum >= threshold).then_some((idx, lum))
        })
        .collect();
    if sources.is_empty() {
        return;
    }
    if sources.len() > constants::SPIKE_MAX_SOURCES {
        let cutoff = constants::SPIKE_MAX_SOURCES - 1;
        sources.select_nth_unstable_by(cutoff, |a, b| {
            b.1.partial_cmp(&a.1).expect("finite luma").then(a.0.cmp(&b.0))
        });
        sources.truncate(constants::SPIKE_MAX_SOURCES);
        sources.sort_unstable_by_key(|(idx, _)| *idx);
    }

    let min_dim = width.min(height) as f64;
    let arm_length = (spikes.length_scale * min_dim).max(2.0);
    let arms = usize::from(spikes.arms.max(2));
    let directions: Vec<(f64, f64)> = (0..arms)
        .map(|arm| {
            let theta = spikes.angle + std::f64::consts::TAU * arm as f64 / arms as f64;
            (theta.cos(), theta.sin())
        })
        .collect();

    let mut overlay = vec![(0.0f64, 0.0f64, 0.0f64, 0.0f64); buffer.len()];
    // usize→f64/isize casts: raster coordinates are far below precision limits.
    let steps = arm_length.ceil() as usize;
    for &(src_idx, lum) in &sources {
        let src = buffer[src_idx];
        // Excess above threshold drives the streak so spikes grow smoothly
        // with brightness instead of popping in.
        let drive = ((lum - threshold) / reference).clamp(0.0, 4.0) * spikes.strength;
        if drive <= 0.0 {
            continue;
        }
        let sx = (src_idx % width) as f64;
        let sy = (src_idx / width) as f64;
        for &(dx, dy) in &directions {
            for t in 1..=steps {
                let distance = t as f64;
                let px = sx + dx * distance;
                let py = sy + dy * distance;
                if px < 0.0 || py < 0.0 || px >= width as f64 || py >= height as f64 {
                    break;
                }
                let falloff = (-constants::SPIKE_DECAY_AT_TIP * distance / arm_length).exp();
                let w = drive * falloff;
                if w < 1e-6 {
                    break;
                }
                let dst = py as usize * width + px as usize;
                let dest = &mut overlay[dst];
                dest.0 += src.0 * w;
                dest.1 += src.1 * w;
                dest.2 += src.2 * w;
                dest.3 += src.3 * w;
            }
        }
    }

    buffer.par_iter_mut().zip(overlay.par_iter()).for_each(|(pixel, add)| {
        pixel.0 += add.0;
        pixel.1 += add.1;
        pixel.2 += add.2;
        pixel.3 = (pixel.3 + add.3).min(1.0);
    });
}

/// Convert SPD buffer to RGBA, with post-process radial dispersion (chromatic aberration)
pub(crate) fn convert_spd_buffer_to_rgba(
    src: &[[f64; NUM_BINS]],
    dest: &mut [(f64, f64, f64, f64)],
    width: usize,
    height: usize,
) {
    assert_eq!(src.len(), dest.len());

    use crate::render::drawing::DISPERSION_BOOST_ENABLED;
    use std::sync::atomic::Ordering;

    let dispersion_strength = if DISPERSION_BOOST_ENABLED.load(Ordering::Relaxed) {
        crate::render::constants::SPECTRAL_DISPERSION_STRENGTH_BOOSTED
    } else {
        crate::render::constants::SPECTRAL_DISPERSION_STRENGTH
    };

    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;
    let max_r = (cx * cx + cy * cy).sqrt();

    dest.par_iter_mut().enumerate().for_each(|(idx, dest_pixel)| {
        let x = (idx % width) as f64;
        let y = (idx / width) as f64;

        let dx = x - cx;
        let dy = y - cy;
        let r = (dx * dx + dy * dy).sqrt();
        let dir_x = if r > 0.0 { dx / r } else { 0.0 };
        let dir_y = if r > 0.0 { dy / r } else { 0.0 };

        let r_norm = r / max_r;

        let mut local_spd = [0.0f64; NUM_BINS];

        if dispersion_strength > 0.0 {
            for bin in 0..NUM_BINS {
                let bin_offset =
                    (bin as f64 - (NUM_BINS as f64 - 1.0) / 2.0) / ((NUM_BINS as f64 - 1.0) / 2.0);
                let shift = bin_offset * dispersion_strength * r_norm * 50.0;

                let sx = (x - dir_x * shift).round() as isize;
                let sy = (y - dir_y * shift).round() as isize;

                if sx >= 0 && sx < width as isize && sy >= 0 && sy < height as isize {
                    let s_idx = sy as usize * width + sx as usize;
                    local_spd[bin] = src[s_idx][bin];
                }
            }
        } else {
            local_spd = src[idx];
        }

        let rgba = spd_to_rgba(&local_spd);
        *dest_pixel = rgba;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_effect_config() -> EffectConfig {
        EffectConfig {
            bloom_mode: "none".to_string(),
            blur_radius_px: 0,
            blur_strength: 0.0,
            blur_core_brightness: 1.0,
            dog_config: DogBloomConfig::default(),
            chromatic_bloom_enabled: false,
            chromatic_bloom_config: ChromaticBloomConfig::default(),
        }
    }

    #[test]
    fn test_finish_pipeline_keeps_image_stage_empty() {
        let mut config = base_effect_config();
        config.chromatic_bloom_enabled = true;

        let pipeline = FinishEffectPipeline::new(config);

        assert_eq!(pipeline.trajectory_len(), 1);
        assert_eq!(pipeline.image_len(), 0);
    }

    #[test]
    fn test_finish_pipeline_routes_bloom_to_trajectory_stage() {
        let mut config = base_effect_config();
        config.bloom_mode = "dog".to_string();

        let pipeline = FinishEffectPipeline::new(config);

        assert_eq!(pipeline.trajectory_len(), 1);
        assert_eq!(pipeline.image_len(), 0);
    }

    #[test]
    fn test_mip_pyramid_dimensions() {
        let w = 64;
        let h = 64;
        let input: Vec<(f64, f64, f64, f64)> = vec![(0.5, 0.5, 0.5, 1.0); w * h];
        let pyramid = MipPyramid::new(&input, w, h, 3);
        assert_eq!(pyramid.widths[0], w);
        assert_eq!(pyramid.heights[0], h);
        assert_eq!(pyramid.widths[1], w / 2);
        assert_eq!(pyramid.heights[1], h / 2);
        assert_eq!(pyramid.widths[2], w / 4);
        assert_eq!(pyramid.heights[2], h / 4);
        assert_eq!(pyramid.levels.len(), 3);
    }

    #[test]
    fn test_mip_pyramid_single_level() {
        let w = 16;
        let h = 16;
        let input: Vec<(f64, f64, f64, f64)> = vec![(1.0, 0.5, 0.25, 1.0); w * h];
        let pyramid = MipPyramid::new(&input, w, h, 1);
        assert_eq!(pyramid.levels.len(), 1);
        assert_eq!(pyramid.levels[0], input);
    }

    #[test]
    fn test_upsample_bilinear_identity() {
        let w = 4;
        let h = 4;
        let input: Vec<(f64, f64, f64, f64)> = vec![(0.5, 0.5, 0.5, 1.0); w * h];
        let result = upsample_bilinear(&input, w, h, w, h);
        assert_eq!(result.len(), w * h);
        for pixel in &result {
            assert!((pixel.0 - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_upsample_bilinear_doubles_size() {
        let w = 4;
        let h = 4;
        let input: Vec<(f64, f64, f64, f64)> = vec![(1.0, 0.5, 0.25, 1.0); w * h];
        let result = upsample_bilinear(&input, w, h, w * 2, h * 2);
        assert_eq!(result.len(), w * h * 4);
        for pixel in &result {
            assert!(
                (pixel.0 - 1.0).abs() < 0.5,
                "Upsampled uniform data should stay near original"
            );
        }
    }

    fn spike_fixture(width: usize, height: usize) -> PixelBuffer {
        // Faint field plus two bright cores.
        let mut buffer = vec![(0.002, 0.002, 0.002, 0.01); width * height];
        buffer[(height / 2) * width + width / 2] = (4.0, 3.5, 3.0, 1.0);
        buffer[(height / 4) * width + width / 3] = (3.0, 3.2, 4.0, 1.0);
        buffer
    }

    fn buffer_energy(buffer: &PixelBuffer) -> f64 {
        buffer.iter().map(|&(r, g, b, _)| r + g + b).sum()
    }

    fn enabled_spikes() -> crate::render::visual_profile::SpikeTraits {
        crate::render::visual_profile::SpikeTraits {
            strength: 0.25,
            arms: 4,
            angle: 0.3,
            length_scale: 0.04,
            threshold_fraction: 0.65,
        }
    }

    #[test]
    fn test_diffraction_spikes_disabled_is_a_strict_noop() {
        let width = 48;
        let height = 32;
        let mut buffer = spike_fixture(width, height);
        let original = buffer.clone();
        apply_diffraction_spikes(
            &mut buffer,
            width,
            height,
            &crate::render::visual_profile::SpikeTraits::disabled(),
        );
        assert_eq!(buffer, original, "disabled spikes must not touch the buffer");
    }

    #[test]
    fn test_diffraction_spikes_add_energy_around_bright_cores() {
        let width = 48;
        let height = 32;
        let mut buffer = spike_fixture(width, height);
        let before = buffer_energy(&buffer);
        apply_diffraction_spikes(&mut buffer, width, height, &enabled_spikes());
        let after = buffer_energy(&buffer);

        assert!(after > before, "spikes must add streak energy: {before} -> {after}");
        // Streaks stay a finish, not a flood.
        assert!(after < before * 3.0, "spike energy exploded: {before} -> {after}");
        for &(r, g, b, a) in &buffer {
            assert!(
                r.is_finite() && g.is_finite() && b.is_finite() && a.is_finite(),
                "spike output must stay finite"
            );
        }
    }

    #[test]
    fn test_diffraction_spikes_are_deterministic() {
        let width = 40;
        let height = 30;
        let mut a = spike_fixture(width, height);
        let mut b = spike_fixture(width, height);
        let spikes = enabled_spikes();
        apply_diffraction_spikes(&mut a, width, height, &spikes);
        apply_diffraction_spikes(&mut b, width, height, &spikes);
        for (idx, (pa, pb)) in a.iter().zip(&b).enumerate() {
            assert_eq!(pa.0.to_bits(), pb.0.to_bits(), "pixel {idx} R diverged");
            assert_eq!(pa.1.to_bits(), pb.1.to_bits(), "pixel {idx} G diverged");
            assert_eq!(pa.2.to_bits(), pb.2.to_bits(), "pixel {idx} B diverged");
            assert_eq!(pa.3.to_bits(), pb.3.to_bits(), "pixel {idx} A diverged");
        }
    }

    #[test]
    fn test_diffraction_spikes_respect_arm_count_directionality() {
        let width = 64;
        let height = 64;
        // Faint field (so the pass engages) plus one bright core.
        let mut buffer = vec![(0.0005, 0.0005, 0.0005, 0.01); width * height];
        buffer[(height / 2) * width + width / 2] = (5.0, 5.0, 5.0, 1.0);
        let spikes = crate::render::visual_profile::SpikeTraits {
            strength: 0.3,
            arms: 4,
            angle: 0.0,
            length_scale: 0.2,
            threshold_fraction: 0.5,
        };
        apply_diffraction_spikes(&mut buffer, width, height, &spikes);

        // With angle 0 and 4 arms, axis-aligned neighbors receive streaks
        // while the diagonals stay dark.
        let cx = width / 2;
        let cy = height / 2;
        let energy = |x: usize, y: usize| {
            let p = buffer[y * width + x];
            p.0 + p.1 + p.2
        };
        let axis =
            energy(cx + 5, cy) + energy(cx - 5, cy) + energy(cx, cy + 5) + energy(cx, cy - 5);
        let diagonal = energy(cx + 5, cy + 5)
            + energy(cx - 5, cy - 5)
            + energy(cx + 5, cy - 5)
            + energy(cx - 5, cy + 5);
        assert!(
            axis > diagonal * 10.0,
            "4-arm spikes at angle 0 must streak along the axes: axis={axis} diag={diagonal}"
        );
    }

    #[test]
    fn test_apply_dog_bloom_output_size() {
        let w = 32;
        let h = 32;
        let input: Vec<(f64, f64, f64, f64)> = vec![(0.5, 0.5, 0.5, 1.0); w * h];
        let result = apply_dog_bloom(&input, w, h, &DogBloomConfig::default());
        assert_eq!(result.len(), w * h);
    }

    #[test]
    fn test_apply_dog_bloom_dark_input_near_zero() {
        let w = 16;
        let h = 16;
        let input: Vec<(f64, f64, f64, f64)> = vec![(0.01, 0.01, 0.01, 1.0); w * h];
        let result = apply_dog_bloom(&input, w, h, &DogBloomConfig::default());
        for pixel in &result {
            assert!(pixel.0.abs() < 0.1, "Dark input should produce near-zero bloom");
        }
    }
}
