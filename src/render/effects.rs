//! Post-processing effects pipeline
//!
//! This module manages the visual effects chain including bloom, blur, and tone mapping.
//! It provides a configurable pipeline for post-processing rendered frames.

use super::constants;
use super::context::PixelBuffer;
use super::drawing::parallel_blur_2d_rgba;
use super::error::{RenderError, Result};
use crate::post_effects::{
    AtmosphericDepth, AtmosphericDepthConfig, BloomPyramid, ChampleveConfig, ChromaticBloom,
    ChromaticBloomConfig, CinematicColorGrade, ColorGradeParams, DiffractionSpikes, DogBloom,
    EdgeLuminance, EdgeLuminanceConfig, FineTexture, FineTextureConfig, GaussianBloom, Glaze,
    GlowEnhancement, GlowEnhancementConfig, GodRays, GradientMap, GradientMapConfig, MicroContrast,
    MicroContrastConfig, Opalescence, OpalescenceConfig, PaletteHarmony, PaletteScheme,
    PerceptualBlur, PerceptualBlurConfig, PostEffect, PostEffectChain, StarField,
    aether::AetherConfig, apply_aether_weave, apply_champleve_iridescence,
    lens_flare::LensFlareDiffractive,
};
use crate::render::pipeline_flags;
use crate::spectrum::{NUM_BINS, spd_to_rgba};
use rayon::prelude::*;

const LUMA_R: f64 = 0.299;
const LUMA_G: f64 = 0.587;
const LUMA_B: f64 = 0.114;

/// Configuration for effect chain creation
///
/// Controls which effects are enabled and their parameters. Effects are applied
/// in a carefully ordered sequence for optimal visual quality:
/// 1. Bloom effects (diffuse glow)
/// 2. Tone mapping and blur
/// 3. Color manipulation (palettes, grading)
/// 4. Material effects (iridescence, structure)
/// 5. Detail enhancement (edges, contrast)
/// 6. Atmospheric effects (depth, texture)
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
    /// Whether perceptual (`OKLab`) blur is enabled.
    pub perceptual_blur_enabled: bool,
    /// Perceptual blur parameters, if enabled
    pub perceptual_blur_config: Option<PerceptualBlurConfig>,

    /// Whether cinematic color grading is enabled
    pub color_grade_enabled: bool,
    /// Cinematic color grading parameters
    pub color_grade_params: ColorGradeParams,
    /// Whether gradient map color remapping is enabled
    pub gradient_map_enabled: bool,
    /// Gradient map configuration
    pub gradient_map_config: GradientMapConfig,

    /// Whether champlevé iridescence effect is enabled
    pub champleve_enabled: bool,
    /// Champlevé iridescence configuration
    pub champleve_config: ChampleveConfig,
    /// Whether aether weave effect is enabled
    pub aether_enabled: bool,
    /// Aether weave configuration
    pub aether_config: AetherConfig,
    /// Whether chromatic bloom (prismatic separation) is enabled
    pub chromatic_bloom_enabled: bool,
    /// Chromatic bloom configuration
    pub chromatic_bloom_config: ChromaticBloomConfig,
    /// Whether opalescence shimmer effect is enabled
    pub opalescence_enabled: bool,
    /// Opalescence shimmer configuration
    pub opalescence_config: OpalescenceConfig,

    /// Whether edge luminance enhancement is enabled
    pub edge_luminance_enabled: bool,
    /// Edge luminance configuration
    pub edge_luminance_config: EdgeLuminanceConfig,
    /// Whether micro-contrast detail enhancement is enabled
    pub micro_contrast_enabled: bool,
    /// Micro-contrast configuration
    pub micro_contrast_config: MicroContrastConfig,
    /// Whether glow enhancement (tight sparkle) is enabled
    pub glow_enhancement_enabled: bool,
    /// Glow enhancement configuration
    pub glow_enhancement_config: GlowEnhancementConfig,

    /// Whether atmospheric depth/fog effect is enabled
    pub atmospheric_depth_enabled: bool,
    /// Atmospheric depth configuration
    pub atmospheric_depth_config: AtmosphericDepthConfig,
    /// Whether fine surface texture effect is enabled
    pub fine_texture_enabled: bool,
    /// Fine texture configuration
    pub fine_texture_config: FineTextureConfig,

    /// Whether the mood-driven multi-tier bloom pyramid is enabled.
    pub bloom_pyramid_enabled: bool,
    /// Whether the wide cyan anamorphic streak variant is enabled.
    pub anamorphic_flare_enabled: bool,
    /// Whether volumetric god rays are enabled.
    pub god_rays_enabled: bool,
    /// Whether the procedural star field is enabled.
    pub star_field_enabled: bool,
    /// Star field random seed (deterministic from simulation seed).
    pub star_field_seed: i32,
    /// Whether diffraction spikes are enabled.
    pub diffraction_spikes_enabled: bool,
    /// Whether the `OKLab` palette harmonizer is enabled.
    pub palette_harmony_enabled: bool,
    /// Palette harmonizer random seed.
    pub palette_harmony_seed: i32,
    /// Whether the warm highlight glaze is enabled.
    pub glaze_enabled: bool,

    /// Output width in pixels (used to size per-image effects at construction).
    pub output_width: usize,
    /// Output height in pixels (used to size per-image effects at construction).
    pub output_height: usize,
}

impl EffectConfig {
    /// Output width in pixels.
    #[must_use]
    #[inline]
    pub fn width_px(&self) -> usize {
        self.output_width
    }

    /// Output height in pixels.
    #[must_use]
    #[inline]
    pub fn height_px(&self) -> usize {
        self.output_height
    }
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
    /// Create a new finish pipeline with given configuration
    #[must_use]
    pub fn new(config: EffectConfig) -> Self {
        let trajectory_chain = Self::build_trajectory_chain(&config);
        let image_chain = Self::build_image_chain(&config);
        Self { trajectory_chain, image_chain }
    }

    /// Build the trajectory finish chain based on configuration.
    ///
    /// Effects are applied in a carefully optimized order:
    /// 1. Bloom effects (diffuse and tight glow)
    /// 2. Tone mapping and perceptual smoothing
    /// 3. Detail enhancement (contrast, clarity)
    /// 4. Color manipulation (palettes, grading)
    /// 5. Material properties (iridescence layers)
    /// 6. Form refinement (edges)
    /// 7. Atmospheric effects (depth)
    fn build_trajectory_chain(config: &EffectConfig) -> PostEffectChain {
        let mut chain = PostEffectChain::new();

        // ===== PHASE 0: COSMIC BACKGROUND =====
        // Drop in a star field before bloom so the bright stars glow too.
        if config.star_field_enabled {
            chain.add(Box::new(StarField::new(
                config.width_px(),
                config.height_px(),
                config.star_field_seed,
                0.55,
            )));
        }

        // ===== PHASE 1: BLOOM & GLOW =====
        // Base lighting effects that work on bright areas

        // 1a. Traditional bloom (large diffuse glow). When the multi-tier
        // pyramid is active, it takes the role of the wide halation layer and
        // the single-radius Gaussian is skipped to avoid double-blooming.
        if config.blur_radius_px > 0 && !config.bloom_pyramid_enabled {
            chain.add(Box::new(GaussianBloom::new(
                config.blur_radius_px,
                config.blur_strength,
                config.blur_core_brightness,
            )));
        }

        // 1a'. Multi-tier bloom pyramid (cinematic mood).
        if config.bloom_pyramid_enabled {
            let base = config.blur_radius_px.max(4);
            chain.add(Box::new(BloomPyramid::new(
                base,
                config.blur_strength.max(0.25),
                config.blur_core_brightness,
            )));
        }

        // 1b. DoG bloom (edge-detected glow, mutually exclusive with Gaussian)
        if config.bloom_mode == "dog" {
            chain.add(Box::new(DogBloom::new(
                config.dog_config.clone(),
                config.blur_core_brightness,
            )));
        }

        // 1c. Glow enhancement (tight sparkle on very bright areas)
        if config.glow_enhancement_enabled {
            chain.add(Box::new(GlowEnhancement::new(config.glow_enhancement_config.clone())));
        }

        // 1d. Chromatic bloom (prismatic color separation)
        if config.chromatic_bloom_enabled {
            chain.add(Box::new(ChromaticBloom::new(config.chromatic_bloom_config.clone())));
        }

        // 1e. Volumetric god rays from thresholded highlights (cinematic mood).
        if config.god_rays_enabled {
            chain.add(Box::new(GodRays::for_image(config.width_px(), config.height_px(), 0.45)));
        }

        // ===== PHASE 2: TONE MAPPING & BLUR =====
        // Perceptual processing for smooth, natural appearance

        // 2a. Perceptual blur (OKLab space smoothing)
        if config.perceptual_blur_enabled
            && let Some(blur_config) = &config.perceptual_blur_config
        {
            chain.add(Box::new(PerceptualBlur::new(blur_config.clone())));
        }

        // ===== PHASE 3: DETAIL ENHANCEMENT =====
        // Clarity and definition improvements

        // 3. Micro-contrast (local contrast enhancement for detail clarity)
        if config.micro_contrast_enabled {
            chain.add(Box::new(MicroContrast::new(config.micro_contrast_config.clone())));
        }

        // ===== PHASE 4: COLOR MANIPULATION =====
        // Artistic color transformations

        // 4a. Gradient mapping (luxury color palettes)
        if config.gradient_map_enabled {
            chain.add(Box::new(GradientMap::new(config.gradient_map_config.clone())));
        }

        // 4a'. OKLab palette harmonizer (painterly mood). Snaps pixel hues
        // toward anchors drawn from a triadic palette for colour cohesion.
        if config.palette_harmony_enabled {
            chain.add(Box::new(PaletteHarmony::from_seed(
                config.palette_harmony_seed,
                0.45,
                PaletteScheme::Triadic,
            )));
        }

        // 4b. Cinematic color grading (film-like look)
        if config.color_grade_enabled && config.color_grade_params.strength > 0.0 {
            chain.add(Box::new(CinematicColorGrade::new(config.color_grade_params.clone())));
        }

        // 4c. Warm highlight glaze (painterly mood, after color grade so it
        // lifts the already-graded highlights rather than fighting the grade).
        if config.glaze_enabled {
            chain.add(Box::new(Glaze::default()));
        }

        // ===== PHASE 5: MATERIAL PROPERTIES =====
        // Iridescence and material quality (layered for depth)

        // 5a. Opalescence (base gem-like shimmer layer)
        if config.opalescence_enabled {
            chain.add(Box::new(Opalescence::new(config.opalescence_config.clone())));
        }

        // 5b. Champlevé (structure layer: Voronoi cells + metallic rims)
        if config.champleve_enabled {
            chain.add(Box::new(ChampleveFinish::new(config.champleve_config.clone())));
        }

        // 5c. Aether (flow layer: woven filaments + volumetric scattering)
        if config.aether_enabled {
            chain.add(Box::new(AetherFinish::new(config.aether_config.clone())));
        }

        // ===== PHASE 6: FORM REFINEMENT =====
        // Edge and shape definition

        // 6. Edge luminance (selective edge brightening for refined forms)
        if config.edge_luminance_enabled {
            chain.add(Box::new(EdgeLuminance::new(config.edge_luminance_config.clone())));
        }

        // ===== PHASE 7: ATMOSPHERIC & SURFACE =====
        // Final spatial and material qualities

        // 7a. Atmospheric depth (spatial perspective + fog)
        if config.atmospheric_depth_enabled {
            chain.add(Box::new(AtmosphericDepth::new(config.atmospheric_depth_config.clone())));
        }

        chain
    }

    fn build_image_chain(config: &EffectConfig) -> PostEffectChain {
        let mut chain = PostEffectChain::new();

        if config.fine_texture_enabled {
            chain.add(Box::new(FineTexture::new(config.fine_texture_config.clone())));
        }

        // Diffraction spikes on bright points (cosmic mood): after tonemap but
        // before lens flare so the flare can also pick up the spikes.
        if config.diffraction_spikes_enabled {
            chain.add(Box::new(DiffractionSpikes::for_image(
                config.width_px(),
                config.height_px(),
                0.30,
            )));
        }

        // Classic star + horizontal streak flare (pipeline-flag opt-in).
        if pipeline_flags::lens_flare_enabled() {
            chain.add(Box::new(LensFlareDiffractive::pipeline_default()));
        }

        // Anamorphic wide blue streak (cinematic mood), independent of the
        // legacy lens-flare flag.
        if config.anamorphic_flare_enabled {
            chain.add(Box::new(LensFlareDiffractive::anamorphic()));
        }

        chain
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
        crate::render::constants::SPECTRAL_DISPERSION_STRENGTH_BOOSTED * 3.0
    } else {
        crate::render::constants::SPECTRAL_DISPERSION_STRENGTH * 3.0
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
        let ortho_x = -dir_y;
        let ortho_y = dir_x;

        let r_norm = (r / max_r).clamp(0.0, 1.0);
        let rf2 = r_norm * r_norm;

        let mut local_spd = [0.0f64; NUM_BINS];

        if dispersion_strength > 0.0 && pipeline_flags::ca_model_physical() {
            // Longitudinal + weak tangential dispersion ~ r² (achromat-style).
            let ca = dispersion_strength * (1.0 + rf2 * 1.5);
            for bin in 0..NUM_BINS {
                let nu = bin as f64 / (NUM_BINS as f64 - 1.0).max(1.0);
                let long = (nu - 0.5) * ca * 38.0 * rf2;
                let tang = (nu - 0.5) * ca * 9.0 * r_norm.sqrt();
                let sx = (x - dir_x * long + ortho_x * tang).round() as isize;
                let sy = (y - dir_y * long + ortho_y * tang).round() as isize;

                if sx >= 0 && sx < width as isize && sy >= 0 && sy < height as isize {
                    let s_idx = sy as usize * width + sx as usize;
                    local_spd[bin] = src[s_idx][bin];
                }
            }
        } else if dispersion_strength > 0.0 {
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

struct ChampleveFinish {
    config: ChampleveConfig,
}

impl ChampleveFinish {
    fn new(config: ChampleveConfig) -> Self {
        Self { config }
    }
}

impl PostEffect for ChampleveFinish {
    fn process(
        &self,
        input: &PixelBuffer,
        width: usize,
        height: usize,
    ) -> std::result::Result<PixelBuffer, crate::post_effects::PostEffectError> {
        let mut buffer = input.clone();
        apply_champleve_iridescence(&mut buffer, width, height, &self.config);
        Ok(buffer)
    }
}

struct AetherFinish {
    config: AetherConfig,
}

impl AetherFinish {
    fn new(config: AetherConfig) -> Self {
        Self { config }
    }
}

impl PostEffect for AetherFinish {
    fn process(
        &self,
        input: &PixelBuffer,
        width: usize,
        height: usize,
    ) -> std::result::Result<PixelBuffer, crate::post_effects::PostEffectError> {
        let mut buffer = input.clone();
        apply_aether_weave(&mut buffer, width, height, &self.config);
        Ok(buffer)
    }
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
            perceptual_blur_enabled: false,
            perceptual_blur_config: None,
            color_grade_enabled: false,
            color_grade_params: ColorGradeParams::default(),
            gradient_map_enabled: false,
            gradient_map_config: GradientMapConfig::default(),
            champleve_enabled: false,
            champleve_config: ChampleveConfig::default(),
            aether_enabled: false,
            aether_config: AetherConfig::default(),
            chromatic_bloom_enabled: false,
            chromatic_bloom_config: ChromaticBloomConfig::default(),
            opalescence_enabled: false,
            opalescence_config: OpalescenceConfig::default(),
            edge_luminance_enabled: false,
            edge_luminance_config: EdgeLuminanceConfig::default(),
            micro_contrast_enabled: false,
            micro_contrast_config: MicroContrastConfig::default(),
            glow_enhancement_enabled: false,
            glow_enhancement_config: GlowEnhancementConfig::default(),
            atmospheric_depth_enabled: false,
            atmospheric_depth_config: AtmosphericDepthConfig::default(),
            fine_texture_enabled: false,
            fine_texture_config: FineTextureConfig::default(),
            bloom_pyramid_enabled: false,
            anamorphic_flare_enabled: false,
            god_rays_enabled: false,
            star_field_enabled: false,
            star_field_seed: 0,
            diffraction_spikes_enabled: false,
            palette_harmony_enabled: false,
            palette_harmony_seed: 0,
            glaze_enabled: false,
            output_width: 320,
            output_height: 180,
        }
    }

    #[test]
    fn test_finish_pipeline_routes_texture_to_image_stage() {
        let mut config = base_effect_config();
        config.fine_texture_enabled = true;

        let pipeline = FinishEffectPipeline::new(config);

        assert_eq!(pipeline.trajectory_len(), 0);
        assert_eq!(pipeline.image_len(), 1);
    }

    #[test]
    fn test_finish_pipeline_keeps_trajectory_effects_out_of_image_stage() {
        let mut config = base_effect_config();
        config.color_grade_enabled = true;
        config.fine_texture_enabled = true;

        let pipeline = FinishEffectPipeline::new(config);

        assert!(pipeline.trajectory_len() >= 1);
        assert_eq!(pipeline.image_len(), 1);
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
