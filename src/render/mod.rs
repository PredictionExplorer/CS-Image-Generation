//! Rendering module: histogram passes, color mapping, line drawing, and output
//!
//! This module provides a complete rendering pipeline for the three-body problem visualization,
//! including coordinate transformations, line drawing, post-processing effects, and video output.

use crate::post_effects::{
    AetherConfig, AtmosphericDepthConfig, ChampleveConfig, ChromaticBloomConfig,
    EdgeLuminanceConfig, FineTextureConfig, GradientMapConfig, LuxuryPalette, MicroContrastConfig,
    OpalescenceConfig, PerceptualBlurConfig,
};
use crate::spectrum::NUM_BINS;
use crate::utils::f64_to_usize_saturating;
use nalgebra::Vector3;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{debug, info};

/// When true, tonemapping uses the `AgX` punchy output matrix instead of default `AgX`.
pub static ACES_TWEAK_ENABLED: AtomicBool = AtomicBool::new(true);

// Module declarations
pub mod batch_drawing;
pub mod camera_path;
pub mod color;
pub mod composition;
pub mod constants;
pub mod context;
pub mod director;
pub mod drawing;
pub mod effect_randomizer;
pub mod effects;
pub mod error;
pub mod framing_fit;
pub mod hero_outputs;
pub mod histogram;
pub mod mood;
pub mod parameter_descriptors;
pub mod pipeline_flags;
pub mod randomizable_config;
pub mod spectral_output;
pub mod spectral_tonemap;
pub mod trajectory;
pub mod types;
pub mod velocity_hdr;
pub mod video;

// Import from our submodules
use self::batch_drawing::{
    BatchDrawParams, draw_triangle_batch_spectral_rows, lerp_body_step, prepare_triangle_vertices,
    prepare_triangle_vertices_at,
};
use self::context::{PixelBuffer, RenderContext};
use self::drawing::draw_body_spheres_spectral_rows;
use self::effects::{EffectConfig, FinishEffectPipeline, FrameParams, convert_spd_buffer_to_rgba};
use self::error::{RenderError, Result};
use self::histogram::HistogramData;

// Re-export core types and functions for public API compatibility
pub use color::{OklabColor, generate_body_color_sequences};
pub use drawing::{
    LineVertex, SpectralLineSegment, draw_line_segment_aa_spectral, parallel_blur_2d_rgba,
};
pub use effects::{DogBloomConfig, apply_dog_bloom};
pub use types::{ChannelLevels, ToneMappingControls};
pub use video::{VideoEncodingOptions, create_video_from_frames_singlepass};

// Re-export types from dependencies used in public API
pub use image::{DynamicImage, ImageBuffer, Rgb};

/// Which bloom algorithm to apply during post-processing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BloomMode {
    /// Difference-of-Gaussians bloom (sharper, default).
    #[default]
    Dog,
    /// Classical Gaussian blur bloom (softer glow).
    Gaussian,
    /// Bloom disabled.
    None,
}

impl BloomMode {
    /// Parse a bloom mode from a CLI argument string (case-insensitive).
    #[must_use]
    pub fn from_arg(value: &str) -> Self {
        match value {
            v if v.eq_ignore_ascii_case("gaussian") => Self::Gaussian,
            v if v.eq_ignore_ascii_case("none") => Self::None,
            _ => Self::Dog,
        }
    }

    /// Return the canonical lowercase string representation of this bloom mode.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dog => "dog",
            Self::Gaussian => "gaussian",
            Self::None => "none",
        }
    }
}

/// Top-level rendering parameters that apply across the entire pipeline.
#[derive(Clone, Copy, Debug)]
pub struct RenderConfig {
    /// Multiplier applied to spectral accumulation values before tone-mapping.
    pub hdr_scale: f64,
    /// Bloom algorithm selection.
    pub bloom_mode: BloomMode,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self { hdr_scale: constants::DEFAULT_HDR_SCALE, bloom_mode: BloomMode::Dog }
    }
}

/// Whether the finish pipeline targets a single still or a video sequence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FinishOutputMode {
    /// Single accumulated frame (still export and histogram-style passes).
    #[default]
    Still,
    /// Multi-frame output (video encoding and temporal options).
    Video,
}

/// Borrowed view of a scene's trajectory data needed for spectral rendering.
#[derive(Clone, Copy)]
pub struct SpectralScene<'a> {
    /// Per-body position trajectories (indexed `[body][step]`).
    pub positions: &'a [Vec<Vector3<f64>>],
    /// Per-body Oklab colour sequences (indexed `[body][step]`).
    pub colors: &'a [Vec<OklabColor>],
    /// Overall opacity weight for each body's lines.
    pub body_alphas: &'a [f64],
    /// Optional body masses for sphere sizing / physics-aware effects.
    pub masses: Option<&'a [f64; 3]>,
    /// Optional finite-difference kinematics (velocity, curvature, close approaches).
    pub kinematics: Option<&'a crate::kinematics::KinematicTrajectories>,
}

impl std::fmt::Debug for SpectralScene<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpectralScene")
            .field("num_bodies", &self.positions.len())
            .field("num_steps", &self.positions.first().map_or(0, std::vec::Vec::len))
            .field("num_body_alphas", &self.body_alphas.len())
            .field("has_kinematics", &self.kinematics.is_some())
            .finish()
    }
}

impl<'a> SpectralScene<'a> {
    /// Bundle trajectory positions, colours, and alpha weights into a scene view.
    #[must_use]
    pub fn new(
        positions: &'a [Vec<Vector3<f64>>],
        colors: &'a [Vec<OklabColor>],
        body_alphas: &'a [f64],
    ) -> Self {
        Self { positions, colors, body_alphas, masses: None, kinematics: None }
    }

    /// Same as [`Self::new`] but attaches masses and kinematics for Doppler, wakes, and spheres.
    #[must_use]
    pub fn with_kinematics_and_masses(
        positions: &'a [Vec<Vector3<f64>>],
        colors: &'a [Vec<OklabColor>],
        body_alphas: &'a [f64],
        masses: &'a [f64; 3],
        kinematics: &'a crate::kinematics::KinematicTrajectories,
    ) -> Self {
        Self { positions, colors, body_alphas, masses: Some(masses), kinematics: Some(kinematics) }
    }

    /// Number of simulation timesteps recorded for each body.
    #[must_use]
    #[inline]
    pub fn step_count(self) -> usize {
        self.positions[0].len()
    }

    /// Extract the three body alphas into a fixed-size array for triangle drawing.
    #[must_use]
    #[inline]
    pub fn triangle_alphas(self) -> [f64; 3] {
        [self.body_alphas[0], self.body_alphas[1], self.body_alphas[2]]
    }
}

/// Aggregated settings for a spectral render pass (effect config + render config + seeds).
#[derive(Clone, Copy)]
pub struct SpectralRenderSettings<'a> {
    /// Fully-resolved effect parameters (randomised values already picked).
    pub resolved_config: &'a randomizable_config::ResolvedEffectConfig,
    /// Core render parameters (HDR scale, bloom mode).
    pub render_config: &'a RenderConfig,
    /// Seed for procedural noise (star field, palette harmony, textures).
    pub noise_seed: i32,
    /// Whether to correct for non-square pixel aspect ratios.
    pub aspect_correction: bool,
}

impl std::fmt::Debug for SpectralRenderSettings<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpectralRenderSettings")
            .field("render_config", &self.render_config)
            .field("noise_seed", &self.noise_seed)
            .field("aspect_correction", &self.aspect_correction)
            .finish_non_exhaustive()
    }
}

impl<'a> SpectralRenderSettings<'a> {
    /// Bundle all spectral render inputs into a single settings struct.
    #[must_use]
    pub fn new(
        resolved_config: &'a randomizable_config::ResolvedEffectConfig,
        render_config: &'a RenderConfig,
        noise_seed: i32,
        aspect_correction: bool,
    ) -> Self {
        Self { resolved_config, render_config, noise_seed, aspect_correction }
    }
}

#[inline]
fn compress_display_highlights(rgb: [f64; 3], paper_white: f64, rolloff: f64) -> [f64; 3] {
    let luminance = constants::rec709_luminance(rgb[0], rgb[1], rgb[2]);
    if luminance <= paper_white || luminance <= 1e-10 {
        return rgb;
    }

    let shoulder_span = (1.0 - paper_white).max(1e-6);
    let excess = luminance - paper_white;
    let compressed_luminance =
        paper_white + shoulder_span * (1.0 - (-(excess * rolloff) / shoulder_span).exp());
    let scale = compressed_luminance / luminance;

    [rgb[0] * scale, rgb[1] * scale, rgb[2] * scale]
}

/// Which space a buffer lives in when computing stage-luminance
/// statistics. Determines whether the per-pixel luminance needs to be
/// composited against black (pre-tonemap premultiplied-RGBA buffers)
/// or is already display-referred (post-tonemap buffers where the
/// tonemap has already folded alpha).
#[derive(Copy, Clone, Debug)]
enum LuminanceSpace {
    /// Pre-tonemap scene-linear buffer. RGB is un-premultiplied, so the
    /// "visual" luminance is `rec709_luminance(r * a, g * a, b * a)`.
    SceneLinear,
    /// Post-tonemap display-referred buffer. RGB already encodes what
    /// the viewer sees, so `rec709_luminance(r, g, b)` is used directly.
    Display,
}

/// Compute p99 / p99.9 / max / bright-fraction (lum > 0.85) over a
/// pixel buffer. Used by the stage-telemetry probes to pinpoint which
/// part of the render pipeline introduces blow-outs.
fn compute_stage_luminance(
    buffer: &PixelBuffer,
    space: LuminanceSpace,
) -> crate::generation_log::StageLuminance {
    if buffer.is_empty() {
        return crate::generation_log::StageLuminance::default();
    }

    let mut lums: Vec<f64> = buffer
        .par_iter()
        .map(|&(r, g, b, a)| match space {
            LuminanceSpace::SceneLinear => constants::rec709_luminance(r * a, g * a, b * a),
            LuminanceSpace::Display => constants::rec709_luminance(r, g, b),
        })
        .collect();

    let mut max_lum = 0.0f64;
    let mut bright_count: u64 = 0;
    for &l in &lums {
        if l > max_lum {
            max_lum = l;
        }
        if l > 0.85 {
            bright_count += 1;
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let total = lums.len() as f64;
    #[allow(clippy::cast_precision_loss)]
    let bright_fraction = bright_count as f64 / total;

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let p99_idx = ((lums.len() as f64 * 0.99) as usize).min(lums.len() - 1);
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let p99_9_idx = ((lums.len() as f64 * 0.999) as usize).min(lums.len() - 1);

    // `select_nth_unstable` is O(n) and leaves the split invariant we need.
    let (_, p99_ref, _) = lums.select_nth_unstable_by(p99_idx, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let p99 = *p99_ref;

    // Second selection is O(n) and still correct — we only need p99.9 to
    // be correct relative to the full multiset, and `select_nth_unstable`
    // doesn't require a sorted prefix. The earlier call only guaranteed
    // elements < p99 sit to the left, which is compatible with asking for
    // a nth that's further right (p99.9 > p99).
    let (_, p99_9_ref, _) = lums.select_nth_unstable_by(p99_9_idx, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let p99_9 = *p99_9_ref;

    crate::generation_log::StageLuminance { p99, p99_9, max: max_lum, bright_fraction }
}

/// Record per-stage luminance for one of the canonical pipeline stages.
/// Wraps [`compute_stage_luminance`] and
/// [`crate::generation_log::record_stage_luminance`] so call sites stay
/// one-liners.
fn record_stage(label: &str, buffer: &PixelBuffer, space: LuminanceSpace) {
    let stats = compute_stage_luminance(buffer, space);
    crate::generation_log::record_stage_luminance(label, stats);
}

/// Luma-shoulder knee for [`apply_display_headroom`]. Pixels with
/// display-space luminance below this are passed through unchanged,
/// so well-exposed midtones never lose contrast.
const HEADROOM_LUMA_KNEE: f64 = 0.70;
/// Exponential asymptote for the luma shoulder. Highlights above the
/// knee compress toward this value — chosen below
/// [`SAFETY_SHOULDER_KNEE`] so the per-pixel shoulder downstream has
/// nothing left to do on a well-behaved frame.
const HEADROOM_LUMA_ASYMPTOTE: f64 = 0.80;
/// Hue-preserving per-channel cap applied after the luma shoulder.
/// Catches the edge case where a strongly tinted pixel keeps the
/// max channel above 1.0 even though rec709 luminance is compressed
/// (e.g. saturated red: R dominates the value but contributes only
/// 21.3% to rec709 luminance, so the luma shoulder alone isn't
/// enough). The cap is a uniform-scale-by-max, so hue is preserved
/// exactly.
const HEADROOM_CHANNEL_CEILING: f64 = 0.92;
/// Target p99 luminance for the residual-gain fallback. Sits just
/// below [`SAFETY_SHOULDER_KNEE`] (0.90) so the per-pixel shoulder
/// still has room to smooth any residual outliers without engaging
/// on a normal frame.
const HEADROOM_P99_CAP: f64 = 0.85;
/// Maximum fraction of pixels allowed above 0.85 luminance before
/// the residual uniform gain engages. Calibrated from a healthy
/// reference run (`0xe7dcb1f0ff4e`: 0.0% bright) vs the failing
/// seeds (17-30% bright) — 5% leaves room for a bright-but-not-
/// blown-out hero moment.
const HEADROOM_BRIGHT_CAP: f64 = 0.05;
/// Hard floor on the residual uniform gain so the headroom guard
/// never produces the "flat grey blob" failure mode seen with the
/// legacy implementation (`0x3d8c8c21b240` was dimmed to p99 = 0.38).
/// If stages 1 and 2 cannot bring the distribution under the caps,
/// the safety shoulder downstream picks up the remaining highlights;
/// midtones must still reach the viewer.
const HEADROOM_RESIDUAL_GAIN_FLOOR: f64 = 0.80;

/// Hue-preserving display-space highlight compressor used by
/// [`apply_display_headroom`]. Mirrors
/// [`compress_display_safety_shoulder`] in shape but engages at a
/// lower knee and asymptote so it can absorb a much larger bright
/// tail without flattening midtone contrast.
#[inline]
fn compress_headroom_highlights(rgb: [f64; 3], knee: f64, asymptote: f64, ceiling: f64) -> [f64; 3] {
    let mut out = rgb;

    // Stage 1: luma-based shoulder. Any above-knee luminance is
    // rolled onto an exponential asymptote at `asymptote`, with every
    // channel scaled uniformly so hue is preserved exactly.
    let luminance = constants::rec709_luminance(out[0], out[1], out[2]);
    if luminance > knee && luminance > 1e-10 {
        let span = (asymptote - knee).max(1e-6);
        let excess = luminance - knee;
        let compressed = knee + span * (1.0 - (-excess / span).exp());
        let scale = compressed / luminance;
        out = [out[0] * scale, out[1] * scale, out[2] * scale];
    }

    // Stage 2: per-channel gamut preservation. A pixel can have
    // luminance below `knee` but still hold a single channel above
    // `ceiling` — e.g. a saturated red where R dominates value but
    // contributes only 21.3% to rec709 luminance. Hard per-channel
    // clamping would break hue; instead, find the max channel and
    // scale *all* channels uniformly so the max lands on `ceiling`,
    // preserving the R:G:B ratios exactly.
    let max_ch = out[0].max(out[1]).max(out[2]);
    if max_ch > ceiling {
        let scale = ceiling / max_ch;
        out[0] *= scale;
        out[1] *= scale;
        out[2] *= scale;
    }
    out
}

/// Distribution-aware display-space headroom cap applied between
/// `process_image` (the image chain: diffraction spikes, anamorphic
/// flare, fine texture, vignette, ...) and
/// [`apply_display_safety_shoulder`] (a per-pixel hue-preserving
/// roll-off at 0.90-0.985).
///
/// # Why three stages
///
/// The image chain can push pixels past 1.0 in display space on
/// dense bright regions (seeds `0x3d8c8c21b240`, `0x4d5af082584d`,
/// `0x8df990f92766`, ...). The legacy implementation responded with
/// a uniform RGB dim scaled by `sqrt(BRIGHT_CAP / bright_fraction)`.
/// For a 30% bright blob that is a ~0.38× dim — correct in the sense
/// of bringing the distribution under the caps, but *catastrophic*
/// for perceptual quality because midtone contrast is compressed too.
/// The resulting image looks like a flat grey blob, which is what the
/// original "white blob" reports identified on the remote server.
///
/// The fix uses three stages:
///
/// 1. **Hue-preserving highlight shoulder** (`compress_headroom_highlights`):
///    pixels with luminance in `[HEADROOM_LUMA_KNEE, ∞)` are rolled
///    onto an exponential asymptote at `HEADROOM_LUMA_ASYMPTOTE` with
///    a uniform-scale-by-max per-channel cap at
///    `HEADROOM_CHANNEL_CEILING`. Midtones (luma < knee) pass through
///    bit-for-bit so contrast is preserved.
/// 2. **Residual uniform gain**, but only if stage 1 left the
///    distribution out of spec *and* the required gain does not drop
///    below `HEADROOM_RESIDUAL_GAIN_FLOOR`. The floor prevents the
///    old flat-grey regression.
/// 3. Implicitly, the downstream [`apply_display_safety_shoulder`]
///    still catches any remaining outliers in `[SAFETY_SHOULDER_KNEE,
///    SAFETY_SHOULDER_CEILING]`.
///
/// On a healthy reference buffer (luma ≤ knee everywhere) this
/// function is a bit-for-bit no-op: neither stage fires.
fn apply_display_headroom(buffer: &mut PixelBuffer) {
    if buffer.is_empty() {
        return;
    }

    // Stage 1: hue-preserving highlight shoulder. Applied per-pixel,
    // so midtones below HEADROOM_LUMA_KNEE are guaranteed unchanged.
    buffer.par_iter_mut().for_each(|pixel| {
        let rolled = compress_headroom_highlights(
            [pixel.0, pixel.1, pixel.2],
            HEADROOM_LUMA_KNEE,
            HEADROOM_LUMA_ASYMPTOTE,
            HEADROOM_CHANNEL_CEILING,
        );
        pixel.0 = rolled[0];
        pixel.1 = rolled[1];
        pixel.2 = rolled[2];
    });

    // Stage 2: residual uniform gain (only if stage 1 left the
    // distribution out of spec). Measure post-shoulder distribution.
    let lums: Vec<f64> =
        buffer.par_iter().map(|&(r, g, b, _)| constants::rec709_luminance(r, g, b)).collect();

    let bright_count = lums.iter().filter(|&&l| l > HEADROOM_P99_CAP).count();
    #[allow(clippy::cast_precision_loss)]
    let bright_fraction = bright_count as f64 / lums.len() as f64;

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let p99_idx = ((lums.len() as f64 * 0.99) as usize).min(lums.len() - 1);
    let mut tmp = lums;
    let (_, p99_ref, _) = tmp.select_nth_unstable_by(p99_idx, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let p99 = *p99_ref;

    let gain_p99 = if p99 > HEADROOM_P99_CAP { HEADROOM_P99_CAP / p99 } else { 1.0 };
    let gain_bright = if bright_fraction > HEADROOM_BRIGHT_CAP {
        (HEADROOM_BRIGHT_CAP / bright_fraction.max(1e-6)).sqrt()
    } else {
        1.0
    };
    // Clamp to the floor: if stage 1 cannot bring the distribution
    // under the caps by itself, we accept a slight residual bright
    // tail rather than collapsing midtones. The safety shoulder
    // downstream will pick up the rest.
    let gain = gain_p99.min(gain_bright).max(HEADROOM_RESIDUAL_GAIN_FLOOR);

    if gain < 1.0 - 1e-6 {
        buffer.par_iter_mut().for_each(|p| {
            p.0 *= gain;
            p.1 *= gain;
            p.2 *= gain;
        });
    }
}

/// Final display-space safety shoulder applied after `process_image` and
/// before the 16-bit quantize clamp.
///
/// `compress_display_highlights` asymptotes at 1.0 because its shoulder
/// span is `1.0 - paper_white`; the image-stage effects (diffraction
/// spikes, anamorphic lens flare, fine texture) then run **after** tonemap
/// and add additively on top of a buffer that's already near 1.0, which
/// [`quantize_display_buffer_to_16bit`] hard-clamps to 1.0 per channel.
/// That per-channel clamp destroys hue (one channel saturates while the
/// others lag) — it's the source of both the pure-white cores and the
/// primary-colour splotches in the blown-out renders.
///
/// This helper rolls off any luminance above `knee` onto an exponential
/// shoulder that asymptotes at a strict `ceiling < 1.0`, then soft-limits
/// any residual per-channel overshoot. Pixels below the knee pass through
/// untouched, so well-behaved seeds are unaffected.
#[inline]
fn compress_display_safety_shoulder(rgb: [f64; 3], knee: f64, ceiling: f64) -> [f64; 3] {
    // Stage 1: luma-based shoulder. Any above-knee luminance is rolled off
    // onto an exponential asymptote at `ceiling`, with every channel scaled
    // by the same factor so hue is preserved exactly.
    let mut out = rgb;
    let luminance = constants::rec709_luminance(out[0], out[1], out[2]);
    if luminance > knee && luminance > 1e-10 {
        let span = (ceiling - knee).max(1e-6);
        let excess = luminance - knee;
        let compressed = knee + span * (1.0 - (-excess / span).exp());
        let scale = compressed / luminance;
        out = [out[0] * scale, out[1] * scale, out[2] * scale];
    }

    // Stage 2: per-channel gamut preservation. A pixel can have luminance
    // below `knee` but still hold a single channel above `ceiling` — e.g. a
    // nearly-pure-red highlight where R is hot but G and B are near zero,
    // so rec709 luminance never clears the knee. Hard per-channel clamping
    // would break hue (R gets crushed while G/B hold). Instead, find the
    // max channel and scale *all* channels uniformly so the max lands on
    // `ceiling`, preserving the R:G:B ratios exactly.
    let max_ch = out[0].max(out[1]).max(out[2]);
    if max_ch > ceiling {
        let scale = ceiling / max_ch;
        out[0] *= scale;
        out[1] *= scale;
        out[2] *= scale;
    }
    out
}

/// Luminance below `SAFETY_SHOULDER_KNEE` is a no-op on the luma axis, so
/// well-behaved pixels pass through the safety shoulder unchanged.
const SAFETY_SHOULDER_KNEE: f64 = 0.90;
/// Strict upper bound on post-shoulder channel values. Leaves 1.5% headroom
/// below the 16-bit quantize max, so the hard `.clamp(0.0, 1.0)` in
/// [`quantize_display_buffer_to_16bit`] can never engage.
const SAFETY_SHOULDER_CEILING: f64 = 0.985;

/// Apply [`compress_display_safety_shoulder`] across the full display
/// buffer. Operates in-place and preserves alpha.
fn apply_display_safety_shoulder(buffer: &mut PixelBuffer) {
    buffer.par_iter_mut().for_each(|pixel| {
        let rolled = compress_display_safety_shoulder(
            [pixel.0, pixel.1, pixel.2],
            SAFETY_SHOULDER_KNEE,
            SAFETY_SHOULDER_CEILING,
        );
        pixel.0 = rolled[0];
        pixel.1 = rolled[1];
        pixel.2 = rolled[2];
    });
}

/// Core tonemapping function (shared logic for both 8-bit and 16-bit).
///
/// Runs `AgX` per channel (same polynomial spline as before) but replaces
/// the legacy hard per-channel `clamp(0, 1)` with a hue-preserving `OKLab`
/// gamut map. Extreme highlights no longer collapse to pure white: their
/// `OKLab` hue and lightness are preserved while chroma is reduced until
/// the result fits inside the `sRGB` cube.
#[inline]
fn tonemap_core(fr: f64, fg: f64, fb: f64, fa: f64, levels: &ChannelLevels) -> [f64; 3] {
    let alpha = fa.clamp(0.0, 1.0);
    if alpha <= 0.0 {
        return [0.0, 0.0, 0.0];
    }

    let source = [fr.max(0.0), fg.max(0.0), fb.max(0.0)];
    let premult = [source[0] * alpha, source[1] * alpha, source[2] * alpha];
    if premult[0] <= 0.0 && premult[1] <= 0.0 && premult[2] <= 0.0 {
        return [0.0, 0.0, 0.0];
    }

    let mut leveled = [0.0; 3];
    for i in 0..3 {
        leveled[i] =
            (((premult[i] - levels.black[i]).max(0.0)) / levels.range[i]) * levels.exposure_scale;
    }

    // 0. Matrix Inset (AgX color space)
    let r = leveled[0];
    let g = leveled[1];
    let b = leveled[2];

    let r_in = 0.842479062253094 * r + 0.0784335999999992 * g + 0.0792237451477643 * b;
    let g_in = 0.0423282422610123 * r + 0.878468636469772 * g + 0.0791661274605434 * b;
    let b_in = 0.0423756549057051 * r + 0.0784336000000000 * g + 0.877456439033405 * b;

    // 1. Log2 allocation
    let min_ev = -10.0;
    let max_ev = 2.5;
    let range = max_ev - min_ev;

    let allocate = |v: f64| -> f64 {
        let val = v.max(1e-10).log2();
        ((val - min_ev) / range).clamp(0.0, 1.0)
    };

    let r_alloc = allocate(r_in);
    let g_alloc = allocate(g_in);
    let b_alloc = allocate(b_in);

    // 2. AgX base spline (same polynomial as before).
    let spline = |x: f64| -> f64 {
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x2 * x2;
        let x5 = x4 * x;
        let x6 = x5 * x;
        12.0625 * x6 - 36.3262 * x5 + 39.5298 * x4 - 17.6534 * x3 + 3.0135 * x2 + 0.3707 * x
    };

    let r_spline = spline(r_alloc);
    let g_spline = spline(g_alloc);
    let b_spline = spline(b_alloc);

    // 3. Matrix outset (AgX Punchy when ACES_TWEAK_ENABLED, else default).
    let is_punchy = ACES_TWEAK_ENABLED.load(Ordering::Relaxed);

    let (r_out, g_out, b_out) = if is_punchy {
        (
            1.133276 * r_spline - 0.117109 * g_spline - 0.016167 * b_spline,
            -0.097008 * r_spline + 1.148151 * g_spline - 0.051143 * b_spline,
            -0.008107 * r_spline - 0.031776 * g_spline + 1.039883 * b_spline,
        )
    } else {
        (
            1.0987524 * r_spline - 0.0880758 * g_spline - 0.0106766 * b_spline,
            -0.0729567 * r_spline + 1.1114562 * g_spline - 0.0384995 * b_spline,
            -0.0060957 * r_spline - 0.0238959 * g_spline + 1.0299916 * b_spline,
        )
    };

    // 4. Hue-preserving gamut map instead of per-channel clamping. Values
    // that would have clipped to `(1.0, 1.0, 1.0)` now retain their OKLab
    // hue while chroma is softly reduced until the RGB point fits inside
    // [0, 1]^3, so hot cores stay coloured instead of collapsing to white.
    let gamut_mapped =
        spectral_tonemap::gamut_map_preserve_hue([r_out.max(0.0), g_out.max(0.0), b_out.max(0.0)]);

    let compressed =
        compress_display_highlights(gamut_mapped, levels.paper_white, levels.highlight_rolloff);

    [compressed[0].clamp(0.0, 1.0), compressed[1].clamp(0.0, 1.0), compressed[2].clamp(0.0, 1.0)]
}

/// Tonemap to 16-bit (primary output format for maximum precision)
#[cfg(test)]
#[inline]
fn tonemap_to_16bit(fr: f64, fg: f64, fb: f64, fa: f64, levels: &ChannelLevels) -> [u16; 3] {
    let channels = tonemap_core(fr, fg, fb, fa, levels);
    [
        crate::utils::f64_to_u16_saturating((channels[0] * constants::U16_MAX_F64).round()),
        crate::utils::f64_to_u16_saturating((channels[1] * constants::U16_MAX_F64).round()),
        crate::utils::f64_to_u16_saturating((channels[2] * constants::U16_MAX_F64).round()),
    ]
}

/// Save a 16-bit RGB image as a PNG with an explicit sRGB chunk so colour-managed
/// viewers honour the colour space instead of assuming it.
///
/// The `image` crate encoder omits the `sRGB`/`gAMA`/`cHRM` chunks; we go through the
/// `png` crate directly to emit a minimal set that all compliant PNG readers understand.
pub fn save_image_as_png_16bit(
    rgb_img: &ImageBuffer<Rgb<u16>, Vec<u16>>,
    path: &str,
) -> Result<()> {
    use std::fs::File;
    use std::io::BufWriter;

    let width = rgb_img.width();
    let height = rgb_img.height();
    if width == 0 || height == 0 {
        return Err(RenderError::ImageEncoding {
            reason: format!("cannot encode zero-sized image {width}x{height}"),
        });
    }

    let file = File::create(path)
        .map_err(|e| RenderError::ImageEncoding { reason: format!("creating {path}: {e}") })?;
    let writer = BufWriter::new(file);

    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Sixteen);
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    let mut png_writer = encoder
        .write_header()
        .map_err(|e| RenderError::ImageEncoding { reason: format!("PNG header: {e}") })?;

    // Convert native-endian u16 samples to big-endian as required by PNG.
    let raw: &[u16] = rgb_img.as_raw();
    let mut be_bytes = Vec::with_capacity(raw.len() * 2);
    for sample in raw {
        be_bytes.extend_from_slice(&sample.to_be_bytes());
    }
    png_writer
        .write_image_data(&be_bytes)
        .map_err(|e| RenderError::ImageEncoding { reason: format!("PNG data: {e}") })?;
    png_writer
        .finish()
        .map_err(|e| RenderError::ImageEncoding { reason: format!("PNG finalize: {e}") })?;

    info!("   Saved 16-bit PNG (sRGB-tagged) => {path}");
    Ok(())
}

fn tonemap_to_display_buffer(pixels: &PixelBuffer, levels: &ChannelLevels) -> PixelBuffer {
    pixels
        .par_iter()
        .map(|&(fr, fg, fb, fa)| {
            let mapped = tonemap_core(fr, fg, fb, fa, levels);
            (mapped[0], mapped[1], mapped[2], fa.clamp(0.0, 1.0))
        })
        .collect()
}

fn quantize_display_buffer_to_16bit(pixels: &PixelBuffer) -> Vec<u16> {
    let mut buf_16bit = vec![0u16; pixels.len() * 3];
    buf_16bit.par_chunks_mut(3).zip(pixels.par_iter()).for_each(|(chunk, &(r, g, b, _a))| {
        chunk[0] = (r.clamp(0.0, 1.0) * constants::U16_MAX_F64).round() as u16;
        chunk[1] = (g.clamp(0.0, 1.0) * constants::U16_MAX_F64).round() as u16;
        chunk[2] = (b.clamp(0.0, 1.0) * constants::U16_MAX_F64).round() as u16;
    });
    buf_16bit
}

// ====================== HELPER FUNCTIONS ===========================

/// Derive the perceptual-blur radius (in pixels) after accounting for the combined
/// softness of all enabled blur/bloom effects. Returns `None` when blur is disabled.
#[must_use]
pub fn compute_softness_radius(
    resolved: &randomizable_config::ResolvedEffectConfig,
    bloom_mode: BloomMode,
) -> Option<usize> {
    if !resolved.enable_perceptual_blur {
        return None;
    }

    let use_gaussian_bloom = bloom_mode == BloomMode::Gaussian && resolved.enable_bloom;
    let softness_stack_score = (if use_gaussian_bloom { 1.0 } else { 0.0 })
        + if resolved.enable_chromatic_bloom { 0.8 } else { 0.0 }
        + if resolved.enable_perceptual_blur { 0.85 } else { 0.0 }
        + if resolved.enable_glow { 0.55 } else { 0.0 }
        + if resolved.enable_atmospheric_depth { 0.35 } else { 0.0 };
    let radius_scale = if softness_stack_score >= 2.0 { 0.0030 } else { 0.0036 };
    let min_dim = resolved.width.min(resolved.height);

    Some(f64_to_usize_saturating((radius_scale * f64::from(min_dim)).round().max(1.0)))
}

fn build_dog_config(
    resolved: &randomizable_config::ResolvedEffectConfig,
    min_dim: usize,
) -> DogBloomConfig {
    let dog_inner_sigma = resolved.dog_sigma_scale * min_dim as f64;
    let dog_threshold = (0.012_f64
        + if resolved.enable_glow { 0.003_f64 } else { 0.0 }
        + if resolved.enable_chromatic_bloom { 0.004_f64 } else { 0.0 }
        + if resolved.enable_perceptual_blur { 0.004_f64 } else { 0.0 })
    .min(0.028_f64);

    DogBloomConfig {
        inner_sigma: dog_inner_sigma,
        outer_ratio: resolved.dog_ratio,
        strength: resolved.dog_strength,
        threshold: dog_threshold,
    }
}

fn build_perceptual_blur_config(
    resolved: &randomizable_config::ResolvedEffectConfig,
    bloom_mode: BloomMode,
) -> Option<PerceptualBlurConfig> {
    use crate::oklab::GamutMapMode;
    compute_softness_radius(resolved, bloom_mode).map(|radius| PerceptualBlurConfig {
        radius,
        strength: resolved.perceptual_blur_strength,
        gamut_mode: GamutMapMode::PreserveHue,
    })
}

fn build_chromatic_bloom_config(
    resolved: &randomizable_config::ResolvedEffectConfig,
    min_dim: usize,
) -> ChromaticBloomConfig {
    let radius =
        f64_to_usize_saturating((resolved.chromatic_bloom_radius_scale * min_dim as f64).round());
    let separation = resolved.chromatic_bloom_separation_scale * min_dim as f64;
    ChromaticBloomConfig {
        radius,
        strength: resolved.chromatic_bloom_strength,
        separation,
        threshold: resolved.chromatic_bloom_threshold,
    }
}

fn build_color_grade_params(
    resolved: &randomizable_config::ResolvedEffectConfig,
    min_dim: usize,
) -> crate::post_effects::ColorGradeParams {
    crate::post_effects::ColorGradeParams {
        strength: resolved.color_grade_strength,
        vignette_strength: resolved.vignette_strength,
        vignette_softness: resolved.vignette_softness,
        vibrance: resolved.vibrance,
        clarity_strength: resolved.clarity_strength,
        clarity_radius: (0.0028 * min_dim as f64).round().max(1.0) as usize,
        tone_curve: resolved.tone_curve_strength,
        shadow_tint: constants::DEFAULT_COLOR_GRADE_SHADOW_TINT,
        highlight_tint: constants::DEFAULT_COLOR_GRADE_HIGHLIGHT_TINT,
        palette_wave_strength: 0.25,
    }
}

fn build_glow_config(
    resolved: &randomizable_config::ResolvedEffectConfig,
    min_dim: usize,
) -> crate::post_effects::GlowEnhancementConfig {
    let glow_radius = (resolved.glow_radius_scale * min_dim as f64).round() as usize;
    crate::post_effects::GlowEnhancementConfig {
        strength: resolved.glow_strength,
        threshold: resolved.glow_threshold,
        radius: glow_radius,
        sharpness: resolved.glow_sharpness,
        saturation_boost: resolved.glow_saturation_boost,
    }
}

fn build_champleve_config(resolved: &randomizable_config::ResolvedEffectConfig) -> ChampleveConfig {
    ChampleveConfig {
        cell_density: constants::DEFAULT_CHAMPLEVE_CELL_DENSITY,
        flow_alignment: resolved.champleve_flow_alignment,
        interference_amplitude: resolved.champleve_interference_amplitude,
        interference_frequency: constants::DEFAULT_CHAMPLEVE_INTERFERENCE_FREQUENCY,
        rim_intensity: resolved.champleve_rim_intensity,
        rim_warmth: resolved.champleve_rim_warmth,
        rim_sharpness: constants::DEFAULT_CHAMPLEVE_RIM_SHARPNESS,
        interior_lift: resolved.champleve_interior_lift,
        anisotropy: constants::DEFAULT_CHAMPLEVE_ANISOTROPY,
        cell_softness: constants::DEFAULT_CHAMPLEVE_CELL_SOFTNESS,
    }
}

fn build_aether_config(resolved: &randomizable_config::ResolvedEffectConfig) -> AetherConfig {
    AetherConfig {
        filament_density: constants::DEFAULT_AETHER_FILAMENT_DENSITY,
        flow_alignment: resolved.aether_flow_alignment,
        scattering_strength: resolved.aether_scattering_strength,
        scattering_falloff: constants::DEFAULT_AETHER_SCATTERING_FALLOFF,
        iridescence_amplitude: resolved.aether_iridescence_amplitude,
        iridescence_frequency: constants::DEFAULT_AETHER_IRIDESCENCE_FREQUENCY,
        caustic_strength: resolved.aether_caustic_strength,
        caustic_softness: constants::DEFAULT_AETHER_CAUSTIC_SOFTNESS,
        luxury_mode: true,
    }
}

fn build_opalescence_config(
    resolved: &randomizable_config::ResolvedEffectConfig,
    width: usize,
    height: usize,
) -> OpalescenceConfig {
    let scale_abs = resolved.opalescence_scale * ((width * height) as f64).sqrt();
    OpalescenceConfig {
        strength: resolved.opalescence_strength,
        scale: scale_abs,
        layers: resolved.opalescence_layers,
        chromatic_shift: 0.5,
        angle_sensitivity: 0.8,
        pearl_sheen: 0.3,
    }
}

fn build_edge_luminance_config(
    resolved: &randomizable_config::ResolvedEffectConfig,
) -> EdgeLuminanceConfig {
    EdgeLuminanceConfig {
        strength: resolved.edge_luminance_strength,
        threshold: resolved.edge_luminance_threshold,
        brightness_boost: resolved.edge_luminance_brightness_boost,
        bright_edges_only: true,
        min_luminance: 0.2,
    }
}

fn build_micro_contrast_config(
    resolved: &randomizable_config::ResolvedEffectConfig,
) -> MicroContrastConfig {
    MicroContrastConfig {
        strength: resolved.micro_contrast_strength,
        radius: resolved.micro_contrast_radius,
        edge_threshold: 0.15,
        luminance_weight: 0.7,
    }
}

fn build_atmospheric_depth_config(
    resolved: &randomizable_config::ResolvedEffectConfig,
) -> AtmosphericDepthConfig {
    AtmosphericDepthConfig {
        strength: resolved.atmospheric_depth_strength,
        fog_color: (
            resolved.atmospheric_fog_color_r,
            resolved.atmospheric_fog_color_g,
            resolved.atmospheric_fog_color_b,
        ),
        density_threshold: 0.15,
        desaturation: resolved.atmospheric_desaturation,
        darkening: resolved.atmospheric_darkening,
        density_radius: 3,
    }
}

fn build_fine_texture_config(
    resolved: &randomizable_config::ResolvedEffectConfig,
    output_mode: FinishOutputMode,
) -> (bool, FineTextureConfig) {
    let width = resolved.width as usize;
    let height = resolved.height as usize;
    let scale_abs = resolved.fine_texture_scale * ((width * height) as f64).sqrt();
    let min_dim = resolved.width.min(resolved.height);
    let enabled = resolved.enable_fine_texture && min_dim >= 720;
    let strength_scale = if output_mode == FinishOutputMode::Video { 0.6 } else { 1.0 };
    (
        enabled,
        FineTextureConfig {
            strength: resolved.fine_texture_strength * strength_scale,
            scale: scale_abs,
            contrast: resolved.fine_texture_contrast,
            anisotropy: 0.3,
            angle: 0.0,
        },
    )
}

/// Build a fully populated [`EffectConfig`] from resolved parameters and render settings.
#[must_use]
pub fn build_effect_config_from_resolved(
    resolved: &randomizable_config::ResolvedEffectConfig,
    render_config: &RenderConfig,
    output_mode: FinishOutputMode,
    noise_seed: i32,
) -> EffectConfig {
    let width = resolved.width as usize;
    let height = resolved.height as usize;
    let min_dim = width.min(height);

    let use_gaussian_bloom =
        resolved.enable_bloom && matches!(render_config.bloom_mode, BloomMode::Gaussian);
    let use_dog_bloom = resolved.enable_bloom && matches!(render_config.bloom_mode, BloomMode::Dog);

    let blur_radius_px = if use_gaussian_bloom {
        (resolved.blur_radius_scale * min_dim as f64).round() as usize
    } else {
        0
    };
    let (fine_texture_enabled, fine_texture_config) =
        build_fine_texture_config(resolved, output_mode);

    EffectConfig {
        bloom_mode: if use_dog_bloom {
            BloomMode::Dog.as_str().to_string()
        } else if use_gaussian_bloom {
            BloomMode::Gaussian.as_str().to_string()
        } else {
            BloomMode::None.as_str().to_string()
        },
        blur_radius_px,
        blur_strength: resolved.blur_strength,
        blur_core_brightness: resolved.blur_core_brightness,
        dog_config: build_dog_config(resolved, min_dim),
        perceptual_blur_enabled: resolved.enable_perceptual_blur,
        perceptual_blur_config: build_perceptual_blur_config(resolved, render_config.bloom_mode),

        color_grade_enabled: resolved.enable_color_grade,
        color_grade_params: build_color_grade_params(resolved, min_dim),
        gradient_map_enabled: resolved.enable_gradient_map,
        gradient_map_config: GradientMapConfig {
            palette: LuxuryPalette::from_index(resolved.gradient_map_palette),
            strength: resolved.gradient_map_strength,
            hue_preservation: resolved.gradient_map_hue_preservation,
        },

        champleve_enabled: resolved.enable_champleve,
        champleve_config: build_champleve_config(resolved),
        aether_enabled: resolved.enable_aether,
        aether_config: build_aether_config(resolved),
        chromatic_bloom_enabled: resolved.enable_chromatic_bloom,
        chromatic_bloom_config: build_chromatic_bloom_config(resolved, min_dim),
        opalescence_enabled: resolved.enable_opalescence,
        opalescence_config: build_opalescence_config(resolved, width, height),

        edge_luminance_enabled: resolved.enable_edge_luminance,
        edge_luminance_config: build_edge_luminance_config(resolved),
        micro_contrast_enabled: resolved.enable_micro_contrast,
        micro_contrast_config: build_micro_contrast_config(resolved),
        glow_enhancement_enabled: resolved.enable_glow,
        glow_enhancement_config: build_glow_config(resolved, min_dim),

        atmospheric_depth_enabled: resolved.enable_atmospheric_depth,
        atmospheric_depth_config: build_atmospheric_depth_config(resolved),
        fine_texture_enabled,
        fine_texture_config,

        bloom_pyramid_enabled: resolved.enable_bloom_pyramid,
        anamorphic_flare_enabled: resolved.enable_anamorphic_flare,
        god_rays_enabled: resolved.enable_god_rays,
        star_field_enabled: resolved.enable_star_field,
        star_field_seed: noise_seed,
        diffraction_spikes_enabled: resolved.enable_diffraction_spikes,
        palette_harmony_enabled: resolved.enable_palette_harmony,
        palette_harmony_seed: noise_seed.rotate_left(7),
        glaze_enabled: resolved.enable_glaze,
        output_width: width,
        output_height: height,
    }
}

/// Gentle scene-linear exposure normalization before temporal blend / tonemap.
///
/// Four guards stack to shape the exposure:
/// 1. A **mean-luminance** normalization pushes the alpha-weighted average
///    toward `target = 0.14`, clamped to `[0.25, 2.5]` so single-frame
///    exposure swings stay bounded.
/// 2. A **99th-percentile highlight headroom** cap keeps the bulk of the
///    bright tail below `HIGHLIGHT_HEADROOM_P99` (well under the tonemap
///    shoulder).
/// 3. A **99.5th-percentile soft cap** at `HIGHLIGHT_HEADROOM_P995` — this
///    is the middle ground between p99 (broad bright tail) and p99.9
///    (extreme peaks). Without it, the top 0.5% (denser than p99.9,
///    still hot relative to p99) could escape both bookends when
///    additive stacks pile up.
/// 4. A **99.9th-percentile** ceiling at `HIGHLIGHT_HEADROOM_P999` that
///    pins the extreme peak below `AgX`'s `max_ev = 2.5` saturation so
///    compact bright cores cannot collapse to display white.
///
/// Each cap is only applied when it actually tightens (not loosens) the
/// auto gain, so underexposed scenes still get the full `TARGET_MEAN`
/// boost.
fn apply_auto_scene_metering(buffer: &mut PixelBuffer) {
    /// Target alpha-weighted mean luminance (middle grey).
    const TARGET_MEAN: f64 = 0.14;
    /// 99th-percentile luminance ceiling after auto gain when the
    /// additive stack is light (≤ 4 HDR-brightening effects). Sits
    /// well below the tonemap's `paper_white` (0.85) so the `AgX`
    /// shoulder has room to shape highlights without the top 1 % of
    /// inked pixels already sitting against the hard upper bound.
    const HIGHLIGHT_HEADROOM_P99: f64 = 0.65;
    /// Tighter 99th-percentile cap activated when the additive
    /// stack reaches 5+ effects. The stacked chain (bloom + god rays +
    /// chromatic + aether + champleve + ...) interacts with the
    /// post-tonemap image chain (diffraction spikes, anamorphic flare,
    /// fine texture) which can re-brighten pixels in display space —
    /// reserving an extra 0.10 of display-space headroom upstream
    /// keeps `apply_display_headroom` from having to do a heavy uniform
    /// dim later.
    const HIGHLIGHT_HEADROOM_P99_HEAVY: f64 = 0.55;
    /// Threshold at which the p99 cap tightens from
    /// `HIGHLIGHT_HEADROOM_P99` to `HIGHLIGHT_HEADROOM_P99_HEAVY`.
    /// Mirrors the Guard 6 "high count" threshold so metering and
    /// budget pressures activate in lock-step.
    const HEAVY_STACK_THRESHOLD: u32 = 5;
    /// 99.5th-percentile soft cap. Sits above `paper_white` (0.85) but
    /// well below `AgX`'s `max_ev = 2.5` clip, giving the top 0.5 % of
    /// bright pixels explicit headroom between the p99 bulk and the
    /// p99.9 extreme tail. Without it, moderately-hot sub-regions
    /// (e.g. rim-lit arcs of a chromatic-bloom + god-rays stack) can
    /// land in the p99->p99.9 gap and still punch past the tonemap
    /// shoulder.
    const HIGHLIGHT_HEADROOM_P995: f64 = 0.85;
    /// 99.9th-percentile ceiling. The p99 cap on its own leaves the
    /// extreme top 0.1 % (compact bright cores stacked with additive
    /// bloom + god rays + chromatic + aether) free to clip past `AgX`'s
    /// `max_ev = 2.5` and collapse to display white. This ceiling
    /// pins that tail at `1.2` in scene-linear space so the `AgX`
    /// allocation still has room to compress instead of saturate.
    const HIGHLIGHT_HEADROOM_P999: f64 = 1.2;
    /// Stride used for percentile sampling. Tightened from 20 to 10 so
    /// the p99.9 tail has ~2x as many samples (~200 on a 1920x1080
    /// frame vs ~100), cutting tail-percentile sampling variance
    /// roughly in half while staying O(n/stride).
    const PERCENTILE_STRIDE: usize = 10;

    let mut sum = 0.0f64;
    let mut w = 0.0f64;
    // Alpha-normalised luminance samples used for the highlight percentile.
    let mut samples: Vec<f64> = Vec::with_capacity(buffer.len() / PERCENTILE_STRIDE + 1);
    for (i, &(r, g, b, a)) in buffer.iter().enumerate() {
        if a > 1e-9 {
            let inv_a = 1.0 / a;
            let lr = r * inv_a;
            let lg = g * inv_a;
            let lb = b * inv_a;
            let lum = constants::rec709_luminance(lr, lg, lb);
            sum += lum * a;
            w += a;
            if i % PERCENTILE_STRIDE == 0 && lum.is_finite() {
                samples.push(lum);
            }
        }
    }
    let mean = if w > 0.0 { sum / w } else { TARGET_MEAN };
    // Upper bound tightened from 6.0 to 2.5: the previous value could
    // boost a dim scene ~6x and drive the additive post-effect peaks
    // far above the tonemap's headroom. 2.5x is still a meaningful
    // lift for legitimately underexposed renders without risking
    // clipping on top.
    let mut auto = (TARGET_MEAN / mean.max(1e-6)).clamp(0.25, 2.5);

    // Highlight headroom: cap `auto` so the post-gain percentile
    // luminances don't exceed their respective headroom values. Each
    // cap is only applied when it actually tightens (not loosens) the
    // auto term.
    if !samples.is_empty() {
        let n = samples.len();
        // Partial-sort enough of the tail to read p99, p99.5, and p99.9.
        let p99_idx = ((n as f64) * 0.99).floor() as usize;
        let p995_idx = ((n as f64) * 0.995).floor() as usize;
        let p999_idx = ((n as f64) * 0.999).floor() as usize;
        let p99_idx = p99_idx.min(n - 1);
        let p995_idx = p995_idx.min(n - 1);
        let p999_idx = p999_idx.min(n - 1);

        // `select_nth_unstable_by` partitions so everything < position
        // `p999_idx` lies in the lower sub-slice. Partition from the
        // top down so each subsequent select operates on a shorter,
        // already-partitioned prefix — this is strictly cheaper than
        // sorting the whole sample array.
        samples.select_nth_unstable_by(p999_idx, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let p999 = samples[p999_idx].max(1e-6);

        let p995_slice = &mut samples[..=p999_idx];
        let p995_idx_in_slice = p995_idx.min(p995_slice.len() - 1);
        p995_slice.select_nth_unstable_by(p995_idx_in_slice, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let p995 = p995_slice[p995_idx_in_slice].max(1e-6);

        let p99_slice = &mut samples[..=p995_idx];
        let p99_idx_in_slice = p99_idx.min(p99_slice.len() - 1);
        p99_slice.select_nth_unstable_by(p99_idx_in_slice, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let p99 = p99_slice[p99_idx_in_slice].max(1e-6);

        // Stack-aware p99 cap: a heavy additive stack reserves more
        // display-space headroom upstream so post-tonemap image-chain
        // effects (diffraction spikes, anamorphic flare, fine texture)
        // don't have to be dimmed reactively by
        // `apply_display_headroom`. The telemetry slot is populated by
        // `apply_conflict_detection` (Guard 6/7) before rendering
        // starts, so peeking is safe here.
        let additive_count = crate::generation_log::peek_telemetry(|t| t.additive_effect_count)
            .unwrap_or(0);
        let p99_cap_target = if additive_count >= HEAVY_STACK_THRESHOLD {
            HIGHLIGHT_HEADROOM_P99_HEAVY
        } else {
            HIGHLIGHT_HEADROOM_P99
        };
        let cap_p99 = p99_cap_target / p99;
        if cap_p99 < auto {
            auto = cap_p99;
        }
        let cap_p995 = HIGHLIGHT_HEADROOM_P995 / p995;
        if cap_p995 < auto {
            auto = cap_p995;
        }
        let cap_p999 = HIGHLIGHT_HEADROOM_P999 / p999;
        if cap_p999 < auto {
            auto = cap_p999;
        }
    }

    let g = auto * pipeline_flags::scene_exposure_scale();
    for p in buffer.iter_mut() {
        p.0 *= g;
        p.1 *= g;
        p.2 *= g;
    }
}

/// Scene-linear, hue-preserving luminance ceiling applied between
/// `apply_auto_scene_metering` and the tonemap.
///
/// The metering step caps the **99th** percentile luminance; a compact
/// hot core sitting in the top 0.1 % can still be many multiples of that
/// value and punch past the `AgX` `max_ev = 2.5` (~5.66 linear) window.
/// Once all three channels clip there, the per-channel spline saturates
/// at ~1.0, the outset matrix leaves each channel near unity, and
/// `gamut_map_preserve_hue` no longer has chroma to recover — the pixel
/// collapses to flat display white.
///
/// This ceiling uniformly scales `r, g, b` by `ceiling / lum` whenever
/// the un-premultiplied luminance of a pixel exceeds `CEILING_LUMA`, so
/// the RGB ratio (hue + saturation in linear space) is preserved
/// exactly. A secondary per-channel guard catches nearly-pure colour
/// highlights (e.g. saturated red) whose luminance alone may stay below
/// the luma ceiling while a single channel runs far above it.
fn apply_scene_linear_ceiling(buffer: &mut PixelBuffer) {
    /// Luminance cap for the un-premultiplied scene-linear RGB. Sits well
    /// below `AgX`'s `2^2.5 ~ 5.66` saturation point so the tonemap still
    /// has room to produce a highlight shoulder instead of a hard clip.
    const CEILING_LUMA: f64 = 3.0;
    /// Per-channel cap. A bit above the luma ceiling so strongly coloured
    /// highlights (e.g. pure red where only R is hot) still get rolled
    /// in without flattening saturated tones that the luma cap alone
    /// would miss.
    const CEILING_CHANNEL: f64 = 4.5;
    buffer.par_iter_mut().for_each(|p| {
        let a = p.3;
        // --- Alpha-positive branch: un-premultiply, cap luma + channel ---
        //
        // Alpha-zero pixels (e.g. from `StarField` which adds RGB without
        // touching alpha, or from `Champleve`'s rim injection into
        // previously-empty background) previously took the
        // `if a <= 1e-9 { return; }` early exit and bypassed the ceiling.
        // They then entered the trajectory chain with unbounded RGB,
        // which `BloomPyramid` / `GodRays` / `PerceptualBlur` smeared into
        // alpha-positive neighbours, producing the 17-30% white-blob
        // fraction observed on seeds `0x17b5c96a2abd` and
        // `0x4767608a4991`. Capping per-channel in the alpha-zero path
        // closes that escape.
        if a > 1e-9 {
            let inv_a = 1.0 / a;
            let rs = p.0 * inv_a;
            let gs = p.1 * inv_a;
            let bs = p.2 * inv_a;
            let lum = constants::rec709_luminance(rs, gs, bs);
            let mut scale = 1.0;
            if lum > CEILING_LUMA {
                scale = CEILING_LUMA / lum;
            }
            // Per-channel safety: scale further if the brightest channel in
            // un-premul space is still above CEILING_CHANNEL after the luma
            // rescale. Uniform rescale preserves hue.
            let max_ch_after = rs.max(gs).max(bs) * scale;
            if max_ch_after > CEILING_CHANNEL {
                scale *= CEILING_CHANNEL / max_ch_after;
            }
            if scale < 1.0 {
                p.0 *= scale;
                p.1 *= scale;
                p.2 *= scale;
            }
        } else {
            // --- Alpha-zero branch: straight RGB, same caps, uniform scale ---
            //
            // With `a = 0` the RGB stored in the buffer is already
            // "raw" (it hasn't been multiplied by alpha), so we can
            // apply the luma + per-channel caps directly. The uniform
            // rescale still preserves hue.
            let lum = constants::rec709_luminance(p.0, p.1, p.2);
            let mut scale = 1.0;
            if lum > CEILING_LUMA {
                scale = CEILING_LUMA / lum;
            }
            let max_ch_after = p.0.max(p.1).max(p.2) * scale;
            if max_ch_after > CEILING_CHANNEL {
                scale *= CEILING_CHANNEL / max_ch_after;
            }
            if scale < 1.0 {
                p.0 *= scale;
                p.1 *= scale;
                p.2 *= scale;
            }
        }
    });
}

/// Energy-density wavelength shift for a single pixel's SPD.
///
/// Hot pixels (`total_energy > ENERGY_DENSITY_SHIFT_THRESHOLD`) have a
/// fraction of their spectral energy moved one bin redward; cool pixels
/// are returned unchanged. The transform is **pure** — the caller's SPD
/// is never modified — and returns a stack-allocated `[f64; NUM_BINS]`
/// (512 bytes), so it's safe to call per-pixel during conversion without
/// allocating a heap-sized parallel buffer.
///
/// The older mutating `apply_energy_density_shift(&mut accum_spd)` was a
/// bug magnet: because `accum_spd` is the persistent spectral accumulator
/// that keeps accruing new energy each frame, calling the shift in-place
/// inside the per-frame loop effectively applied the shift
/// `DEFAULT_TARGET_FRAMES` times (i.e. 1800). The shift is a one-way
/// conveyor belt (no bin ever loses to the left, bin `NUM_BINS-1` is a
/// terminal accumulator), so hot pixels had all of their energy migrate
/// to the red end, producing the red-dominance artefacts we saw in the
/// composite even though the per-bin spectral gallery still carried the
/// true hue. The fix is to never mutate the accumulator; the conversion
/// from SPD to RGBA applies the shift on the fly.
#[inline]
#[must_use]
pub(crate) fn energy_density_shifted_spd(spd: &[f64; NUM_BINS]) -> [f64; NUM_BINS] {
    use constants::{ENERGY_DENSITY_SHIFT_STRENGTH, ENERGY_DENSITY_SHIFT_THRESHOLD};

    let total_energy: f64 = spd.iter().sum();
    if total_energy < ENERGY_DENSITY_SHIFT_THRESHOLD {
        return *spd;
    }

    let excess_energy = total_energy - ENERGY_DENSITY_SHIFT_THRESHOLD;
    let shift_amount = (excess_energy * ENERGY_DENSITY_SHIFT_STRENGTH).min(1.0);

    let mut shifted = *spd;
    for i in (1..NUM_BINS).rev() {
        shifted[i] = spd[i] * (1.0 - shift_amount) + spd[i - 1] * shift_amount;
    }
    shifted[0] = spd[0] * (1.0 - shift_amount);
    shifted
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AccumulationBackend {
    ParallelScanlines,
    #[cfg(test)]
    SerialReference,
}

#[inline]
fn default_accumulation_backend() -> AccumulationBackend {
    AccumulationBackend::ParallelScanlines
}

fn checkpoint_steps(total_steps: usize, frame_interval: usize) -> Vec<usize> {
    if total_steps == 0 {
        return Vec::new();
    }

    let mut checkpoints = Vec::new();
    let mut checkpoint = frame_interval;
    while checkpoint < total_steps {
        if checkpoint > 0 {
            checkpoints.push(checkpoint);
        }
        checkpoint += frame_interval;
    }

    let final_step = total_steps - 1;
    if checkpoints.last().copied() != Some(final_step) {
        checkpoints.push(final_step);
    }

    checkpoints
}

struct AccumulationParams<'a> {
    scene: SpectralScene<'a>,
    ctx: &'a RenderContext,
    velocity_calc: &'a velocity_hdr::VelocityHdrCalculator<'a>,
    step_start: usize,
    step_end: usize,
    hdr_scale: f64,
}

fn accumulate_spectral_steps_into_rows(
    accum_spd: &mut [[f64; NUM_BINS]],
    params: &AccumulationParams<'_>,
    row_start: usize,
    row_end: usize,
) {
    if params.step_start >= params.step_end || row_start >= row_end {
        return;
    }

    let triangle_alphas = params.scene.triangle_alphas();
    let shutter = usize::from(pipeline_flags::shutter_samples().max(1));
    let n = params.scene.step_count();
    let inv_shutter = 1.0 / shutter as f64;

    for step in params.step_start..params.step_end {
        let hdr_mult_01 = params.velocity_calc.compute_segment_multiplier(step, 0, 1);
        let hdr_mult_12 = params.velocity_calc.compute_segment_multiplier(step, 1, 2);
        let hdr_mult_20 = params.velocity_calc.compute_segment_multiplier(step, 2, 0);

        for sub in 0..shutter {
            let frac = (sub as f64 + 0.5) * inv_shutter;
            let corners = if shutter > 1 && step + 1 < n {
                [
                    lerp_body_step(&params.scene.positions[0], step, frac),
                    lerp_body_step(&params.scene.positions[1], step, frac),
                    lerp_body_step(&params.scene.positions[2], step, frac),
                ]
            } else {
                [
                    params.scene.positions[0][step],
                    params.scene.positions[1][step],
                    params.scene.positions[2][step],
                ]
            };

            let vertices = prepare_triangle_vertices_at(
                corners,
                params.scene.colors,
                &triangle_alphas,
                step,
                params.ctx,
            );

            draw_triangle_batch_spectral_rows(
                accum_spd,
                &BatchDrawParams {
                    width: params.ctx.width,
                    height: params.ctx.height,
                    row_start,
                    row_end,
                    vertices,
                    hdr_multipliers: [hdr_mult_01, hdr_mult_12, hdr_mult_20],
                    hdr_scale: params.hdr_scale * inv_shutter,
                    kinematics: params.scene.kinematics,
                    step,
                },
            );
        }
    }

    if pipeline_flags::comet_wake_enabled() {
        const WAKE: usize = 64;
        let wake_start = params.step_end.saturating_sub(WAKE).max(params.step_start);
        for step in wake_start..params.step_end {
            let fade = (step + 1 - wake_start) as f64 / WAKE as f64;
            let wake_scale = params.hdr_scale * 0.12 * fade * fade;
            let vertices = prepare_triangle_vertices(
                params.scene.positions,
                params.scene.colors,
                &triangle_alphas,
                step,
                params.ctx,
            );
            let hdr_mult_01 = params.velocity_calc.compute_segment_multiplier(step, 0, 1);
            let hdr_mult_12 = params.velocity_calc.compute_segment_multiplier(step, 1, 2);
            let hdr_mult_20 = params.velocity_calc.compute_segment_multiplier(step, 2, 0);
            draw_triangle_batch_spectral_rows(
                accum_spd,
                &BatchDrawParams {
                    width: params.ctx.width,
                    height: params.ctx.height,
                    row_start,
                    row_end,
                    vertices,
                    hdr_multipliers: [hdr_mult_01, hdr_mult_12, hdr_mult_20],
                    hdr_scale: wake_scale,
                    kinematics: params.scene.kinematics,
                    step,
                },
            );
        }
    }

    if pipeline_flags::body_spheres_enabled()
        && let Some(masses) = params.scene.masses
    {
        let st = params.step_end.saturating_sub(1);
        draw_body_spheres_spectral_rows(
            accum_spd,
            &self::drawing::BodySphereDrawArgs {
                ctx: params.ctx,
                row_start,
                row_end,
                positions: params.scene.positions,
                colors: params.scene.colors,
                masses,
                body_alphas: &triangle_alphas,
                step: st,
                hdr_scale: params.hdr_scale * 0.35,
            },
        );
    }
}

fn accumulate_spectral_steps(
    accum_spd: &mut [[f64; NUM_BINS]],
    params: &AccumulationParams<'_>,
    backend: AccumulationBackend,
) {
    if params.step_start >= params.step_end || accum_spd.is_empty() {
        return;
    }

    match backend {
        AccumulationBackend::ParallelScanlines => {
            let ctx = params.ctx;
            let band_count = ctx.height_usize.min(rayon::current_num_threads().max(1));
            if band_count <= 1 {
                accumulate_spectral_steps_into_rows(accum_spd, params, 0, ctx.height_usize);
                return;
            }

            let rows_per_band = ctx.height_usize.div_ceil(band_count);
            let pixels_per_band = ctx.width_usize * rows_per_band;
            accum_spd.par_chunks_mut(pixels_per_band).enumerate().for_each(|(band_idx, band)| {
                let row_start = band_idx * rows_per_band;
                let row_end = row_start + band.len() / ctx.width_usize;
                accumulate_spectral_steps_into_rows(band, params, row_start, row_end);
            });
        }
        #[cfg(test)]
        AccumulationBackend::SerialReference => {
            accumulate_spectral_steps_into_rows(accum_spd, params, 0, params.ctx.height_usize);
        }
    }
}

fn pass_1_build_histogram_spectral_with_backend(
    scene: SpectralScene<'_>,
    frame_interval: usize,
    settings: SpectralRenderSettings<'_>,
    backend: AccumulationBackend,
) -> HistogramData {
    let SpectralRenderSettings {
        resolved_config,
        render_config,
        noise_seed,
        aspect_correction,
        ..
    } = settings;
    let width = resolved_config.width;
    let height = resolved_config.height;
    let ctx = RenderContext::new(width, height, scene.positions, aspect_correction);
    let mut accum_spd = vec![[0.0f64; NUM_BINS]; ctx.pixel_count()];
    let mut accum_rgba = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];
    let effect_config = build_effect_config_from_resolved(
        resolved_config,
        render_config,
        FinishOutputMode::Still,
        noise_seed,
    );
    let finish_pipeline = FinishEffectPipeline::new(effect_config);
    let mut histogram = HistogramData::with_capacity(ctx.pixel_count() * 10);

    let total_steps = scene.step_count();
    let checkpoints = checkpoint_steps(total_steps, frame_interval);
    let chunk_line = (total_steps / 10).max(1);
    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);
    let mut step_start = 0;

    for &checkpoint_step in &checkpoints {
        if step_start < total_steps && step_start % chunk_line == 0 {
            let pct = (step_start as f64 / total_steps as f64) * constants::PERCENT_FACTOR;
            debug!(progress = pct, pass = 1, mode = "spectral", "Histogram pass progress");
        }

        accumulate_spectral_steps(
            &mut accum_spd,
            &AccumulationParams {
                scene,
                ctx: &ctx,
                velocity_calc: &velocity_calc,
                step_start,
                step_end: checkpoint_step + 1,
                hdr_scale: render_config.hdr_scale,
            },
            backend,
        );

        convert_spd_buffer_to_rgba(&accum_spd, &mut accum_rgba, width as usize, height as usize);

        let frame_params =
            FrameParams { frame_number: checkpoint_step / frame_interval, density: None };
        let rgba_buffer = std::mem::take(&mut accum_rgba);
        let mut trajectory_proxy = finish_pipeline
            .process_trajectory(rgba_buffer, width as usize, height as usize, &frame_params)
            .expect("effect chain invariant: histogram-pass trajectory processing must not fail");
        // Pass-1 histogram must see the same pixel values that pass-2
        // will feed into `tonemap_core`: both auto-metering and the
        // scene-linear ceiling happen before tonemap in pass 2, so they
        // must also happen here. Without this, `ChannelLevels` are
        // calibrated to the unmetered proxy, and pass-2 metering can
        // push the actual tonemap inputs off that calibrated range —
        // contributing to blowouts on pathological seeds.
        apply_auto_scene_metering(&mut trajectory_proxy);
        apply_scene_linear_ceiling(&mut trajectory_proxy);
        accum_rgba.clear();
        accum_rgba.resize(ctx.pixel_count(), (0.0, 0.0, 0.0, 0.0));

        histogram.reserve(ctx.pixel_count());
        for &(r, g, b, a) in &trajectory_proxy {
            histogram.push(r * a, g * a, b * a);
        }

        step_start = checkpoint_step + 1;
    }

    info!("   pass 1 (spectral histogram): 100% done");
    histogram
}

// ====================== PASS 1 (SPECTRAL) ===========================
/// Pass 1: gather global histogram for final color leveling (spectral)
pub fn pass_1_build_histogram_spectral(
    scene: SpectralScene<'_>,
    frame_interval: usize,
    settings: SpectralRenderSettings<'_>,
) -> HistogramData {
    pass_1_build_histogram_spectral_with_backend(
        scene,
        frame_interval,
        settings,
        default_accumulation_backend(),
    )
}

#[cfg(test)]
pub(crate) fn pass_1_build_histogram_spectral_serial_reference(
    scene: SpectralScene<'_>,
    frame_interval: usize,
    settings: SpectralRenderSettings<'_>,
) -> HistogramData {
    pass_1_build_histogram_spectral_with_backend(
        scene,
        frame_interval,
        settings,
        AccumulationBackend::SerialReference,
    )
}

/// Bundled parameters for `pass_2_write_frames_spectral` and its backend variants.
///
/// Groups the non-closure state that every pass-2 call site must supply,
/// keeping the function signatures under clippy's argument-count threshold.
pub struct Pass2Params<'a> {
    /// Trajectory positions, colours, and per-body alphas to draw.
    pub scene: SpectralScene<'a>,
    /// Simulation steps between emitted video frames.
    pub frame_interval: usize,
    /// Per-channel black/white levels from pass 1 histogram analysis.
    pub levels: &'a ChannelLevels,
    /// Resolved effects, HDR scale, bloom, noise seed, and aspect handling.
    pub settings: SpectralRenderSettings<'a>,
    /// Receives the final frame as 16-bit RGB when the pass completes.
    pub last_frame_out: &'a mut Option<ImageBuffer<Rgb<u16>, Vec<u16>>>,
    /// When true, blends consecutive display frames to reduce temporal noise.
    pub enable_temporal_smoothing: bool,
    /// Scratch buffer for per-pixel spectral power distributions (reused across checkpoints).
    pub accum_spd: &'a mut Vec<[f64; NUM_BINS]>,
}

#[cfg(test)]
pub(crate) fn pass_2_write_frames_spectral_serial_reference(
    params: Pass2Params<'_>,
    frame_sink: impl FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    pass_2_write_frames_spectral_with_backend(
        params,
        frame_sink,
        AccumulationBackend::SerialReference,
    )
}

// ====================== PASS 2 (SPECTRAL) ===========================
/// Pass 2: final frames => color mapping => write frames (spectral, 16-bit output)
///
/// The caller-provided `accum_spd` buffer is populated incrementally and contains
/// the fully accumulated spectral data when this function returns.
fn pass_2_write_frames_spectral_with_backend(
    params: Pass2Params<'_>,
    mut frame_sink: impl FnMut(&[u8]) -> Result<()>,
    backend: AccumulationBackend,
) -> Result<()> {
    let Pass2Params {
        scene,
        frame_interval,
        levels,
        settings,
        last_frame_out,
        enable_temporal_smoothing,
        accum_spd,
    } = params;
    let SpectralRenderSettings { resolved_config, render_config, noise_seed, aspect_correction } =
        settings;
    let width = resolved_config.width;
    let height = resolved_config.height;
    let ctx = RenderContext::new(width, height, scene.positions, aspect_correction);
    accum_spd.resize(ctx.pixel_count(), [0.0f64; NUM_BINS]);
    for s in accum_spd.iter_mut() {
        *s = [0.0; NUM_BINS];
    }
    let mut accum_rgba = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];

    let effect_config = build_effect_config_from_resolved(
        resolved_config,
        render_config,
        FinishOutputMode::Video,
        noise_seed,
    );
    let finish_pipeline = FinishEffectPipeline::new(effect_config);

    let total_steps = scene.step_count();
    let checkpoints = if pipeline_flags::multi_act_director_enabled() {
        director::resolve_video_checkpoints(total_steps, true)
    } else {
        checkpoint_steps(total_steps, frame_interval)
    };
    let chunk_line = (total_steps / 10).max(1);
    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);

    use crate::post_effects::{TemporalSmoothing, TemporalSmoothingConfig};
    let temporal_smoother = if enable_temporal_smoothing {
        Some(TemporalSmoothing::new(TemporalSmoothingConfig {
            blend_factor: 0.10,
            alpha_threshold: 0.01,
        }))
    } else {
        None
    };
    let mut step_start = 0;

    for (frame_idx, &checkpoint_step) in checkpoints.iter().enumerate() {
        if step_start < total_steps && step_start % chunk_line == 0 {
            let pct = (step_start as f64 / total_steps as f64) * constants::PERCENT_FACTOR;
            debug!(progress = pct, pass = 2, mode = "spectral", "Render pass progress");
        }

        accumulate_spectral_steps(
            accum_spd,
            &AccumulationParams {
                scene,
                ctx: &ctx,
                velocity_calc: &velocity_calc,
                step_start,
                step_end: checkpoint_step + 1,
                hdr_scale: render_config.hdr_scale,
            },
            backend,
        );

        convert_spd_buffer_to_rgba(accum_spd, &mut accum_rgba, width as usize, height as usize);

        let frame_params = FrameParams { frame_number: frame_idx, density: None };
        let rgba_buffer = std::mem::take(&mut accum_rgba);
        let mut trajectory_pixels = finish_pipeline
            .process_trajectory(rgba_buffer, width as usize, height as usize, &frame_params)
            .map_err(|e| RenderError::EffectChain {
                effect_name: "trajectory_chain".into(),
                reason: e.to_string(),
            })?;

        record_stage("post_trajectory", &trajectory_pixels, LuminanceSpace::SceneLinear);
        apply_auto_scene_metering(&mut trajectory_pixels);
        record_stage("post_metering", &trajectory_pixels, LuminanceSpace::SceneLinear);
        apply_scene_linear_ceiling(&mut trajectory_pixels);
        record_stage("post_ceiling", &trajectory_pixels, LuminanceSpace::SceneLinear);

        let smoothed_linear = match &temporal_smoother {
            Some(smoother) => smoother.process_frame(trajectory_pixels),
            None => trajectory_pixels,
        };

        // Reclaim the trajectory buffer's allocation back into accum_rgba.
        // It will be fully overwritten by convert_spd_buffer_to_rgba next iteration,
        // so we just need the capacity -- no need to clear or resize.
        let mut reclaimed = smoothed_linear;
        reclaimed.resize(ctx.pixel_count(), (0.0, 0.0, 0.0, 0.0));
        accum_rgba = reclaimed;

        let display_buffer = tonemap_to_display_buffer(&accum_rgba, levels);
        record_stage("post_tonemap", &display_buffer, LuminanceSpace::Display);

        let mut final_display = finish_pipeline
            .process_image(display_buffer, width as usize, height as usize, &frame_params)
            .map_err(|e| RenderError::EffectChain {
                effect_name: "image_chain".into(),
                reason: e.to_string(),
            })?;
        record_stage("post_image_chain", &final_display, LuminanceSpace::Display);
        // Distribution-aware display-space headroom cap: dims the whole
        // buffer uniformly if p99 luminance or bright-pixel fraction is
        // out of spec. Runs before the per-pixel safety shoulder so the
        // shoulder still gets to smooth residual outliers.
        apply_display_headroom(&mut final_display);
        record_stage("post_display_headroom", &final_display, LuminanceSpace::Display);
        // Final hue-preserving safety shoulder: the image chain
        // (diffraction_spikes, anamorphic_flare, fine_texture, ...) runs
        // on the already-tonemapped display buffer and can push channels
        // past 1.0. Without this roll-off, `quantize_display_buffer_to_16bit`
        // hard-clamps per channel, destroying hue on hot pixels and
        // producing the pure-white cores + primary-colour splotches.
        apply_display_safety_shoulder(&mut final_display);
        record_stage("post_shoulder", &final_display, LuminanceSpace::Display);
        let buf_16bit = quantize_display_buffer_to_16bit(&final_display);
        let buf_bytes: &[u8] = bytemuck::cast_slice(&buf_16bit);

        frame_sink(buf_bytes)?;
        if checkpoint_step + 1 == total_steps {
            *last_frame_out = ImageBuffer::from_raw(width, height, buf_16bit);
        }

        step_start = checkpoint_step + 1;
    }

    info!("   pass 2 (spectral render): 100% done");
    Ok(())
}

/// Pass 2: render frames with spectral accumulation and feed 16-bit bytes to `frame_sink`.
pub fn pass_2_write_frames_spectral(
    params: Pass2Params<'_>,
    frame_sink: impl FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    pass_2_write_frames_spectral_with_backend(params, frame_sink, default_accumulation_backend())
}

/// Render the fully accumulated final frame without writing intermediate video frames.
///
/// This is the correct preview path for still-image QA because it matches the final
/// accumulated composition instead of an early timeline slice.
pub fn render_final_frame_spectral(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
) -> Result<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    render_final_frame_spectral_with_backend(
        scene,
        levels,
        settings,
        default_accumulation_backend(),
    )
}

#[cfg(test)]
pub(crate) fn render_final_frame_spectral_serial_reference(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
) -> Result<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    render_final_frame_spectral_with_backend(
        scene,
        levels,
        settings,
        AccumulationBackend::SerialReference,
    )
}

fn render_final_frame_spectral_with_backend(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    backend: AccumulationBackend,
) -> Result<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    let SpectralRenderSettings { resolved_config, render_config, noise_seed, aspect_correction } =
        settings;
    info!("   Rendering final accumulated frame (preview mode)...");

    let width = resolved_config.width;
    let height = resolved_config.height;
    let ctx = RenderContext::new(width, height, scene.positions, aspect_correction);
    let mut accum_spd = vec![[0.0f64; NUM_BINS]; ctx.pixel_count()];
    let mut accum_rgba = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];

    let effect_config = build_effect_config_from_resolved(
        resolved_config,
        render_config,
        FinishOutputMode::Still,
        noise_seed,
    );
    let finish_pipeline = FinishEffectPipeline::new(effect_config);

    let total_steps = scene.step_count();
    let dt = constants::DEFAULT_DT;
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);

    accumulate_spectral_steps(
        &mut accum_spd,
        &AccumulationParams {
            scene,
            ctx: &ctx,
            velocity_calc: &velocity_calc,
            step_start: 0,
            step_end: total_steps,
            hdr_scale: render_config.hdr_scale,
        },
        backend,
    );

    convert_spd_buffer_to_rgba(&accum_spd, &mut accum_rgba, width as usize, height as usize);

    let frame_interval = (total_steps / constants::DEFAULT_TARGET_FRAMES as usize).max(1);
    let preview_frame_number = total_steps.saturating_sub(1) / frame_interval;
    let frame_params = FrameParams { frame_number: preview_frame_number, density: None };
    let mut trajectory_pixels = finish_pipeline
        .process_trajectory(accum_rgba, width as usize, height as usize, &frame_params)
        .map_err(|e| RenderError::EffectChain {
            effect_name: "trajectory_chain".into(),
            reason: e.to_string(),
        })?;

    // Match the pass-2 video path exactly: global exposure metering and
    // a hue-preserving scene-linear ceiling before tonemap. Without
    // these, extreme seeds produce blown-out white cores because the
    // additive trajectory chain (bloom, god rays, chromatic bloom,
    // aether, ...) can pile far above the AgX headroom before the
    // tonemap gets a chance to compress it.
    record_stage("post_trajectory", &trajectory_pixels, LuminanceSpace::SceneLinear);
    apply_auto_scene_metering(&mut trajectory_pixels);
    record_stage("post_metering", &trajectory_pixels, LuminanceSpace::SceneLinear);
    apply_scene_linear_ceiling(&mut trajectory_pixels);
    record_stage("post_ceiling", &trajectory_pixels, LuminanceSpace::SceneLinear);

    let display_buffer = tonemap_to_display_buffer(&trajectory_pixels, levels);
    record_stage("post_tonemap", &display_buffer, LuminanceSpace::Display);
    let mut final_display = finish_pipeline
        .process_image(display_buffer, width as usize, height as usize, &frame_params)
        .map_err(|e| RenderError::EffectChain {
            effect_name: "image_chain".into(),
            reason: e.to_string(),
        })?;
    record_stage("post_image_chain", &final_display, LuminanceSpace::Display);
    apply_display_headroom(&mut final_display);
    record_stage("post_display_headroom", &final_display, LuminanceSpace::Display);
    apply_display_safety_shoulder(&mut final_display);
    record_stage("post_shoulder", &final_display, LuminanceSpace::Display);
    let buf_16bit = quantize_display_buffer_to_16bit(&final_display);

    ImageBuffer::from_raw(width, height, buf_16bit).ok_or_else(|| RenderError::ImageEncoding {
        reason: "Failed to create 16-bit image buffer".into(),
    })
}

// ====================== SINGLE FRAME RENDERING ===========================
/// Render the first timeline slice only for tests.
#[cfg(test)]
pub(crate) fn render_single_frame_spectral(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
) -> Result<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    render_single_frame_spectral_with_backend(
        scene,
        levels,
        settings,
        default_accumulation_backend(),
    )
}

#[cfg(test)]
pub(crate) fn render_single_frame_spectral_serial_reference(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
) -> Result<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    render_single_frame_spectral_with_backend(
        scene,
        levels,
        settings,
        AccumulationBackend::SerialReference,
    )
}

#[cfg(test)]
fn render_single_frame_spectral_with_backend(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    backend: AccumulationBackend,
) -> Result<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    let SpectralRenderSettings { resolved_config, render_config, noise_seed, aspect_correction } =
        settings;
    info!("   Rendering first timeline slice only (legacy test mode)...");

    let width = resolved_config.width;
    let height = resolved_config.height;
    // Create render context
    let ctx = RenderContext::new(width, height, scene.positions, aspect_correction);
    let mut accum_spd = vec![[0.0f64; NUM_BINS]; ctx.pixel_count()];
    let mut accum_rgba = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];

    // Build effect configuration from resolved config
    let effect_config = build_effect_config_from_resolved(
        resolved_config,
        render_config,
        FinishOutputMode::Still,
        noise_seed,
    );
    let finish_pipeline = FinishEffectPipeline::new(effect_config);

    let total_steps = scene.step_count();
    let dt = constants::DEFAULT_DT;

    // Create velocity HDR calculator for efficient multiplier computation
    let velocity_calc = velocity_hdr::VelocityHdrCalculator::new(scene.positions, dt);

    // Render all trajectory steps up to and including the first output frame interval
    let frame_interval = (total_steps / constants::DEFAULT_TARGET_FRAMES as usize).max(1);
    let first_frame_step = frame_interval;

    accumulate_spectral_steps(
        &mut accum_spd,
        &AccumulationParams {
            scene,
            ctx: &ctx,
            velocity_calc: &velocity_calc,
            step_start: 0,
            step_end: first_frame_step + 1,
            hdr_scale: render_config.hdr_scale,
        },
        backend,
    );

    // Process the accumulated frame
    convert_spd_buffer_to_rgba(&accum_spd, &mut accum_rgba, width as usize, height as usize);

    let frame_params = FrameParams { frame_number: 0, density: None };
    let trajectory_pixels = finish_pipeline
        .process_trajectory(accum_rgba, width as usize, height as usize, &frame_params)
        .map_err(|e| RenderError::EffectChain {
            effect_name: "trajectory_chain".into(),
            reason: e.to_string(),
        })?;

    let display_buffer = tonemap_to_display_buffer(&trajectory_pixels, levels);
    let mut final_display = finish_pipeline
        .process_image(display_buffer, width as usize, height as usize, &frame_params)
        .map_err(|e| RenderError::EffectChain {
            effect_name: "image_chain".into(),
            reason: e.to_string(),
        })?;
    apply_display_headroom(&mut final_display);
    apply_display_safety_shoulder(&mut final_display);

    // Quantize display buffer to 16-bit
    let buf_16bit = quantize_display_buffer_to_16bit(&final_display);

    // Create ImageBuffer and return
    let image = ImageBuffer::from_raw(width, height, buf_16bit).ok_or_else(|| {
        RenderError::ImageEncoding { reason: "Failed to create 16-bit image buffer".into() }
    })?;

    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::randomizable_config::ResolvedEffectConfig;
    use nalgebra::Vector3;
    use rayon::ThreadPoolBuilder;

    /// Reconstruct the un-premultiplied RGB from a premul pixel and
    /// compare each channel as a ratio (NaN-safe, 1e-9 alpha guard).
    fn unpremul(p: (f64, f64, f64, f64)) -> (f64, f64, f64) {
        let a = p.3.max(1e-12);
        (p.0 / a, p.1 / a, p.2 / a)
    }

    #[test]
    fn scene_linear_ceiling_preserves_hue_when_clamping() {
        // A pixel with a hot core: luminance well above the ceiling but
        // with a distinct saturated colour (high blue, low green).
        let rs = 6.0;
        let gs = 1.0;
        let bs = 10.0;
        let alpha = 0.8;
        let mut buffer: PixelBuffer =
            vec![(rs * alpha, gs * alpha, bs * alpha, alpha)];
        apply_scene_linear_ceiling(&mut buffer);
        let (rn, gn, bn) = unpremul(buffer[0]);
        // Luminance must now be at or below the 3.0 ceiling (with a
        // small epsilon for floating-point slack and the secondary
        // channel cap which may pull luminance even lower).
        let lum = constants::rec709_luminance(rn, gn, bn);
        assert!(lum <= 3.0 + 1e-9, "luminance {lum} exceeds ceiling");
        // Ratios r:g:b must match the input: uniform scale preserves
        // hue and saturation exactly.
        let orig_ratio_rg = rs / gs;
        let orig_ratio_bg = bs / gs;
        let new_ratio_rg = rn / gn;
        let new_ratio_bg = bn / gn;
        assert!(
            (new_ratio_rg - orig_ratio_rg).abs() / orig_ratio_rg < 1e-6,
            "r/g ratio drifted: {new_ratio_rg} vs {orig_ratio_rg}"
        );
        assert!(
            (new_ratio_bg - orig_ratio_bg).abs() / orig_ratio_bg < 1e-6,
            "b/g ratio drifted: {new_ratio_bg} vs {orig_ratio_bg}"
        );
    }

    #[test]
    fn scene_linear_ceiling_leaves_well_exposed_pixels_untouched() {
        // A well-exposed pixel (luminance < 3.0, all channels < 4.5)
        // should emerge bit-identical.
        let alpha = 0.5;
        let original = (0.8 * alpha, 1.2 * alpha, 0.5 * alpha, alpha);
        let mut buffer: PixelBuffer = vec![original];
        apply_scene_linear_ceiling(&mut buffer);
        assert_eq!(buffer[0], original);
    }

    #[test]
    fn scene_linear_ceiling_ignores_zero_alpha() {
        // Transparent pixels with zero RGB must remain unchanged
        // (no division-by-zero, no spurious energy injection).
        let mut buffer: PixelBuffer = vec![(0.0, 0.0, 0.0, 0.0)];
        apply_scene_linear_ceiling(&mut buffer);
        assert_eq!(buffer[0], (0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn scene_linear_ceiling_caps_bright_zero_alpha_pixels() {
        // Regression: `StarField` (and similar additive passes) inject
        // bright RGB into alpha=0 pixels. Before the fix, this branch
        // early-exited and the energy leaked past the ceiling into
        // `BloomPyramid`, producing the observed white blobs. The fix
        // caps these pixels' luma at `CEILING_LUMA = 3.0` regardless of
        // alpha.
        let mut buffer: PixelBuffer = vec![(10.0, 10.0, 10.0, 0.0)];
        apply_scene_linear_ceiling(&mut buffer);
        let lum = constants::rec709_luminance(buffer[0].0, buffer[0].1, buffer[0].2);
        assert!(
            lum <= 3.0 + 1e-6,
            "alpha=0 bright pixel must still be capped (got lum = {lum})"
        );
        // Hue must be preserved (uniform white in, uniform white out).
        assert!((buffer[0].0 - buffer[0].1).abs() < 1e-9);
        assert!((buffer[0].0 - buffer[0].2).abs() < 1e-9);
    }

    #[test]
    fn display_headroom_caps_bright_fraction_on_synthetic_blob() {
        // 30% of the pixels have luminance = 1.2 (simulating the
        // worst observed white blob: 30.1% bright on seed
        // 0x4767608a4991). Post-headroom the bright fraction must fall
        // below the 5% cap, and p99 luminance must land near/below
        // the 0.85 cap.
        let mut buffer: PixelBuffer = Vec::with_capacity(10_000);
        for i in 0..10_000 {
            if i < 3_000 {
                buffer.push((1.2, 1.2, 1.2, 1.0)); // Bright blob.
            } else {
                buffer.push((0.2, 0.2, 0.2, 1.0)); // Dim background.
            }
        }
        apply_display_headroom(&mut buffer);

        let lums: Vec<f64> = buffer
            .iter()
            .map(|&(r, g, b, _)| constants::rec709_luminance(r, g, b))
            .collect();
        let bright = lums.iter().filter(|&&l| l > 0.85).count();
        let bright_frac = bright as f64 / lums.len() as f64;
        let mut sorted = lums.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p99 = sorted[(sorted.len() as f64 * 0.99) as usize];

        assert!(
            bright_frac <= 0.05 + 1e-6,
            "bright fraction must fall under cap (got {bright_frac})"
        );
        assert!(p99 <= 0.85 + 1e-6, "p99 must fall under cap (got {p99})");
    }

    #[test]
    fn display_headroom_leaves_well_exposed_buffer_untouched() {
        // A frame with a healthy distribution (all pixels < 0.7 lum)
        // must pass through apply_display_headroom bit-for-bit: the
        // guard is additive, never corrective on already-good frames.
        let mut buffer: PixelBuffer =
            (0..1000).map(|i| (0.1 + f64::from(i % 10) * 0.05, 0.1, 0.1, 1.0)).collect();
        let before = buffer.clone();
        apply_display_headroom(&mut buffer);
        assert_eq!(buffer, before, "well-exposed buffer must be unchanged");
    }

    #[test]
    fn display_headroom_preserves_hue_on_uniform_dim() {
        // When the guard engages, the dim is *uniform* across all
        // channels — per-pixel R:G:B ratios must be preserved exactly.
        let mut buffer: PixelBuffer = (0..5_000)
            .map(|i| {
                if i < 2_000 {
                    (2.0, 0.5, 1.0, 1.0) // Bright with a specific hue.
                } else {
                    (0.1, 0.02, 0.05, 1.0)
                }
            })
            .collect();
        apply_display_headroom(&mut buffer);
        // First bright pixel — check the r:g and r:b ratios survive.
        let p = buffer[0];
        let ratio_rg = p.0 / p.1;
        let ratio_rb = p.0 / p.2;
        assert!((ratio_rg - 4.0).abs() < 1e-6, "r/g hue drift: {ratio_rg}");
        assert!((ratio_rb - 2.0).abs() < 1e-6, "r/b hue drift: {ratio_rb}");
    }

    /// Midtones (luma < `HEADROOM_LUMA_KNEE`) must pass through
    /// bit-for-bit — this is the key contrast-preservation property
    /// the legacy uniform-dim implementation violated. A 30%-bright
    /// buffer with a midtone background must leave the midtone
    /// untouched.
    #[test]
    fn display_headroom_preserves_midtone_contrast() {
        // 30% bright-white pixels + 70% pixels spanning [0.1, 0.65]
        // luma (all below the HEADROOM_LUMA_KNEE=0.70 cut). The bright
        // pixels get compressed but the midtones must emerge identical.
        let mut buffer: PixelBuffer = Vec::with_capacity(10_000);
        for i in 0..10_000 {
            if i < 3_000 {
                buffer.push((1.2, 1.2, 1.2, 1.0));
            } else {
                let v = 0.1 + f64::from(i % 11) * 0.05;
                buffer.push((v, v, v, 1.0));
            }
        }
        let before = buffer.clone();
        apply_display_headroom(&mut buffer);
        // Every midtone pixel (originally below knee) must be exactly
        // equal to its pre-guard value.
        for i in 3_000..10_000 {
            assert_eq!(
                buffer[i], before[i],
                "midtone pixel {i} drifted: {:?} -> {:?}",
                before[i], buffer[i]
            );
        }
        // The bright pixels must have been compressed to strictly
        // below the knee (so the stage 1 shoulder actually engaged).
        let bright = buffer[0];
        let bright_luma = constants::rec709_luminance(bright.0, bright.1, bright.2);
        assert!(
            bright_luma < 0.85 + 1e-6,
            "bright pixel not compressed: {bright:?} luma={bright_luma}"
        );
    }

    /// Stage 2 (per-channel gamut cap) must engage for strongly
    /// tinted pixels whose rec709 luminance is below the knee but
    /// whose max channel is above `HEADROOM_CHANNEL_CEILING` (e.g.
    /// saturated red: R contributes only 21.3% of luma).
    #[test]
    fn display_headroom_stage2_caps_saturated_tints() {
        // Saturated red: R=1.5, G=0.1, B=0.05.
        // rec709 luma = 0.2126*1.5 + 0.7152*0.1 + 0.0722*0.05 = 0.394
        // — comfortably below HEADROOM_LUMA_KNEE=0.70, so stage 1
        // does nothing. But max channel = 1.5 > HEADROOM_CHANNEL_
        // CEILING=0.92, so stage 2 must rescale hue-preservingly.
        let mut buffer: PixelBuffer = vec![(1.5, 0.1, 0.05, 1.0); 500];
        apply_display_headroom(&mut buffer);
        let p = buffer[0];
        // Max channel must land at the ceiling (within f64 eps).
        assert!(
            (p.0 - HEADROOM_CHANNEL_CEILING).abs() < 1e-9,
            "stage 2 did not cap max channel: {p:?}"
        );
        // Hue ratios preserved (R:G = 15, R:B = 30).
        assert!((p.0 / p.1 - 15.0).abs() < 1e-6, "R/G drifted: {}", p.0 / p.1);
        assert!((p.0 / p.2 - 30.0).abs() < 1e-6, "R/B drifted: {}", p.0 / p.2);
    }

    /// Stage 1 shoulder must be monotonic: a brighter input must
    /// produce an equal-or-brighter output, so highlights remain
    /// ordered (no inversions). This is the classical tonemapper
    /// contract — the exponential asymptote preserves it.
    #[test]
    fn display_headroom_is_monotonic_in_luminance() {
        let lums = [0.71_f64, 0.80, 0.90, 1.00, 1.20, 1.50, 2.00, 5.00];
        let mut outs = Vec::new();
        for &l in &lums {
            let rolled = compress_headroom_highlights(
                [l, l, l],
                HEADROOM_LUMA_KNEE,
                HEADROOM_LUMA_ASYMPTOTE,
                HEADROOM_CHANNEL_CEILING,
            );
            outs.push(rolled[0]);
        }
        for w in outs.windows(2) {
            assert!(w[0] <= w[1] + 1e-12, "monotonicity violated: {outs:?}");
        }
        // And the top of the range must not exceed the asymptote
        // (modulo the channel ceiling which is higher than it).
        let last = outs.last().copied().unwrap();
        assert!(
            last <= HEADROOM_LUMA_ASYMPTOTE + 1e-9,
            "asymptote violated: {last} > {HEADROOM_LUMA_ASYMPTOTE}"
        );
    }

    /// Hue is preserved exactly under `compress_headroom_highlights`
    /// for any positive RGB — a property-style sanity check.
    #[test]
    fn compress_headroom_highlights_preserves_hue_exactly() {
        let samples = [
            [0.71_f64, 0.36, 0.18], // Bright tinted.
            [1.20, 0.60, 0.30],
            [0.50, 1.00, 0.25],
            [0.05, 0.05, 0.05], // Untouched midtone.
            [2.00, 2.00, 2.00], // Uniform bright.
            [1.50, 0.10, 0.05], // Saturated — exercises stage 2.
        ];
        for rgb in samples {
            let out = compress_headroom_highlights(
                rgb,
                HEADROOM_LUMA_KNEE,
                HEADROOM_LUMA_ASYMPTOTE,
                HEADROOM_CHANNEL_CEILING,
            );
            // Hue ratio: R/G and R/B must match (skip zero channels).
            if rgb[1] > 1e-9 && out[1] > 1e-9 {
                let orig = rgb[0] / rgb[1];
                let new = out[0] / out[1];
                assert!(
                    (orig - new).abs() / orig.max(1e-12) < 1e-9,
                    "R/G drift for {rgb:?}: {orig} -> {new}"
                );
            }
            if rgb[2] > 1e-9 && out[2] > 1e-9 {
                let orig = rgb[0] / rgb[2];
                let new = out[0] / out[2];
                assert!(
                    (orig - new).abs() / orig.max(1e-12) < 1e-9,
                    "R/B drift for {rgb:?}: {orig} -> {new}"
                );
            }
        }
    }

    /// If stage 1 alone is enough to bring the distribution under
    /// spec, stage 3 must NOT engage — the residual uniform gain is
    /// a last-resort fallback.
    #[test]
    fn display_headroom_skips_residual_gain_when_shoulder_suffices() {
        // 40% of pixels at luma = 0.92 (above the P99 cap). Stage 1
        // compresses luma to ~0.78, well below cap. Stage 3 must NOT
        // apply a further uniform dim: midtones must stay at exactly
        // their input value.
        let mut buffer: PixelBuffer = Vec::with_capacity(10_000);
        for i in 0..10_000 {
            if i < 4_000 {
                buffer.push((0.92, 0.92, 0.92, 1.0));
            } else {
                buffer.push((0.3, 0.3, 0.3, 1.0));
            }
        }
        apply_display_headroom(&mut buffer);
        // Midtone pixel: must be exactly 0.3 (no residual gain).
        let mid = buffer[5_000];
        assert!(
            (mid.0 - 0.3).abs() < 1e-12,
            "midtone got dimmed by residual gain: {mid:?}"
        );
    }

    /// Residual gain never drops below `HEADROOM_RESIDUAL_GAIN_FLOOR`,
    /// even on adversarial inputs where stage 1 can't pull things
    /// under the caps. This prevents the legacy "flat grey blob"
    /// regression where a 30%-bright blob dimmed the whole image to
    /// near-black.
    #[test]
    fn display_headroom_residual_gain_floors_midtone_dim() {
        // Pathological: 99% of pixels at uniform bright white, 1% at
        // dim. Stage 1 compresses luma to ~0.80. Stage 3 can only
        // attenuate by HEADROOM_RESIDUAL_GAIN_FLOOR.
        let mut buffer: PixelBuffer = Vec::with_capacity(10_000);
        for i in 0..10_000 {
            if i < 9_900 {
                buffer.push((2.0, 2.0, 2.0, 1.0));
            } else {
                buffer.push((0.3, 0.3, 0.3, 1.0));
            }
        }
        let dim_before = buffer[9_999];
        apply_display_headroom(&mut buffer);
        let dim_after = buffer[9_999];
        // The dim pixel started below the knee so stage 1 left it
        // alone. Stage 3 can dim it at most by
        // HEADROOM_RESIDUAL_GAIN_FLOOR.
        let ratio = dim_after.0 / dim_before.0;
        assert!(
            ratio >= HEADROOM_RESIDUAL_GAIN_FLOOR - 1e-9,
            "midtone dimmed below residual gain floor: {ratio} < {HEADROOM_RESIDUAL_GAIN_FLOOR}"
        );
    }

    /// End-to-end safeguard-stack regression for the observed white-blob
    /// failure mode on remote seeds `0x17b5c96a2abd` (17.6% bright) and
    /// `0x4767608a4991` (30.1% bright). Synthesises an input buffer that
    /// reproduces the pathological shape:
    ///
    /// 1. A **bright alpha-zero starfield** across 20% of pixels — this
    ///    is what `StarField` / `Champleve` inject before the alpha-gated
    ///    metering could previously see it. (Fixed in
    ///    `scene_linear_ceiling_caps_bright_zero_alpha_pixels`.)
    /// 2. A **hot alpha-positive core** across another 10% of pixels,
    ///    emulating an already-bloomed trajectory chain.
    /// 3. A **dim alpha-positive background** for the remaining 70%.
    ///
    /// After the scene-linear ceiling, auto metering, tonemap, and
    /// `apply_display_headroom`, the final display buffer must land
    /// below the museum-quality bright-fraction target (2%).
    #[test]
    fn white_blob_regression_synthetic_end_to_end() {
        let n = 10_000;
        let mut buffer: PixelBuffer = Vec::with_capacity(n);
        for i in 0..n {
            if i < 2_000 {
                // Alpha-zero bright "stars" — the historical escape
                // route. Before the ceiling fix these leaked straight
                // into the blur pyramid.
                buffer.push((5.0, 5.0, 5.0, 0.0));
            } else if i < 3_000 {
                // Alpha-positive hot core — mirrors a bloomed cluster.
                buffer.push((4.0 * 0.8, 4.0 * 0.8, 4.0 * 0.8, 0.8));
            } else {
                // Dim background.
                buffer.push((0.1 * 0.3, 0.1 * 0.3, 0.1 * 0.3, 0.3));
            }
        }

        // Apply the same safeguard stack as the production pipeline.
        apply_auto_scene_metering(&mut buffer);
        apply_scene_linear_ceiling(&mut buffer);
        let levels = default_levels();
        let mut display = tonemap_to_display_buffer(&buffer, &levels);
        apply_display_headroom(&mut display);
        apply_display_safety_shoulder(&mut display);

        // Measure the final display-space bright fraction.
        let bright = display
            .iter()
            .filter(|&&(r, g, b, _)| constants::rec709_luminance(r, g, b) > 0.85)
            .count();
        let bright_frac = bright as f64 / display.len() as f64;
        assert!(
            bright_frac <= 0.02,
            "end-to-end bright fraction must be museum-quality ({bright_frac} > 0.02)"
        );
    }

    #[test]
    fn stage_luminance_matches_manual_computation() {
        // Sanity: the stage-telemetry helper reports p99, p99.9, max,
        // and bright_fraction consistent with a manual reference.
        let mut buffer: PixelBuffer = Vec::with_capacity(1000);
        for i in 0..1000 {
            // 0.0 for first 990, 0.95 for next 9, 1.2 for last one.
            let lum = if i < 990 {
                0.0
            } else if i < 999 {
                0.95
            } else {
                1.2
            };
            buffer.push((lum, lum, lum, 1.0));
        }
        let stats = compute_stage_luminance(&buffer, LuminanceSpace::Display);
        // bright_fraction: 10/1000 = 0.01.
        assert!((stats.bright_fraction - 0.01).abs() < 1e-9);
        // max = 1.2.
        assert!((stats.max - 1.2).abs() < 1e-9);
        // p99 sits at the boundary between 0.0 and 0.95, so near 0.95.
        assert!(stats.p99 >= 0.94 && stats.p99 <= 0.96, "p99 = {}", stats.p99);
        // p99.9 lands in the tail — should be either 0.95 or 1.2.
        assert!(stats.p99_9 >= 0.94, "p99.9 = {}", stats.p99_9);
    }

    #[test]
    fn auto_scene_metering_bounds_p99_near_headroom() {
        // Build a buffer with a bright tail that the old clamp (6.0x)
        // would happily boost past paper white. The tightened bounds
        // plus the p99 cap must keep the top percentile well-controlled.
        let mut buffer: PixelBuffer = Vec::with_capacity(1000);
        for i in 0..1000 {
            let a = 1.0;
            let lum = if i < 990 { 0.05 } else { 1.5 }; // 99% dim, 1% hot
            buffer.push((lum * a, lum * a, lum * a, a));
        }
        apply_auto_scene_metering(&mut buffer);
        // Measure p99 of the post-metering un-premul luminance.
        let mut lums: Vec<f64> = buffer
            .iter()
            .map(|p| {
                let (r, g, b) = unpremul(*p);
                constants::rec709_luminance(r, g, b)
            })
            .collect();
        lums.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p99 = lums[(lums.len() as f64 * 0.99) as usize - 1];
        // With HIGHLIGHT_HEADROOM_P99 = 0.65 and a 1% hot tail at lum=1.5,
        // the metering must cap the post-gain p99 near (or below) 0.65.
        // Allow a modest slack for the discrete percentile.
        assert!(p99 <= 0.8, "p99 {p99} exceeded bounded headroom");
    }

    fn default_levels() -> ChannelLevels {
        ChannelLevels::new(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    }

    fn baseline_resolved_config(width: u32, height: u32) -> ResolvedEffectConfig {
        ResolvedEffectConfig {
            width,
            height,
            enable_bloom: false,
            enable_glow: false,
            enable_chromatic_bloom: false,
            enable_perceptual_blur: false,
            enable_micro_contrast: false,
            enable_gradient_map: false,
            enable_color_grade: false,
            enable_champleve: false,
            enable_aether: false,
            enable_opalescence: false,
            enable_edge_luminance: false,
            enable_atmospheric_depth: false,
            enable_fine_texture: false,
            enable_bloom_pyramid: false,
            enable_anamorphic_flare: false,
            enable_god_rays: false,
            enable_rim_light: false,
            enable_star_field: false,
            enable_diffraction_spikes: false,
            enable_airy_disc: false,
            enable_palette_harmony: false,
            enable_glaze: false,
            blur_strength: 4.0,
            blur_radius_scale: 0.006,
            blur_core_brightness: 10.0,
            dog_strength: 0.3,
            dog_sigma_scale: 0.005,
            dog_ratio: 2.6,
            glow_strength: 0.25,
            glow_threshold: 0.7,
            glow_radius_scale: 0.003,
            glow_sharpness: 2.6,
            glow_saturation_boost: 0.2,
            chromatic_bloom_strength: 0.4,
            chromatic_bloom_radius_scale: 0.005,
            chromatic_bloom_separation_scale: 0.001,
            chromatic_bloom_threshold: 0.2,
            perceptual_blur_strength: 0.45,
            color_grade_strength: 0.55,
            vignette_strength: 0.35,
            vignette_softness: 2.5,
            vibrance: 1.2,
            clarity_strength: 0.3,
            tone_curve_strength: 0.6,
            gradient_map_strength: 0.25,
            gradient_map_hue_preservation: 0.6,
            gradient_map_palette: 0,
            opalescence_strength: 0.08,
            opalescence_scale: 0.01,
            opalescence_layers: 2,
            champleve_flow_alignment: 0.6,
            champleve_interference_amplitude: 0.5,
            champleve_rim_intensity: 1.8,
            champleve_rim_warmth: 0.6,
            champleve_interior_lift: 0.65,
            aether_flow_alignment: 0.7,
            aether_scattering_strength: 0.9,
            aether_iridescence_amplitude: 0.6,
            aether_caustic_strength: 0.3,
            micro_contrast_strength: 0.25,
            micro_contrast_radius: 4,
            edge_luminance_strength: 0.3,
            edge_luminance_threshold: 0.2,
            edge_luminance_brightness_boost: 0.4,
            atmospheric_depth_strength: 0.1,
            atmospheric_desaturation: 0.12,
            atmospheric_darkening: 0.06,
            atmospheric_fog_color_r: 0.04,
            atmospheric_fog_color_g: 0.07,
            atmospheric_fog_color_b: 0.12,
            fine_texture_strength: 0.12,
            fine_texture_scale: 0.0018,
            fine_texture_contrast: 0.35,
            hdr_scale: 0.12,
            clip_black: 0.01,
            clip_white: 0.99,
        }
    }

    fn image_energy(image: &ImageBuffer<Rgb<u16>, Vec<u16>>) -> u64 {
        image.as_raw().iter().map(|&channel| u64::from(channel)).sum()
    }

    type SceneData = (Vec<Vec<Vector3<f64>>>, Vec<Vec<OklabColor>>, Vec<f64>);
    type CapturedFrameResult = (Vec<u8>, Option<ImageBuffer<Rgb<u16>, Vec<u16>>>);

    fn assert_frame_bytes_eq(actual: &[u8], expected: &[u8], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: frame byte lengths differ");
        if actual != expected {
            let first = actual
                .iter()
                .zip(expected)
                .position(|(a, b)| a != b)
                .expect("expected differing byte position");
            panic!(
                "{label}: frame bytes differ at index {first} ({} vs {})",
                actual[first], expected[first],
            );
        }
    }

    fn assert_spd_buffers_bits_eq(
        actual: &[[f64; NUM_BINS]],
        expected: &[[f64; NUM_BINS]],
        label: &str,
    ) {
        assert_eq!(actual.len(), expected.len(), "{label}: SPD buffer lengths differ");
        for (pixel_idx, (lhs, rhs)) in actual.iter().zip(expected).enumerate() {
            for (bin_idx, (&lhs_bin, &rhs_bin)) in lhs.iter().zip(rhs.iter()).enumerate() {
                assert_eq!(
                    lhs_bin.to_bits(),
                    rhs_bin.to_bits(),
                    "{label}: pixel {pixel_idx} bin {bin_idx} differed ({lhs_bin} vs {rhs_bin})"
                );
            }
        }
    }

    fn assert_histogram_bits_eq(actual: &HistogramData, expected: &HistogramData, label: &str) {
        assert_eq!(actual.data().len(), expected.data().len(), "{label}: histogram lengths differ");
        for (sample_idx, (lhs, rhs)) in actual.data().iter().zip(expected.data()).enumerate() {
            for channel_idx in 0..3 {
                assert_eq!(
                    lhs[channel_idx].to_bits(),
                    rhs[channel_idx].to_bits(),
                    "{label}: sample {sample_idx} channel {channel_idx} differed"
                );
            }
        }
    }

    fn assert_image_bits_eq(
        actual: &ImageBuffer<Rgb<u16>, Vec<u16>>,
        expected: &ImageBuffer<Rgb<u16>, Vec<u16>>,
        label: &str,
    ) {
        assert_eq!(actual.as_raw(), expected.as_raw(), "{label}: 16-bit image buffers differed");
    }

    fn sample_scene() -> SceneData {
        let positions = vec![
            vec![
                Vector3::new(0.10, 0.10, -0.30),
                Vector3::new(0.16, 0.14, -0.15),
                Vector3::new(0.24, 0.22, 0.05),
                Vector3::new(0.32, 0.28, 0.18),
                Vector3::new(0.38, 0.32, 0.30),
            ],
            vec![
                Vector3::new(0.86, 0.12, 0.24),
                Vector3::new(0.80, 0.18, 0.15),
                Vector3::new(0.72, 0.26, 0.02),
                Vector3::new(0.64, 0.34, -0.12),
                Vector3::new(0.58, 0.42, -0.24),
            ],
            vec![
                Vector3::new(0.45, 0.88, -0.18),
                Vector3::new(0.48, 0.80, -0.08),
                Vector3::new(0.52, 0.72, 0.00),
                Vector3::new(0.56, 0.64, 0.14),
                Vector3::new(0.60, 0.56, 0.26),
            ],
        ];
        let colors = vec![
            vec![
                (0.72, 0.22, 0.08),
                (0.74, 0.21, 0.10),
                (0.76, 0.19, 0.11),
                (0.78, 0.18, 0.12),
                (0.80, 0.17, 0.13),
            ],
            vec![
                (0.70, -0.18, 0.18),
                (0.72, -0.16, 0.17),
                (0.74, -0.14, 0.16),
                (0.76, -0.12, 0.15),
                (0.78, -0.10, 0.14),
            ],
            vec![
                (0.68, 0.04, -0.20),
                (0.70, 0.05, -0.18),
                (0.72, 0.06, -0.16),
                (0.74, 0.07, -0.14),
                (0.76, 0.08, -0.12),
            ],
        ];
        let body_alphas = vec![0.65, 0.85, 0.95];
        (positions, colors, body_alphas)
    }

    fn stylized_resolved_config(width: u32, height: u32) -> ResolvedEffectConfig {
        ResolvedEffectConfig {
            enable_bloom: true,
            enable_glow: true,
            enable_chromatic_bloom: true,
            enable_perceptual_blur: true,
            enable_micro_contrast: true,
            enable_gradient_map: true,
            enable_color_grade: true,
            enable_champleve: true,
            enable_aether: true,
            enable_opalescence: true,
            enable_edge_luminance: true,
            enable_atmospheric_depth: true,
            enable_fine_texture: false,
            ..baseline_resolved_config(width, height)
        }
    }

    fn derived_levels_from_serial_histogram(
        scene: SpectralScene<'_>,
        frame_interval: usize,
        settings: SpectralRenderSettings<'_>,
    ) -> ChannelLevels {
        let histogram =
            pass_1_build_histogram_spectral_serial_reference(scene, frame_interval, settings);
        let analysis = histogram::analyze_tonemapping(
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
                exposure_scale: analysis.exposure_scale,
                paper_white: constants::DEFAULT_TONEMAP_PAPER_WHITE,
                highlight_rolloff: constants::DEFAULT_TONEMAP_HIGHLIGHT_ROLLOFF,
            },
        )
    }

    fn capture_frame_bytes_with_pool(
        scene: SpectralScene<'_>,
        frame_interval: usize,
        levels: &ChannelLevels,
        settings: SpectralRenderSettings<'_>,
        enable_temporal_smoothing: bool,
        serial_reference: bool,
        thread_count: usize,
    ) -> CapturedFrameResult {
        let mut frame_bytes = Vec::new();
        let mut last_frame = None;

        ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build()
            .expect("thread pool should build")
            .install(|| {
                let frame_sink = |bytes: &[u8]| {
                    frame_bytes.extend_from_slice(bytes);
                    Ok(())
                };
                let mut spd_buf = Vec::new();

                let params = Pass2Params {
                    scene,
                    frame_interval,
                    levels,
                    settings,
                    last_frame_out: &mut last_frame,
                    enable_temporal_smoothing,
                    accum_spd: &mut spd_buf,
                };
                if serial_reference {
                    pass_2_write_frames_spectral_serial_reference(params, frame_sink)
                } else {
                    pass_2_write_frames_spectral(params, frame_sink)
                }
            })
            .expect("frame rendering should succeed");

        (frame_bytes, last_frame)
    }

    #[test]
    fn test_tonemap_black_produces_black() {
        let result = tonemap_core(0.0, 0.0, 0.0, 0.0, &default_levels());
        assert_eq!(result, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_tonemap_produces_valid_range() {
        let levels = default_levels();
        for alpha in [0.1, 0.5, 1.0] {
            let result = tonemap_core(0.5, 0.3, 0.8, alpha, &levels);
            for ch in result {
                assert!(ch >= 0.0, "channel {ch} should be non-negative at alpha {alpha}");
                assert!(ch < 2.0, "channel {ch} unreasonably large at alpha {alpha}");
            }
        }
    }

    #[test]
    fn test_tonemap_reserves_paper_white_headroom() {
        let levels = ChannelLevels::with_tone_mapping(
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            crate::render::types::ToneMappingControls {
                exposure_scale: 1.0,
                paper_white: 0.9,
                highlight_rolloff: 2.5,
            },
        );
        let result = tonemap_core(8.0, 8.0, 8.0, 1.0, &levels);

        // OKLab luminance tonemap + soft highlight shoulder asymptotes
        // below 1.0 and retains perceived brightness. A scene-linear
        // (8,8,8) input is 8x the display-white reference, so the
        // hue-preserving tonemap intentionally leaves it at roughly mid
        // brightness (~0.4-0.6) with plenty of headroom for brighter
        // highlights above it — rather than crushing it to 0.99 which
        // would lose every perceivable specular accent.
        assert!(result[0] < 1.0);
        assert!(result[1] < 1.0);
        assert!(result[2] < 1.0);
        assert!(
            result[0] > 0.35,
            "should still register as a lit pixel after compression (got {result:?})"
        );
    }

    #[test]
    fn test_tonemap_exposure_scale_reduces_hot_input() {
        let unity = ChannelLevels::with_tone_mapping(
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            crate::render::types::ToneMappingControls {
                exposure_scale: 1.0,
                paper_white: 0.92,
                highlight_rolloff: 2.25,
            },
        );
        let reduced = ChannelLevels::with_tone_mapping(
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            crate::render::types::ToneMappingControls {
                exposure_scale: 0.6,
                paper_white: 0.92,
                highlight_rolloff: 2.25,
            },
        );

        let unity_out = tonemap_core(2.0, 2.0, 2.0, 1.0, &unity);
        let reduced_out = tonemap_core(2.0, 2.0, 2.0, 1.0, &reduced);

        assert!(reduced_out[0] < unity_out[0]);
        assert!(reduced_out[1] < unity_out[1]);
        assert!(reduced_out[2] < unity_out[2]);
    }

    #[test]
    fn test_agx_tweak_changes_output() {
        let levels = default_levels();

        ACES_TWEAK_ENABLED.store(true, Ordering::Relaxed);
        let tweaked = tonemap_core(0.5, 0.3, 0.7, 0.8, &levels);

        ACES_TWEAK_ENABLED.store(false, Ordering::Relaxed);
        let original = tonemap_core(0.5, 0.3, 0.7, 0.8, &levels);

        ACES_TWEAK_ENABLED.store(true, Ordering::Relaxed);

        let diff = (tweaked[0] - original[0]).abs()
            + (tweaked[1] - original[1]).abs()
            + (tweaked[2] - original[2]).abs();
        assert!(diff > 1e-6, "AgX punchy tweak should produce different tonemapping");
    }

    #[test]
    fn test_tonemap_16bit_range() {
        let levels = default_levels();
        let result = tonemap_to_16bit(0.5, 0.4, 0.6, 0.9, &levels);
        for ch in result {
            assert!(u32::from(ch) <= 65535, "16-bit channel {ch} out of range");
        }
    }

    #[test]
    fn test_build_effect_config_uses_dog_bloom_exclusively() {
        let resolved =
            ResolvedEffectConfig { enable_bloom: true, ..baseline_resolved_config(640, 360) };
        let render_config =
            RenderConfig { hdr_scale: resolved.hdr_scale, bloom_mode: BloomMode::Dog };

        let effect_config = build_effect_config_from_resolved(
            &resolved,
            &render_config,
            FinishOutputMode::Still,
            0,
        );

        assert_eq!(effect_config.bloom_mode, "dog");
        assert_eq!(effect_config.blur_radius_px, 0);
        assert!(effect_config.dog_config.inner_sigma > 0.0);
    }

    #[test]
    fn test_build_effect_config_uses_gaussian_bloom_exclusively() {
        let resolved =
            ResolvedEffectConfig { enable_bloom: true, ..baseline_resolved_config(640, 360) };
        let render_config =
            RenderConfig { hdr_scale: resolved.hdr_scale, bloom_mode: BloomMode::Gaussian };

        let effect_config = build_effect_config_from_resolved(
            &resolved,
            &render_config,
            FinishOutputMode::Still,
            0,
        );

        assert_eq!(effect_config.bloom_mode, "gaussian");
        assert!(effect_config.blur_radius_px > 0);
    }

    #[test]
    fn test_build_effect_config_disables_texture_for_proxy_resolution() {
        let resolved = ResolvedEffectConfig {
            enable_fine_texture: true,
            ..baseline_resolved_config(640, 360)
        };
        let render_config =
            RenderConfig { hdr_scale: resolved.hdr_scale, bloom_mode: BloomMode::Dog };

        let effect_config = build_effect_config_from_resolved(
            &resolved,
            &render_config,
            FinishOutputMode::Still,
            0,
        );

        assert!(!effect_config.fine_texture_enabled, "proxy-sized renders should skip texture");
    }

    #[test]
    fn test_build_effect_config_scales_texture_for_video() {
        let resolved = ResolvedEffectConfig {
            enable_fine_texture: true,
            fine_texture_strength: 0.2,
            ..baseline_resolved_config(1920, 1080)
        };
        let render_config =
            RenderConfig { hdr_scale: resolved.hdr_scale, bloom_mode: BloomMode::Dog };

        let still_config = build_effect_config_from_resolved(
            &resolved,
            &render_config,
            FinishOutputMode::Still,
            0,
        );
        let video_config = build_effect_config_from_resolved(
            &resolved,
            &render_config,
            FinishOutputMode::Video,
            0,
        );

        assert!(still_config.fine_texture_enabled);
        assert!(video_config.fine_texture_enabled);
        assert!(
            (video_config.fine_texture_config.strength
                - still_config.fine_texture_config.strength * 0.6)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn test_build_effect_config_tightens_softness_stack_settings() {
        let resolved = ResolvedEffectConfig {
            enable_bloom: true,
            enable_glow: true,
            enable_chromatic_bloom: true,
            enable_perceptual_blur: true,
            ..baseline_resolved_config(1920, 1080)
        };
        let render_config =
            RenderConfig { hdr_scale: resolved.hdr_scale, bloom_mode: BloomMode::Dog };

        let effect_config = build_effect_config_from_resolved(
            &resolved,
            &render_config,
            FinishOutputMode::Still,
            0,
        );
        let perceptual =
            effect_config.perceptual_blur_config.expect("perceptual blur should remain configured");

        assert!(
            effect_config.dog_config.threshold > 0.012,
            "softness stacks should raise the DoG threshold"
        );
        assert!(
            perceptual.radius < (0.0036_f64 * 1080.0).round() as usize,
            "softness stacks should tighten perceptual blur radius"
        );
    }

    #[test]
    fn test_scanline_accumulation_matches_serial_reference_bits() {
        let (positions, colors, body_alphas) = sample_scene();
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let render_config = RenderConfig { hdr_scale: 3.5, bloom_mode: BloomMode::None };
        let ctx = RenderContext::new(9, 7, &positions, false);
        let velocity_calc =
            velocity_hdr::VelocityHdrCalculator::new(&positions, constants::DEFAULT_DT);

        let accum_params = AccumulationParams {
            scene,
            ctx: &ctx,
            velocity_calc: &velocity_calc,
            step_start: 0,
            step_end: scene.step_count(),
            hdr_scale: render_config.hdr_scale,
        };

        let mut serial = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
        accumulate_spectral_steps(&mut serial, &accum_params, AccumulationBackend::SerialReference);

        for thread_count in [1usize, 2, 3, ctx.height_usize] {
            let mut parallel = vec![[0.0; NUM_BINS]; ctx.pixel_count()];
            ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build()
                .expect("thread pool should build")
                .install(|| {
                    accumulate_spectral_steps(
                        &mut parallel,
                        &accum_params,
                        AccumulationBackend::ParallelScanlines,
                    );
                });
            assert_spd_buffers_bits_eq(
                &parallel,
                &serial,
                &format!("accumulation/threads={thread_count}"),
            );
        }
    }

    #[test]
    fn test_histogram_pass_is_finish_aware() {
        let positions = vec![
            vec![Vector3::new(0.1, 0.1, 0.0), Vector3::new(0.2, 0.2, 0.0)],
            vec![Vector3::new(0.9, 0.1, 0.0), Vector3::new(0.8, 0.2, 0.0)],
            vec![Vector3::new(0.5, 0.9, 0.0), Vector3::new(0.5, 0.8, 0.0)],
        ];
        let colors = vec![
            vec![(0.7, 0.2, 0.1), (0.72, 0.18, 0.12)],
            vec![(0.68, -0.15, 0.2), (0.70, -0.12, 0.18)],
            vec![(0.65, 0.04, -0.18), (0.67, 0.05, -0.16)],
        ];
        let body_alphas = vec![0.8, 0.9, 1.0];
        let render_config = RenderConfig { hdr_scale: 3.0, bloom_mode: BloomMode::Dog };

        let clean = baseline_resolved_config(48, 48);
        let stylized = ResolvedEffectConfig {
            enable_bloom: true,
            enable_glow: true,
            enable_chromatic_bloom: true,
            enable_perceptual_blur: true,
            enable_micro_contrast: true,
            enable_gradient_map: true,
            enable_color_grade: true,
            enable_champleve: true,
            enable_aether: true,
            enable_opalescence: true,
            enable_edge_luminance: true,
            enable_atmospheric_depth: true,
            enable_fine_texture: true,
            ..baseline_resolved_config(48, 48)
        };

        let clean_hist = pass_1_build_histogram_spectral(
            SpectralScene::new(&positions, &colors, &body_alphas),
            1,
            SpectralRenderSettings::new(&clean, &render_config, 17, false),
        );

        let styled_hist = pass_1_build_histogram_spectral(
            SpectralScene::new(&positions, &colors, &body_alphas),
            1,
            SpectralRenderSettings::new(&stylized, &render_config, 999, false),
        );

        assert_ne!(clean_hist.data(), styled_hist.data());
    }

    #[test]
    fn test_histogram_pass_parallel_matches_serial_reference_bits() {
        let (positions, colors, body_alphas) = sample_scene();
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let resolved = baseline_resolved_config(64, 40);
        let render_config = RenderConfig { hdr_scale: 2.8, bloom_mode: BloomMode::Dog };
        let settings = SpectralRenderSettings::new(&resolved, &render_config, 19, false);
        let serial = pass_1_build_histogram_spectral_serial_reference(scene, 2, settings);

        for thread_count in [1usize, 2, 3, resolved.height as usize] {
            let parallel = ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build()
                .expect("thread pool should build")
                .install(|| pass_1_build_histogram_spectral(scene, 2, settings));
            assert_histogram_bits_eq(
                &parallel,
                &serial,
                &format!("histogram/threads={thread_count}"),
            );
        }
    }

    #[test]
    fn test_render_final_frame_accumulates_late_color_steps() {
        let resolved = baseline_resolved_config(48, 48);
        let render_config = RenderConfig { hdr_scale: 6.0, bloom_mode: BloomMode::None };
        let positions = vec![
            vec![
                Vector3::new(0.1, 0.1, 0.0),
                Vector3::new(0.1, 0.1, 0.0),
                Vector3::new(0.1, 0.1, 0.0),
                Vector3::new(0.1, 0.1, 0.0),
            ],
            vec![
                Vector3::new(0.9, 0.1, 0.0),
                Vector3::new(0.9, 0.1, 0.0),
                Vector3::new(0.9, 0.1, 0.0),
                Vector3::new(0.9, 0.1, 0.0),
            ],
            vec![
                Vector3::new(0.5, 0.9, 0.0),
                Vector3::new(0.5, 0.9, 0.0),
                Vector3::new(0.5, 0.9, 0.0),
                Vector3::new(0.5, 0.9, 0.0),
            ],
        ];
        let colors = vec![
            vec![(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.82, 0.22, 0.08), (0.82, 0.22, 0.08)],
            vec![(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.78, -0.18, 0.15), (0.78, -0.18, 0.15)],
            vec![(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.74, 0.05, -0.22), (0.74, 0.05, -0.22)],
        ];
        let body_alphas = vec![1.0, 1.0, 1.0];
        let levels = ChannelLevels::new(0.0, 0.05, 0.0, 0.05, 0.0, 0.05);

        let single_frame = render_single_frame_spectral(
            SpectralScene::new(&positions, &colors, &body_alphas),
            &levels,
            SpectralRenderSettings::new(&resolved, &render_config, 7, false),
        )
        .expect("legacy single-frame preview should render");
        let final_frame = render_final_frame_spectral(
            SpectralScene::new(&positions, &colors, &body_alphas),
            &levels,
            SpectralRenderSettings::new(&resolved, &render_config, 7, false),
        )
        .expect("final preview should render");

        let single_energy = image_energy(&single_frame);
        let final_energy = image_energy(&final_frame);

        assert!(final_energy > 0, "final preview should contain visible energy");
        // The OKLab-space tonemap + per-channel Reinhard in `finalize_rgba`
        // preserve relative brightness proportions, so both previews now
        // register appreciable energy even from a handful of steps. The
        // invariant that still matters is that the final preview picks up
        // the late colorful steps the single-frame preview misses; we
        // check it's at least ~20% richer rather than 20x.
        assert!(
            final_energy > single_energy + single_energy / 5,
            "final preview should retain meaningfully more energy than the legacy early-slice preview (single={single_energy}, final={final_energy})"
        );
    }

    #[test]
    fn test_render_previews_parallel_match_serial_reference_bits() {
        let (positions, colors, body_alphas) = sample_scene();
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let resolved = baseline_resolved_config(64, 40);
        let render_config = RenderConfig { hdr_scale: 3.2, bloom_mode: BloomMode::None };
        let settings = SpectralRenderSettings::new(&resolved, &render_config, 29, false);
        let levels = ChannelLevels::new(0.0, 0.12, 0.0, 0.12, 0.0, 0.12);

        let serial_single = render_single_frame_spectral_serial_reference(scene, &levels, settings)
            .expect("serial single frame render should succeed");
        let serial_final = render_final_frame_spectral_serial_reference(scene, &levels, settings)
            .expect("serial final frame render should succeed");

        for thread_count in [1usize, 2, 3, resolved.height as usize] {
            let (parallel_single, parallel_final) = ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build()
                .expect("thread pool should build")
                .install(|| {
                    (
                        render_single_frame_spectral(scene, &levels, settings)
                            .expect("parallel single frame render should succeed"),
                        render_final_frame_spectral(scene, &levels, settings)
                            .expect("parallel final frame render should succeed"),
                    )
                });
            assert_image_bits_eq(
                &parallel_single,
                &serial_single,
                &format!("single-preview/threads={thread_count}"),
            );
            assert_image_bits_eq(
                &parallel_final,
                &serial_final,
                &format!("final-preview/threads={thread_count}"),
            );
        }
    }

    #[test]
    fn test_video_frame_stream_parallel_matches_serial_reference_bits() {
        let (positions, colors, body_alphas) = sample_scene();
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let resolved = baseline_resolved_config(64, 40);
        let render_config = RenderConfig { hdr_scale: 3.0, bloom_mode: BloomMode::None };
        let settings = SpectralRenderSettings::new(&resolved, &render_config, 31, false);
        let frame_interval = 1usize;
        let levels = derived_levels_from_serial_histogram(scene, frame_interval, settings);

        let (serial_frames, serial_last_frame) =
            capture_frame_bytes_with_pool(scene, frame_interval, &levels, settings, false, true, 1);
        let serial_last_frame = serial_last_frame.expect("serial path should capture last frame");

        for thread_count in [1usize, 2, 3, 4] {
            let (parallel_frames, parallel_last_frame) = capture_frame_bytes_with_pool(
                scene,
                frame_interval,
                &levels,
                settings,
                false,
                false,
                thread_count,
            );
            let parallel_last_frame =
                parallel_last_frame.expect("parallel path should capture last frame");
            assert_frame_bytes_eq(
                &parallel_frames,
                &serial_frames,
                &format!("video-frames/plain/threads={thread_count}"),
            );
            assert_image_bits_eq(
                &parallel_last_frame,
                &serial_last_frame,
                &format!("video-last-frame/plain/threads={thread_count}"),
            );
        }
    }

    #[test]
    fn test_stylized_video_frame_stream_parallel_matches_serial_reference_bits() {
        let (positions, colors, body_alphas) = sample_scene();
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let resolved = stylized_resolved_config(96, 72);
        let render_config = RenderConfig { hdr_scale: 4.2, bloom_mode: BloomMode::Dog };
        let settings = SpectralRenderSettings::new(&resolved, &render_config, 57, false);
        let frame_interval = 2usize;
        let levels = derived_levels_from_serial_histogram(scene, frame_interval, settings);

        let (serial_frames, serial_last_frame) =
            capture_frame_bytes_with_pool(scene, frame_interval, &levels, settings, true, true, 1);
        let serial_last_frame = serial_last_frame.expect("serial path should capture last frame");

        for thread_count in [1usize, 2, 3, 4] {
            let (parallel_frames, parallel_last_frame) = capture_frame_bytes_with_pool(
                scene,
                frame_interval,
                &levels,
                settings,
                true,
                false,
                thread_count,
            );
            let parallel_last_frame =
                parallel_last_frame.expect("parallel path should capture last frame");
            assert_frame_bytes_eq(
                &parallel_frames,
                &serial_frames,
                &format!("video-frames/stylized/threads={thread_count}"),
            );
            assert_image_bits_eq(
                &parallel_last_frame,
                &serial_last_frame,
                &format!("video-last-frame/stylized/threads={thread_count}"),
            );
        }
    }

    #[test]
    fn test_rayon_pool_respects_custom_stack_size() {
        let pool = ThreadPoolBuilder::new()
            .stack_size(constants::THREAD_STACK_SIZE)
            .num_threads(2)
            .build()
            .expect("pool with THREAD_STACK_SIZE should build");

        let result = pool.install(|| {
            let mut v = vec![0u64; 1024];
            for (i, slot) in v.iter_mut().enumerate() {
                *slot = i as u64;
            }
            v.iter().sum::<u64>()
        });
        assert_eq!(result, (0..1024u64).sum::<u64>());
    }

    #[test]
    fn test_compute_softness_radius_disabled() {
        let mut cfg = baseline_resolved_config(1920, 1080);
        cfg.enable_perceptual_blur = false;
        assert!(compute_softness_radius(&cfg, BloomMode::Dog).is_none());
    }

    #[test]
    fn test_compute_softness_radius_enabled_returns_some() {
        let mut cfg = baseline_resolved_config(1920, 1080);
        cfg.enable_perceptual_blur = true;
        let radius = compute_softness_radius(&cfg, BloomMode::Dog);
        assert!(radius.is_some());
        assert!(radius.expect("softness radius should be some") >= 1);
    }

    #[test]
    fn test_compute_softness_radius_high_softness_uses_smaller_scale() {
        let mut low = baseline_resolved_config(1920, 1080);
        low.enable_perceptual_blur = true;

        let mut high = low.clone();
        high.enable_chromatic_bloom = true;
        high.enable_glow = true;
        high.enable_atmospheric_depth = true;

        let r_low = compute_softness_radius(&low, BloomMode::Dog)
            .expect("low softness radius should resolve");
        let r_high = compute_softness_radius(&high, BloomMode::Dog)
            .expect("high softness radius should resolve");
        assert!(
            r_high <= r_low,
            "higher softness stack should produce equal or smaller radius: {r_high} vs {r_low}",
        );
    }

    /// Regression: when a compact, very bright subject is embedded in a
    /// mostly-dark frame (the stacked-Cinematic-bloom failure mode that
    /// produced the pure-white blob on seed `0xb6dbcf780360`), the mean
    /// luminance alone is not enough to defend the highlights. The 99th
    /// percentile headroom cap in `apply_auto_scene_metering` must pull
    /// the bright tail back below the tonemap's paper-white ceiling.
    #[test]
    fn auto_metering_caps_peak_below_clip() {
        // 1920x1080 buffer: 99.5% dim background + 0.5% ultra-bright cluster.
        // All pixels carry alpha=1 so both tiers contribute to the mean and
        // to the percentile sampling.
        let w = 1920usize;
        let h = 1080usize;
        let total = w * h;
        let mut buffer: PixelBuffer = vec![(0.05, 0.05, 0.05, 1.0); total];
        let hot_count = total / 200; // 0.5%
        let hot = (12.0, 12.0, 12.0, 1.0); // scene-linear 12x display-white
        for pixel in buffer.iter_mut().take(hot_count) {
            *pixel = hot;
        }

        apply_auto_scene_metering(&mut buffer);

        // Find the post-metering 99th-percentile luminance.
        let mut lums: Vec<f64> = buffer
            .iter()
            .filter(|(_, _, _, a)| *a > 1e-9)
            .map(|&(r, g, b, a)| constants::rec709_luminance(r / a, g / a, b / a))
            .collect();
        let p99_idx = ((lums.len() as f64) * 0.99).floor() as usize;
        let p99_idx = p99_idx.min(lums.len() - 1);
        lums.select_nth_unstable_by(p99_idx, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let p99 = lums[p99_idx];

        // Paper-white is 0.85; the tightened HIGHLIGHT_HEADROOM pins p99
        // at (not past) the tonemap shoulder so `compress_display_highlights`
        // never engages its asymptote-at-1.0 rolloff on the top 1%. A small
        // floating-point slack (1e-6) absorbs the percentile sampling's
        // strided quantisation.
        assert!(
            p99 <= 0.85 + 1e-6,
            "auto-metering failed to cap peak: p99={p99} (should be <= 0.85)"
        );
        // The dim background shouldn't be completely crushed — it should
        // remain visible (non-zero) after the guard has pulled highlights
        // down.
        let bg_lum =
            constants::rec709_luminance(buffer[total - 1].0, buffer[total - 1].1, buffer[total - 1].2);
        assert!(bg_lum > 0.0, "background crushed to zero: {bg_lum}");
    }

    /// Regression: the safety shoulder must cap every channel strictly
    /// below the quantize clamp so the hard `.clamp(0.0, 1.0)` in
    /// [`quantize_display_buffer_to_16bit`] never engages on any channel.
    #[test]
    fn safety_shoulder_caps_hot_channels_below_ceiling() {
        let rolled =
            compress_display_safety_shoulder([1.4, 1.2, 1.1], SAFETY_SHOULDER_KNEE, SAFETY_SHOULDER_CEILING);
        for (i, &ch) in rolled.iter().enumerate() {
            assert!(
                ch <= SAFETY_SHOULDER_CEILING + 1e-9,
                "channel {i} = {ch} exceeded ceiling {SAFETY_SHOULDER_CEILING}"
            );
        }
    }

    /// Regression: the safety shoulder must preserve hue when scaling down
    /// out-of-range pixels. Per-channel hard clamping (the bug) produces
    /// pure primary-colour splotches; the fix must keep the R:G ratio
    /// approximately stable on a saturated red pixel.
    #[test]
    fn safety_shoulder_preserves_hue_on_saturated_pixels() {
        let input = [2.0, 0.5, 0.0];
        let ratio_in = input[0] / input[1];
        let rolled =
            compress_display_safety_shoulder(input, SAFETY_SHOULDER_KNEE, SAFETY_SHOULDER_CEILING);
        assert!(
            rolled[1] > 1e-6,
            "green channel crushed to zero, losing hue: {rolled:?}"
        );
        let ratio_out = rolled[0] / rolled[1];
        // Uniform scaling by `ceiling / max_channel` preserves hue exactly;
        // the only drift is floating-point rounding.
        let drift = (ratio_out / ratio_in - 1.0).abs();
        assert!(drift < 1e-6, "hue drift too large: in={ratio_in}, out={ratio_out}");
    }

    /// Regression: pixels below the knee must pass through the safety
    /// shoulder untouched, so well-behaved seeds keep their exact look.
    #[test]
    fn safety_shoulder_passes_well_behaved_pixels_unchanged() {
        let input = [0.4, 0.3, 0.2];
        let rolled =
            compress_display_safety_shoulder(input, SAFETY_SHOULDER_KNEE, SAFETY_SHOULDER_CEILING);
        for i in 0..3 {
            assert!(
                (rolled[i] - input[i]).abs() < 1e-9,
                "well-behaved pixel channel {i} shifted: {} -> {}",
                input[i],
                rolled[i]
            );
        }
    }

    /// Regression: on a well-behaved scene (no extreme highlights, modest
    /// mean), the headroom guard must NOT tighten `auto` — otherwise we'd
    /// darken already-good seeds. Verify that a flat, middle-grey-ish
    /// buffer comes back approximately unchanged.
    #[test]
    fn auto_metering_leaves_balanced_scene_alone() {
        let total = 1920 * 1080;
        // Flat alpha-weighted luminance around the target of 0.14.
        let mut buffer: PixelBuffer = vec![(0.14, 0.14, 0.14, 1.0); total];
        apply_auto_scene_metering(&mut buffer);

        // Mean was already at target, and p99 equals the mean, so neither
        // guard should move pixels measurably.
        let lum = constants::rec709_luminance(buffer[0].0, buffer[0].1, buffer[0].2);
        assert!(
            (lum - 0.14).abs() < 0.02,
            "balanced scene unexpectedly shifted: post-gain lum={lum}"
        );
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig {
            cases: 64,
            .. proptest::prelude::ProptestConfig::default()
        })]

        // Pixel sanity: for any combination of (dim mean, bright hot
        // fraction, hot channel intensity), post-metering p99.9
        // luminance must land below the documented scene-linear
        // ceiling. Guarantees that the four stacked percentile caps
        // (p99 / p99.5 / p99.9 / mean) leave no reachable input that
        // sends the top 0.1 % past the tonemap shoulder.
        #[test]
        fn proptest_auto_metering_caps_tail_below_ceiling(
            dim_lum in 0.0f64..0.3,
            hot_fraction in 0.001f64..0.05,
            hot_channel_peak in 1.5f64..40.0,
        ) {
            let w = 640usize;
            let h = 360usize;
            let total = w * h;
            let mut buffer: PixelBuffer = vec![(dim_lum, dim_lum, dim_lum, 1.0); total];
            let hot_count = ((hot_fraction * total as f64) as usize).max(1);
            let hot = (hot_channel_peak, hot_channel_peak, hot_channel_peak, 1.0);
            for pixel in buffer.iter_mut().take(hot_count) {
                *pixel = hot;
            }

            apply_auto_scene_metering(&mut buffer);

            // Compute the post-metering p99.9 luminance.
            let mut lums: Vec<f64> = buffer
                .iter()
                .filter(|(_, _, _, a)| *a > 1e-9)
                .map(|&(r, g, b, a)| constants::rec709_luminance(r / a, g / a, b / a))
                .collect();
            let p999_idx = ((lums.len() as f64) * 0.999).floor() as usize;
            let p999_idx = p999_idx.min(lums.len() - 1);
            lums.select_nth_unstable_by(p999_idx, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            let p999 = lums[p999_idx];

            // HIGHLIGHT_HEADROOM_P999 is 1.2; allow a small slack for
            // the percentile stride's quantisation of the measured tail.
            proptest::prop_assert!(
                p999 <= 1.25,
                "post-metering p99.9={p999} exceeded ceiling 1.25 \
                 (dim={dim_lum}, hot_fraction={hot_fraction}, peak={hot_channel_peak})",
            );
        }
    }

    /// Regression: `energy_density_shifted_spd` must be a pure function of
    /// its input. The older mutating `apply_energy_density_shift(&mut ...)`
    /// was called once per video frame on the persistent accumulator, so
    /// repeated application compounded the shift and migrated all hot
    /// spectral mass into `bin[NUM_BINS-1]`. The new helper must return
    /// the same value no matter how many times it is called.
    #[test]
    fn energy_density_shift_is_idempotent_on_immutable_input() {
        let mut spd = [0.0f64; NUM_BINS];
        // Hot pixel: concentrate total_energy = 1.0 in a mid bin.
        spd[32] = 1.0;
        let first = energy_density_shifted_spd(&spd);
        for _ in 0..5 {
            let again = energy_density_shifted_spd(&spd);
            assert_eq!(again, first, "repeated calls produced different output");
        }
        // And the source itself must never have been touched.
        assert_eq!(spd[32], 1.0);
    }

    /// Regression: cool pixels (below the threshold) must pass through the
    /// shift with bit-identical output, so the background isn't nudged.
    #[test]
    fn energy_density_shift_leaves_cold_pixels_untouched() {
        let mut spd = [0.0f64; NUM_BINS];
        // Total energy 0.2, well below `ENERGY_DENSITY_SHIFT_THRESHOLD`.
        spd[10] = 0.1;
        spd[20] = 0.1;
        let shifted = energy_density_shifted_spd(&spd);
        assert_eq!(shifted, spd, "cold pixel unexpectedly shifted");
    }

    /// Regression: for a hot pixel, the shift must actually move mass
    /// toward the higher (red-end) bins. Checks the physical direction of
    /// the artistic redshift.
    #[test]
    fn energy_density_shift_moves_energy_redward_for_hot_pixels() {
        let mut spd = [0.0f64; NUM_BINS];
        // Total energy 2.0 — far above the threshold, guarantees a shift.
        let source_bin = 30;
        spd[source_bin] = 2.0;
        let shifted = energy_density_shifted_spd(&spd);
        assert!(
            shifted[source_bin] < spd[source_bin],
            "source bin did not shed energy: before={}, after={}",
            spd[source_bin],
            shifted[source_bin]
        );
        assert!(
            shifted[source_bin + 1] > spd[source_bin + 1],
            "next redder bin did not receive energy: before={}, after={}",
            spd[source_bin + 1],
            shifted[source_bin + 1]
        );
        // Energy is approximately conserved (one-bin diffusion).
        let before: f64 = spd.iter().sum();
        let after: f64 = shifted.iter().sum();
        assert!(
            (before - after).abs() < 0.35,
            "bulk energy drift too large: before={before}, after={after}"
        );
    }
}
