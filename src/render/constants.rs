//! Constants used throughout the render module
//!
//! This module contains all numeric constants used in rendering operations,
//! color space conversions, and video encoding. Each constant is documented
//! with its purpose and typical usage range.

// ========== Thread Configuration ==========

/// Stack size for worker threads (Rayon pool and scoped encoding threads).
///
/// Rust's default is 2 MiB, which is insufficient for seeds that enable
/// many post-effects simultaneously. 512 MiB provides a 256x safety margin.
/// Only pages actually touched consume physical RAM (Linux lazy allocation),
/// so the real memory cost is negligible.
pub const THREAD_STACK_SIZE: usize = 512 * 1024 * 1024;

// ========== Color Generation Constants ==========

/// Degrees in a full rotation
pub const HUE_FULL_CIRCLE: f64 = 360.0;

/// Separation between body hues (360/3 for even distribution)
/// This ensures the three bodies have maximally separated base colors
pub const BODY_HUE_SEPARATION: f64 = 120.0;

/// Controls drift rate of hue over time (higher = more palette movement)
pub const HUE_DRIFT_SCALE: f64 = 1.15;

/// Base time drift factor for subtle color evolution
pub const BASE_HUE_DRIFT: f64 = 1.4;

/// Amplitude, in degrees, applied by the palette sway wave
pub const HUE_WAVE_AMPLITUDE: f64 = 34.0;

/// Additional per-body phase offsets (degrees) to guarantee separation
pub const BODY_HUE_PHASE: [f64; 3] = [0.0, 120.0, 240.0];

// ========== OKLab Perceptual Color Space Constants ==========

/// Base chroma value (typical range 0-0.3 for natural colors)
pub const OKLAB_CHROMA_BASE: f64 = 0.18;
/// Boosted base chroma for museum-quality output
pub const OKLAB_CHROMA_BASE_BOOSTED: f64 = 0.24;

/// Range of chroma variation around the base value
pub const OKLAB_CHROMA_RANGE: f64 = 0.12;
/// Boosted chroma range
pub const OKLAB_CHROMA_RANGE_BOOSTED: f64 = 0.10;

/// Additional chroma modulation applied via palette waves
pub const OKLAB_CHROMA_WAVE_AMPLITUDE: f64 = 0.07;
/// Boosted chroma wave amplitude
pub const OKLAB_CHROMA_WAVE_AMPLITUDE_BOOSTED: f64 = 0.06;

/// Base lightness value (0=black, 1=white)
pub const OKLAB_LIGHTNESS_BASE: f64 = 0.68;

/// Range of lightness variation around the base value
pub const OKLAB_LIGHTNESS_RANGE: f64 = 0.22;

/// Additional lightness modulation applied via palette waves
pub const OKLAB_LIGHTNESS_WAVE_AMPLITUDE: f64 = 0.18;

// ========== Rendering Constants ==========

/// Default HDR scale factor when HDR mode is disabled
pub const DEFAULT_HDR_SCALE: f64 = 1.0;

/// Pre-tonemap luminance target for the solved white percentile.
/// Values above this still retain headroom for specular accents.
pub const DEFAULT_PRETONEMAP_LUMA_TARGET: f64 = 0.88;

/// Samples above this normalized luminance are considered near-clipped during proxy analysis.
pub const DEFAULT_PRETONEMAP_NEAR_CLIP_THRESHOLD: f64 = 1.10;

/// Maximum tolerated fraction of near-clipped proxy samples before the governor darkens exposure.
pub const DEFAULT_PRETONEMAP_NEAR_CLIP_BUDGET: f64 = 0.0025;

/// Response multiplier for the highlight budget governor.
pub const DEFAULT_PRETONEMAP_BUDGET_RESPONSE: f64 = 1.5;

/// Lower clamp for the global exposure scale derived from proxy analysis.
pub const DEFAULT_MIN_EXPOSURE_SCALE: f64 = 0.35;

/// Display-space luminance reserved for "paper white".
pub const DEFAULT_TONEMAP_PAPER_WHITE: f64 = 0.92;

/// Strength of the luminance-preserving shoulder above paper white.
pub const DEFAULT_TONEMAP_HIGHLIGHT_ROLLOFF: f64 = 2.25;

/// Bilinear interpolation averaging factor (1/4 for 4 samples)
pub const BILINEAR_AVG_FACTOR: f64 = 0.25;

/// Edge extension for bounding box to ensure all particles are visible
pub const BOUNDING_BOX_PADDING: f64 = 0.5;

/// Sigma calculation factor for Gaussian blur (radius/3)
pub const GAUSSIAN_SIGMA_FACTOR: f64 = 3.0;

/// Minimum sigma value to prevent division by zero
pub const GAUSSIAN_SIGMA_MIN: f64 = 1.0;

/// Factor for two-sigma-squared calculation in Gaussian
pub const GAUSSIAN_TWO_FACTOR: f64 = 2.0;

/// Highlight threshold for bloom/glow residual extraction in linear space.
pub const DEFAULT_HIGHLIGHT_EXTRACT_THRESHOLD: f64 = 0.58;

/// Width of the soft-knee used for highlight residual extraction.
pub const DEFAULT_HIGHLIGHT_EXTRACT_KNEE: f64 = 0.18;

/// Default strength for the cinematic color grading effect (0-1)
pub const DEFAULT_COLOR_GRADE_STRENGTH: f64 = 0.48;

/// Default vignette strength for color grading (0-1)
pub const DEFAULT_COLOR_GRADE_VIGNETTE: f64 = 0.45;

/// Default vignette softness exponent (> 1.0)
pub const DEFAULT_COLOR_GRADE_VIGNETTE_SOFTNESS: f64 = 2.6;

/// Default vibrance boost applied during color grading
pub const DEFAULT_COLOR_GRADE_VIBRANCE: f64 = 1.12;

/// Default clarity strength (high-pass contrast) during color grading
pub const DEFAULT_COLOR_GRADE_CLARITY: f64 = 0.30;

/// Default tone curve strength for midtone contrast shaping
pub const DEFAULT_COLOR_GRADE_TONE_CURVE: f64 = 0.55;

/// Default cool tint added to shadows during color grading (linear RGB deltas)
pub const DEFAULT_COLOR_GRADE_SHADOW_TINT: [f64; 3] = [-0.08, -0.02, 0.16];

/// Default warm tint added to highlights during color grading (linear RGB deltas)
pub const DEFAULT_COLOR_GRADE_HIGHLIGHT_TINT: [f64; 3] = [0.11, 0.05, -0.03];

/// Default cell density for the champlevé effect (cells per normalized unit)
pub const DEFAULT_CHAMPLEVE_CELL_DENSITY: f64 = 55.0;

/// Influence of luminance on champlevé interference alignment
pub const DEFAULT_CHAMPLEVE_FLOW_ALIGNMENT: f64 = 0.65;

/// Default interference amplitude for iridescence
pub const DEFAULT_CHAMPLEVE_INTERFERENCE_AMPLITUDE: f64 = 0.6;

/// Default interference frequency for iridescent striations
pub const DEFAULT_CHAMPLEVE_INTERFERENCE_FREQUENCY: f64 = 30.0;

/// Default rim intensity for metal inlay
pub const DEFAULT_CHAMPLEVE_RIM_INTENSITY: f64 = 2.0;

/// Default rim warmth blend factor (0 = original color, 1 = full gold)
pub const DEFAULT_CHAMPLEVE_RIM_WARMTH: f64 = 0.72;

/// Default rim sharpness exponent
pub const DEFAULT_CHAMPLEVE_RIM_SHARPNESS: f64 = 4.5;

/// Default interior lift for opaline glow
pub const DEFAULT_CHAMPLEVE_INTERIOR_LIFT: f64 = 0.70;

/// Default anisotropy strength for brushed-metal sheen
pub const DEFAULT_CHAMPLEVE_ANISOTROPY: f64 = 0.95;

/// Default centre highlight compression for champlevé cells
pub const DEFAULT_CHAMPLEVE_CELL_SOFTNESS: f64 = 1.1;

// ========== Aether Effect Constants ==========

/// Default density of filaments in the aether weave
pub const DEFAULT_AETHER_FILAMENT_DENSITY: f64 = 90.0;

/// Default strength of flow alignment for anisotropic warp
pub const DEFAULT_AETHER_FLOW_ALIGNMENT: f64 = 0.85;

/// Base intensity of the volumetric scattering effect
pub const DEFAULT_AETHER_SCATTERING_STRENGTH: f64 = 1.0;

/// Exponent for the scattering falloff curve
pub const DEFAULT_AETHER_SCATTERING_FALLOFF: f64 = 2.5;

/// Amplitude of the iridescent color shifting
pub const DEFAULT_AETHER_IRIDESCENCE_AMPLITUDE: f64 = 0.65;

/// Frequency of the iridescent color bands
pub const DEFAULT_AETHER_IRIDESCENCE_FREQUENCY: f64 = 12.0;

/// Intensity of the negative space caustics
pub const DEFAULT_AETHER_CAUSTIC_STRENGTH: f64 = 0.35;

/// Softness of the caustic bleed effect
pub const DEFAULT_AETHER_CAUSTIC_SOFTNESS: f64 = 3.0;

// ========== Special Mode Enhancement Constants ==========

/// Production crisp line base thickness in pixels.
pub const CRISP_LINE_BASE_THICKNESS: f32 = 0.95;

/// Minimum production line thickness in pixels.
pub const CRISP_LINE_MIN_THICKNESS: f32 = 0.30;

/// Maximum production line thickness in pixels.
///
/// Raised from 3.20 so that bold seeds (mode-aware `line_weight` up to ~3.1,
/// wildcard ~4.2, times the slow-arc velocity multiplier) and slow, close
/// passages can render genuinely weighty strokes instead of being clamped flat.
pub const CRISP_LINE_MAX_THICKNESS: f32 = 5.50;

/// Offset term of the proximity response in the line width model.
///
/// `width ∝ 1 / (offset + segment_length_px * slope)`: short segments (close
/// encounters, per-step ribbon strokes) draw bold, long spans draw fine.
pub const CRISP_LINE_PROXIMITY_OFFSET: f32 = 0.55;

/// Slope term of the proximity response in the line width model (per pixel).
///
/// Softened from 0.0125: fast passages and drift sweeps produce 40-150 px
/// per-step strokes, and the steeper slope throttled that whole band to
/// hairlines no matter how bold the seed's `line_weight` was. Long ruled
/// chord sheets (300 px+) still resolve gossamer-fine.
pub const CRISP_LINE_PROXIMITY_SLOPE: f32 = 0.008;

/// Z-depth broadening factor for production stills; zero means no depth-of-field blur.
pub const CRISP_DEPTH_BROADENING_FACTOR: f32 = 0.0;

/// Super-Gaussian exponent used for crisp anti-aliased line splats.
pub const CRISP_LINE_FALLOFF_EXPONENT: f32 = 2.0;

/// Minimum averaged per-pixel coverage retained by crisp line splats.
pub const CRISP_LINE_ENERGY_CUTOFF: f32 = 0.004;

/// Subpixel grid dimension used for crisp line coverage integration.
pub const CRISP_LINE_SUBPIXEL_GRID: usize = 2;

/// Additional guard rows beyond the maximum scaled crisp footprint.
pub const HIGH_RES_TILE_GUARD_MARGIN_ROWS: usize = 2;

/// Reference short-edge resolution for crisp line thickness tuning.
pub const CRISP_LINE_REFERENCE_MIN_DIM: f32 = 2234.0;

/// Lower bound for resolution-aware crisp line scaling.
pub const CRISP_LINE_RESOLUTION_SCALE_MIN: f32 = 0.55;

/// Upper bound for resolution-aware crisp line scaling.
pub const CRISP_LINE_RESOLUTION_SCALE_MAX: f32 = 32.0;

/// Minimum projected body motion that enables render-time line interpolation.
pub const CRISP_INTERPOLATION_MIN_MOTION_PX: f32 = 1.0;

/// Resolution scale at which normal motion interpolation begins.
pub const CRISP_INTERPOLATION_SCALE_START: f32 = 1.5;

/// Default-resolution motion that is large enough to justify interpolation.
pub const CRISP_INTERPOLATION_EXTREME_MOTION_PX: f32 = 12.0;

/// Target maximum pixel travel per interpolated high-resolution sample.
pub const CRISP_INTERPOLATION_TARGET_STEP_PX: f32 = 2.5;

/// Maximum number of render-time interpolation samples per simulation interval.
pub const CRISP_INTERPOLATION_MAX_SUBSTEPS: usize = 8;

/// Compute the resolution-aware scale for crisp spectral line widths.
#[must_use]
#[inline]
pub fn crisp_line_resolution_scale(width: u32, height: u32) -> f32 {
    if width == 0 || height == 0 {
        return CRISP_LINE_RESOLUTION_SCALE_MIN;
    }

    let min_dim = width.min(height) as f32;
    (min_dim / CRISP_LINE_REFERENCE_MIN_DIM)
        .clamp(CRISP_LINE_RESOLUTION_SCALE_MIN, CRISP_LINE_RESOLUTION_SCALE_MAX)
}

/// Compute adaptive render-time samples for high-resolution trajectory intervals.
#[must_use]
#[inline]
pub fn crisp_line_interpolation_substeps(width: u32, height: u32, max_motion_px: f32) -> usize {
    let scale = crisp_line_resolution_scale(width, height);
    if max_motion_px <= CRISP_INTERPOLATION_MIN_MOTION_PX {
        return 1;
    }

    let high_resolution = scale >= CRISP_INTERPOLATION_SCALE_START;
    let extreme_motion = max_motion_px >= CRISP_INTERPOLATION_EXTREME_MOTION_PX;
    if !high_resolution && !extreme_motion {
        return 1;
    }

    let scale_factor = if high_resolution { scale.sqrt().clamp(1.0, 2.0) } else { 1.0 };
    ((max_motion_px / (CRISP_INTERPOLATION_TARGET_STEP_PX / scale_factor)).ceil() as usize)
        .clamp(1, CRISP_INTERPOLATION_MAX_SUBSTEPS)
}

/// Minimum guard rows needed for tiled rendering at the given resolution.
#[must_use]
#[inline]
pub fn crisp_tiled_guard_rows(width: u32, height: u32) -> usize {
    let scale = crisp_line_resolution_scale(width, height);
    // Matches the splat bounding-box pad (1.5x thickness + 1 px margin).
    let max_footprint = (CRISP_LINE_MAX_THICKNESS * scale * 1.5).ceil() as usize + 1;
    HIGH_RES_TILE_GUARD_ROWS.max(max_footprint + HIGH_RES_TILE_GUARD_MARGIN_ROWS)
}

/// Minimum spectral lobe width, in SPD bins, for crisp color deposits.
pub const CRISP_SPECTRAL_SIGMA_MIN_BINS: f64 = 0.45;

/// Maximum spectral lobe width, in SPD bins, for crisp color deposits.
pub const CRISP_SPECTRAL_SIGMA_MAX_BINS: f64 = 1.35;

/// Maximum bin radius included when depositing crisp spectral lobes.
pub const CRISP_SPECTRAL_KERNEL_RADIUS_BINS: isize = 3;

/// Production still spectral dispersion strength. Zero disables radial chromatic smear.
pub const CRISP_DISPERSION_STRENGTH: f64 = 0.0;

/// Pixel-count threshold above which final still rendering switches to row stripes.
pub const HIGH_RES_TILED_PIXEL_THRESHOLD: usize = 32_000_000;

/// Row count per stripe in the high-resolution still renderer.
pub const HIGH_RES_TILE_ROWS: usize = 384;

/// Guard rows above and below each high-resolution stripe.
pub const HIGH_RES_TILE_GUARD_ROWS: usize = 4;

/// Spectral dispersion strength - controls prismatic trail separation in non-crisp modes.
pub const SPECTRAL_DISPERSION_STRENGTH: f64 = CRISP_DISPERSION_STRENGTH;
/// Boosted dispersion for wider rainbow trails
pub const SPECTRAL_DISPERSION_STRENGTH_BOOSTED: f64 = CRISP_DISPERSION_STRENGTH;

/// Velocity-based HDR boost factor - multiplies HDR scale at high velocities
/// 1.0 = no boost, 2.0 = double brightness at max velocity
pub const VELOCITY_HDR_BOOST_FACTOR: f64 = 8.0; // Increased from 2.5 for dramatic flares

/// Quantile of the orbit's own speed distribution mapped to "slow" (norm 0).
pub const VELOCITY_NORM_LOW_QUANTILE: f64 = 0.15;

/// Quantile of the orbit's own speed distribution mapped to "fast" (norm 1).
pub const VELOCITY_NORM_HIGH_QUANTILE: f64 = 0.97;

/// Exponent shaping the flare response curve; > 1 reserves the brightest
/// flares for genuinely fast passages instead of the orbit's median speed.
pub const VELOCITY_FLARE_GAMMA: f64 = 1.35;

/// Line thickness multiplier for the slowest arcs (bold, contemplative strokes).
pub const VELOCITY_THICKNESS_SLOW: f64 = 1.55;

/// Line thickness multiplier for the fastest whips (fine but never hairline flares).
pub const VELOCITY_THICKNESS_FAST: f64 = 0.72;

/// Alpha multiplier for the faint ribbon underlay beneath time-lagged chords.
pub const CHORD_RIBBON_UNDERLAY_ALPHA: f64 = 0.30;

/// Per-echo decay multipliers for trailing ribbon echo bands
/// (applied on top of the seed's `ribbon_echo_alpha` at lags 1x, 2x, 3x).
pub const COMET_ECHO_DECAY: [f64; 3] = [1.0, 0.55, 0.30];

// ========== Layered Vocabulary Constants ==========

/// Veil fill lines are drawn every Nth step (energy-compensated) to keep the
/// swept-gauze vocabulary within the accumulation budget.
pub const VEIL_STEP_STRIDE: usize = 2;

/// Weave chords are drawn every Nth step (energy-compensated).
pub const WEAVE_STEP_STRIDE: usize = 2;

/// Straight segments per tessellated harmonic-weave Bezier chord.
pub const WEAVE_SEGMENTS: usize = 6;

/// Depth of the slow timeline modulation applied to the weave bow factor.
pub const WEAVE_BOW_WOBBLE: f64 = 0.45;

/// Minimum simulation steps between stipple dots (guards tiny test scenes).
pub const STIPPLE_MIN_PITCH_STEPS: usize = 24;

/// Stipple energy relative to the line ink the skipped steps would deposit.
pub const STIPPLE_ENERGY_FACTOR: f64 = 0.55;

/// Thickness multiplier for ordinary stipple dots.
pub const STIPPLE_DOT_THICKNESS: f64 = 1.6;

/// Thickness multiplier for the periodic bright pearls.
pub const STIPPLE_PEARL_THICKNESS: f64 = 2.6;

/// Energy multiplier for the periodic bright pearls.
pub const STIPPLE_PEARL_ENERGY: f64 = 2.0;

/// Pixels of tangent length per pixel of per-step screen motion.
pub const TANGENT_VELOCITY_GAIN: f64 = 14.0;

/// Minimum tangent half-length as a fraction of the output short edge.
pub const TANGENT_MIN_LEN_FRAC: f64 = 0.004;

/// Maximum tangent half-length as a fraction of the output short edge.
pub const TANGENT_MAX_LEN_FRAC: f64 = 0.045;

/// Reference tangent length (fraction of short edge) for energy normalization.
pub const TANGENT_REFERENCE_LEN_FRAC: f64 = 0.012;

/// Stardust dot energy relative to the scene's mean per-body trail budget.
pub const STARDUST_ENERGY_FACTOR: f64 = 0.0012;

// ========== Diffraction Spike Constants ==========

/// Maximum bright sources marched per spike pass (brightest kept).
pub const SPIKE_MAX_SOURCES: usize = 20_000;

/// Luminance percentile used as the reference brightness for spike thresholds.
/// Near-maximum so only genuine cores spike, robust to single hot outliers.
pub const SPIKE_LUMINANCE_PERCENTILE: f64 = 0.9999;

/// Exponential decay constants per arm: energy falls to ~5% at full length.
pub const SPIKE_DECAY_AT_TIP: f64 = 3.0;

/// Floor of the lightness-to-energy response (keeps dark bodies visible).
pub const LIGHTNESS_ENERGY_FLOOR: f64 = 0.30;

/// Span of the lightness-to-energy response above the floor.
pub const LIGHTNESS_ENERGY_SPAN: f64 = 1.10;

/// Exponent of the lightness-to-energy response.
pub const LIGHTNESS_ENERGY_GAMMA: f64 = 1.6;

// ========== Video Encoding Constants ==========

/// Default video framerate
pub const DEFAULT_VIDEO_FPS: u32 = 60;

/// Default target duration in frames (~30 seconds at 60 FPS)
pub const DEFAULT_TARGET_FRAMES: u32 = 1800;

/// Histogram sampling budget for still/image exposure analysis.
///
/// This is intentionally much lower than `DEFAULT_TARGET_FRAMES` because pass 1
/// only needs representative luminance coverage, not full video temporal density.
pub const DEFAULT_HISTOGRAM_SAMPLE_FRAMES: u32 = 240;

// ========== Spectral Output Constants ==========

/// Duration of the spectral sweep video in seconds.
pub const CYCLE_DURATION_SECONDS: f64 = 10.0;

/// Total frames in the spectral sweep video (10s * 60fps).
pub const CYCLE_TOTAL_FRAMES: u32 = 600;

/// Display gamma used for spectral gallery and bin image output
pub const DISPLAY_GAMMA: f64 = 2.2;

/// First bin included in the spectral sweep (skips the dimmest violet bins).
pub const SWEEP_BIN_START: usize = 4;

/// Last bin included in the spectral sweep (skips the dimmest red bins).
pub const SWEEP_BIN_END: usize = 59;

/// Gaussian kernel sigma (in bin-units) for multi-bin blending during the sweep.
pub const SWEEP_GAUSSIAN_SIGMA: f64 = 0.42;

/// Gaussian bloom blur radius (pixels) applied to each sweep frame.
pub const SWEEP_BLOOM_RADIUS: usize = 0;

/// Gaussian bloom strength multiplier for sweep frames.
pub const SWEEP_BLOOM_STRENGTH: f64 = 0.0;

/// Gaussian bloom core brightness multiplier for sweep frames.
pub const SWEEP_BLOOM_CORE_BRIGHTNESS: f64 = 1.0;

/// Vignette strength applied during sweep color grading.
pub const SWEEP_VIGNETTE_STRENGTH: f64 = 0.35;

/// Vignette softness exponent for sweep color grading.
pub const SWEEP_VIGNETTE_SOFTNESS: f64 = 2.6;

/// Vibrance boost factor for sweep color grading.
pub const SWEEP_VIBRANCE: f64 = 1.08;

/// Faint full-spectrum structure blended beneath every sweep frame.
pub const SWEEP_AMBIENT_COMPOSITE_STRENGTH: f64 = 0.32;

/// Soft blurred full-spectrum halo blended around the sweep structure.
pub const SWEEP_AMBIENT_HALO_STRENGTH: f64 = 1.65;

/// Blur radius, in pixels at reference sweep size, for the full-spectrum halo.
pub const SWEEP_AMBIENT_HALO_RADIUS_PX: f64 = 14.0;

/// Minimum spectral atmosphere added before grading so frames never collapse to black.
pub const SWEEP_BACKGROUND_LUMINANCE_FLOOR: f64 = 0.018;

/// Strength of the precomputed spectral atmosphere mask around the composition.
pub const SWEEP_BACKGROUND_AURA_STRENGTH: f64 = 0.095;

/// Primary spectral afterglow mixed behind the moving wavelength center.
pub const SWEEP_AFTERGLOW_STRENGTH: f64 = 0.55;

/// Secondary, longer afterglow mixed behind the moving wavelength center.
pub const SWEEP_AFTERGLOW_SECONDARY_STRENGTH: f64 = 0.24;

/// Distance, in spectral bins, between the sweep center and afterglow echoes.
pub const SWEEP_AFTERGLOW_OFFSET_BINS: f64 = 2.15;

/// Small per-wavelength image displacement, in pixels at reference sweep size.
pub const SWEEP_PRISM_DISPLACEMENT_PX: f64 = 4.8;

/// Absolute minimum Rec.709 luminance enforced before sweep color grading.
pub const SWEEP_MIN_FRAME_LUMINANCE: f64 = 0.022;

/// Maximum deterministic wavelength-center vibrato, in spectral bins.
pub const SWEEP_CENTER_VIBRATO_BINS: f64 = 0.16;

/// Number of vibrato cycles across the full ping-pong sweep.
pub const SWEEP_CENTER_VIBRATO_CYCLES: f64 = 7.0;

/// Strength of extra spectral drama near the ping-pong turnaround.
pub const SWEEP_TURNAROUND_FLARE_STRENGTH: f64 = 0.85;

/// Width of the turnaround flare, as a fraction of the full video.
pub const SWEEP_TURNAROUND_FLARE_WIDTH: f64 = 0.055;

/// Extra radial prism displacement added toward image edges.
pub const SWEEP_RADIAL_PRISM_BURST: f64 = 0.85;

/// Additional prism displacement driven by local spectral luminance.
pub const SWEEP_LUMINANCE_DISPERSION: f64 = 0.55;

// ========== Simulation Constants ==========

/// Default simulation timestep
pub const DEFAULT_DT: f64 = 0.001;

/// Kinetic energy factor (1/2 in KE = 1/2 * m * v²)
pub const KINETIC_ENERGY_FACTOR: f64 = 0.5;

// ========== Mathematical Constants ==========

/// Two times PI (full circle in radians)
pub const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

// ========== Rec. 709 Luma Coefficients ==========

/// Rec. 709 red luminance weight
pub const LUMA_R: f64 = 0.2126;
/// Rec. 709 green luminance weight
pub const LUMA_G: f64 = 0.7152;
/// Rec. 709 blue luminance weight
pub const LUMA_B: f64 = 0.0722;

/// Compute Rec. 709 luminance from straight (un-premultiplied) RGB.
#[inline]
#[must_use]
pub fn rec709_luminance(r: f64, g: f64, b: f64) -> f64 {
    LUMA_R * r + LUMA_G * g + LUMA_B * b
}

// ========== Quantization Constants ==========

/// Maximum value for 16-bit unsigned integer, as f64.
pub const U16_MAX_F64: f64 = 65535.0;

// ========== Progress Reporting Constants ==========

/// Percentage conversion factor
pub const PERCENT_FACTOR: f64 = 100.0;

#[cfg(test)]
mod tests {
    use super::*;

    const RUST_DEFAULT_STACK: usize = 2 * 1024 * 1024;

    #[test]
    fn test_thread_stack_size_at_least_256x_default() {
        const { assert!(THREAD_STACK_SIZE >= 256 * RUST_DEFAULT_STACK) };
    }

    #[test]
    fn test_thread_stack_size_is_power_of_two() {
        const { assert!(THREAD_STACK_SIZE.is_power_of_two()) };
    }

    #[test]
    fn test_thread_stack_size_does_not_exceed_1_gib() {
        const ONE_GIB: usize = 1024 * 1024 * 1024;
        const { assert!(THREAD_STACK_SIZE <= ONE_GIB) };
    }
}
