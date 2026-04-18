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
pub const HUE_DRIFT_SCALE: f64 = 1.85;

/// Base time drift factor for subtle color evolution
pub const BASE_HUE_DRIFT: f64 = 1.4;

/// Amplitude, in degrees, applied by the palette sway wave
pub const HUE_WAVE_AMPLITUDE: f64 = 52.0;

/// Additional per-body phase offsets (degrees) to guarantee separation
pub const BODY_HUE_PHASE: [f64; 3] = [0.0, 120.0, 240.0];

// ========== OKLab Perceptual Color Space Constants ==========

/// Base chroma value (typical range 0-0.3 for natural colors)
pub const OKLAB_CHROMA_BASE: f64 = 0.18;
/// Boosted base chroma for museum-quality output
pub const OKLAB_CHROMA_BASE_BOOSTED: f64 = 0.22;

/// Range of chroma variation around the base value
pub const OKLAB_CHROMA_RANGE: f64 = 0.12;
/// Boosted chroma range
pub const OKLAB_CHROMA_RANGE_BOOSTED: f64 = 0.14;

/// Additional chroma modulation applied via palette waves
pub const OKLAB_CHROMA_WAVE_AMPLITUDE: f64 = 0.07;
/// Boosted chroma wave amplitude
pub const OKLAB_CHROMA_WAVE_AMPLITUDE_BOOSTED: f64 = 0.10;

/// Base lightness value (0=black, 1=white)
pub const OKLAB_LIGHTNESS_BASE: f64 = 0.62;

/// Range of lightness variation around the base value
pub const OKLAB_LIGHTNESS_RANGE: f64 = 0.32;

/// Additional lightness modulation applied via palette waves
pub const OKLAB_LIGHTNESS_WAVE_AMPLITUDE: f64 = 0.28;

// ========== Rendering Constants ==========

/// Default HDR scale factor when HDR mode is disabled
pub const DEFAULT_HDR_SCALE: f64 = 1.0;

/// Pre-tonemap luminance target for the solved white percentile.
/// Values above this still retain headroom for specular accents.
///
/// Lowered from `0.88` -> `0.78` so the high-percentile target sits
/// well below the `AgX` saturation point, giving the tonemap a real
/// shoulder to shape instead of having highlights slam into the hard
/// upper bound of `AgX`'s allocation window.
pub const DEFAULT_PRETONEMAP_LUMA_TARGET: f64 = 0.78;

/// Samples above this normalized luminance are considered near-clipped during proxy analysis.
pub const DEFAULT_PRETONEMAP_NEAR_CLIP_THRESHOLD: f64 = 1.10;

/// Maximum tolerated fraction of near-clipped proxy samples before the governor darkens exposure.
pub const DEFAULT_PRETONEMAP_NEAR_CLIP_BUDGET: f64 = 0.0025;

/// Response multiplier for the highlight budget governor.
pub const DEFAULT_PRETONEMAP_BUDGET_RESPONSE: f64 = 1.5;

/// Lower clamp for the global exposure scale derived from proxy analysis.
///
/// Lowered from `0.35` → `0.20` so the governor has more room to darken
/// truly extreme scenes (many additive bloom-family effects stacked at
/// top-of-range strengths). Without this, the governor bottoms out at
/// `0.35` for pathological seeds and the tonemap still gets inputs
/// that collapse to flat display white.
pub const DEFAULT_MIN_EXPOSURE_SCALE: f64 = 0.20;

/// Display-space luminance reserved for "paper white".
///
/// Lowered from `0.92` to `0.85` in the gallery-grade pipeline so the
/// OKLab-space tonemap leaves more headroom for specular accents to read
/// as bright-but-not-blown rather than saturating at display white.
pub const DEFAULT_TONEMAP_PAPER_WHITE: f64 = 0.85;

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

/// Spectral dispersion strength - controls prismatic trail separation
pub const SPECTRAL_DISPERSION_STRENGTH: f64 = 0.8;
/// Boosted dispersion for wider rainbow trails
pub const SPECTRAL_DISPERSION_STRENGTH_BOOSTED: f64 = 1.1;

/// Velocity-based HDR boost factor - multiplies HDR scale at high velocities.
/// `1.0` = no boost, `2.0` = double brightness at max velocity.
///
/// Tuned back to `2.5` from the "dramatic flares" value of `8.0` because
/// the 8× boost pushed the line-splat accumulation so far above the SPD
/// saturation point that hot cores collapsed to pure white through the
/// `OKLab` tonemap. A 2.5× boost still emphasizes fast segments without
/// clipping.
pub const VELOCITY_HDR_BOOST_FACTOR: f64 = 2.5;

/// Velocity threshold for HDR boost (normalized units per timestep)
/// Velocities above this get maximum boost
pub const VELOCITY_HDR_BOOST_THRESHOLD: f64 = 0.15;

/// Energy density threshold for wavelength shift (normalized energy).
///
/// Pixels above this threshold shift toward red (heat). Raised from `0.08`
/// to `0.35` so only genuinely dense regions pick up a redshift tint,
/// instead of mildly bright mid-trail pixels collapsing toward red or white.
pub const ENERGY_DENSITY_SHIFT_THRESHOLD: f64 = 0.35;

/// Wavelength shift strength (fraction of bin to shift per density unit).
///
/// Reduced from `0.75` to `0.25` to preserve hue in hot cores. The `OKLab`
/// tonemap protects chroma through compression, so the aggressive redshift
/// is no longer needed to convey "heat" and was flattening the palette.
pub const ENERGY_DENSITY_SHIFT_STRENGTH: f64 = 0.25;

/// Maximum per-pixel energy a single line splat may deposit in one bin.
///
/// Prevents a single coincidence of thin segment + high velocity boost +
/// grazing-distance Gaussian splat from dumping an order of magnitude
/// more energy than neighbouring pixels, which would force the tonemap to
/// collapse the rest of the trail into a black backdrop.
pub const LINE_SPLAT_ENERGY_CAP: f64 = 0.02;

/// Lateral chromatic dispersion, in pixels, applied per bin during line
/// accumulation. Gives the full trail a gentle wavelength-dependent
/// spread so the core shows hue separation (instead of saturating to
/// white), not only the rim where image-center `CA` becomes visible.
///
/// Kept deliberately small (sub-pixel) so it reads as smooth prismatic
/// shimmer rather than smeared colour blobs.
pub const ACCUMULATION_DISPERSION_PX: f64 = 0.35;

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

/// Duration of the spectral sweep video in seconds
pub const CYCLE_DURATION_SECONDS: f64 = 12.0;

/// Total frames in the spectral sweep video (12s * 60fps)
pub const CYCLE_TOTAL_FRAMES: u32 = 720;

/// Display gamma used for spectral gallery and bin image output
pub const DISPLAY_GAMMA: f64 = 2.2;

/// First bin included in the spectral sweep (skips the dimmest violet bins).
pub const SWEEP_BIN_START: usize = 4;

/// Last bin included in the spectral sweep (skips the dimmest red bins).
pub const SWEEP_BIN_END: usize = 59;

/// Gaussian kernel sigma (in bin-units) for multi-bin blending during the sweep.
pub const SWEEP_GAUSSIAN_SIGMA: f64 = 2.5;

/// Gaussian bloom blur radius (pixels) applied to each sweep frame.
pub const SWEEP_BLOOM_RADIUS: usize = 12;

/// Gaussian bloom strength multiplier for sweep frames.
pub const SWEEP_BLOOM_STRENGTH: f64 = 0.3;

/// Gaussian bloom core brightness multiplier for sweep frames.
pub const SWEEP_BLOOM_CORE_BRIGHTNESS: f64 = 1.0;

/// Vignette strength applied during sweep color grading.
pub const SWEEP_VIGNETTE_STRENGTH: f64 = 0.35;

/// Vignette softness exponent for sweep color grading.
pub const SWEEP_VIGNETTE_SOFTNESS: f64 = 2.6;

/// Vibrance boost factor for sweep color grading.
pub const SWEEP_VIBRANCE: f64 = 1.08;

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
