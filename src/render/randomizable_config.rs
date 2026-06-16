//! Resolved finish configuration for the active `CosmicSignature` renderer.
//!
//! Parameter sampling now lives in `visual_profile`; this module is intentionally
//! limited to the small set of finish controls still consumed by render passes.

/// Fully resolved finish configuration.
#[derive(Clone, Debug)]
pub struct ResolvedEffectConfig {
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,

    /// Whether halation bloom is enabled.
    pub enable_bloom: bool,
    /// Whether the rare prism/chromatic-bloom trait is enabled.
    pub enable_chromatic_bloom: bool,

    /// Gaussian bloom blend strength, used only when [`crate::render::BloomMode::Gaussian`] is
    /// selected by an embedding caller.
    pub blur_strength: f64,
    /// Gaussian bloom radius as a fraction of image size.
    pub blur_radius_scale: f64,
    /// Core brightness retention during bloom.
    pub blur_core_brightness: f64,

    /// Difference-of-Gaussians halation strength.
    pub dog_strength: f64,
    /// `DoG` inner sigma as a fraction of image size.
    pub dog_sigma_scale: f64,
    /// Ratio between outer and inner `DoG` sigma.
    pub dog_ratio: f64,

    /// Prismatic color separation strength.
    pub chromatic_bloom_strength: f64,
    /// Chromatic bloom radius as a fraction of image size.
    pub chromatic_bloom_radius_scale: f64,
    /// Chromatic bloom channel separation as a fraction of image size.
    pub chromatic_bloom_separation_scale: f64,
    /// Luminance threshold for prism extraction.
    pub chromatic_bloom_threshold: f64,

    /// HDR scaling factor.
    pub hdr_scale: f64,
    /// Black point clipping threshold.
    pub clip_black: f64,
    /// White point clipping threshold.
    pub clip_white: f64,
}

impl Default for ResolvedEffectConfig {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            enable_bloom: false,
            enable_chromatic_bloom: false,
            blur_strength: 0.0,
            blur_radius_scale: 0.0,
            blur_core_brightness: 1.0,
            dog_strength: 0.0,
            dog_sigma_scale: 0.0,
            dog_ratio: 1.0,
            chromatic_bloom_strength: 0.0,
            chromatic_bloom_radius_scale: 0.0,
            chromatic_bloom_separation_scale: 0.0,
            chromatic_bloom_threshold: 1.0,
            hdr_scale: 1.0,
            clip_black: 0.0,
            clip_white: 1.0,
        }
    }
}

impl ResolvedEffectConfig {
    /// Return true when any finish pass is active.
    #[must_use]
    pub fn any_legacy_effect_enabled(&self) -> bool {
        self.enable_bloom || self.enable_chromatic_bloom
    }

    /// Return true when any effect other than halation is active.
    #[must_use]
    pub fn any_effect_beyond_halation_enabled(&self) -> bool {
        self.enable_chromatic_bloom
    }

    /// Return true when any effect outside the curated `CosmicSignature` trait set is active.
    #[must_use]
    pub fn any_effect_beyond_signature_traits_enabled(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_finish_detection_tracks_halation_and_prism() {
        let clean = ResolvedEffectConfig::default();
        assert!(!clean.any_legacy_effect_enabled());
        assert!(!clean.any_effect_beyond_halation_enabled());
        assert!(!clean.any_effect_beyond_signature_traits_enabled());

        let halation =
            ResolvedEffectConfig { enable_bloom: true, ..ResolvedEffectConfig::default() };
        assert!(halation.any_legacy_effect_enabled());
        assert!(!halation.any_effect_beyond_halation_enabled());

        let prism = ResolvedEffectConfig {
            enable_chromatic_bloom: true,
            ..ResolvedEffectConfig::default()
        };
        assert!(prism.any_legacy_effect_enabled());
        assert!(prism.any_effect_beyond_halation_enabled());
        assert!(!prism.any_effect_beyond_signature_traits_enabled());
    }
}
