//! Mood-curated visual identity.
//!
//! Every render picks one of three moods (Cinematic, Cosmic, Painterly) which
//! biases post-effect enable probabilities and strength parameters toward a
//! coherent look. This produces renders with a recognizable identity rather
//! than a random mix of all effects.
//!
//! The chosen mood is sampled deterministically from the seed RNG (so the same
//! seed always produces the same mood) unless the CLI forces a specific mood.

use crate::sim::Sha3RandomByteStream;

/// High-level visual mood.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mood {
    /// Film / photographic: multi-tier bloom, anamorphic lens flare, god rays,
    /// rim-lit bodies, moody color grade.
    Cinematic,
    /// Astrophotography: procedural star field, diffraction spikes on bright
    /// points, airy-disc body cores, richly-coloured nebula background.
    Cosmic,
    /// Ethereal / illuminated-manuscript: harmonized `OKLab` palette, boosted
    /// opalescence + aether, glazed highlights, canvas texture.
    Painterly,
}

impl Mood {
    /// Short textual identifier for logging / JSON output.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cinematic => "cinematic",
            Self::Cosmic => "cosmic",
            Self::Painterly => "painterly",
        }
    }

    /// Parse a mood from a CLI string. Returns `None` if unknown / blank /
    /// `"auto"` (callers should sample in that case).
    #[must_use]
    pub fn from_cli(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "cinematic" | "cine" | "film" => Some(Self::Cinematic),
            "cosmic" | "astro" | "space" => Some(Self::Cosmic),
            "painterly" | "paint" | "ethereal" => Some(Self::Painterly),
            _ => None,
        }
    }

    /// Sample one mood uniformly from the seed RNG.
    #[must_use]
    pub fn sample(rng: &mut Sha3RandomByteStream) -> Self {
        let r = rng.next_f64();
        if r < 1.0 / 3.0 {
            Self::Cinematic
        } else if r < 2.0 / 3.0 {
            Self::Cosmic
        } else {
            Self::Painterly
        }
    }

    /// Per-mood effect biases. Multipliers on base enable probabilities;
    /// `1.0` means no change, `>1.0` promotes the effect, `<1.0` suppresses it.
    #[must_use]
    pub fn biases(self) -> MoodBiases {
        match self {
            Self::Cinematic => MoodBiases {
                bloom: 2.5,
                glow: 1.4,
                chromatic_bloom: 2.0,
                perceptual_blur: 1.2,
                micro_contrast: 1.0,
                gradient_map: 0.6,
                color_grade: 1.4,
                champleve: 0.4,
                aether: 0.5,
                opalescence: 0.6,
                edge_luminance: 1.1,
                atmospheric_depth: 1.5,
                fine_texture: 0.7,
                // New effects
                bloom_pyramid: 1.0,
                anamorphic_flare: 1.0,
                god_rays: 0.7,
                rim_light: 1.0,
                star_field: 0.2,
                diffraction_spikes: 0.8,
                airy_disc: 0.2,
                nebula_tint: 0.4,
                palette_harmony: 0.4,
                glaze: 0.4,
                bloom_strength_scale: 1.35,
                color_saturation_scale: 1.05,
                vignette_scale: 1.25,
            },
            Self::Cosmic => MoodBiases {
                bloom: 1.6,
                glow: 1.9,
                chromatic_bloom: 1.3,
                perceptual_blur: 0.8,
                micro_contrast: 1.1,
                gradient_map: 0.8,
                color_grade: 1.1,
                champleve: 0.3,
                aether: 0.5,
                opalescence: 0.7,
                edge_luminance: 1.2,
                atmospheric_depth: 0.8,
                fine_texture: 0.4,
                // New effects
                bloom_pyramid: 0.6,
                anamorphic_flare: 0.4,
                god_rays: 0.4,
                rim_light: 0.7,
                star_field: 1.0,
                diffraction_spikes: 1.0,
                airy_disc: 1.0,
                nebula_tint: 1.0,
                palette_harmony: 0.5,
                glaze: 0.3,
                bloom_strength_scale: 1.1,
                color_saturation_scale: 1.15,
                vignette_scale: 1.1,
            },
            Self::Painterly => MoodBiases {
                bloom: 1.1,
                glow: 1.2,
                chromatic_bloom: 0.9,
                perceptual_blur: 1.3,
                micro_contrast: 1.0,
                gradient_map: 2.2,
                color_grade: 1.5,
                champleve: 1.8,
                aether: 2.3,
                opalescence: 2.8,
                edge_luminance: 0.9,
                atmospheric_depth: 1.0,
                fine_texture: 2.0,
                // New effects
                bloom_pyramid: 0.5,
                anamorphic_flare: 0.2,
                god_rays: 0.2,
                rim_light: 0.4,
                star_field: 0.1,
                diffraction_spikes: 0.2,
                airy_disc: 0.2,
                nebula_tint: 0.5,
                palette_harmony: 1.0,
                glaze: 1.0,
                bloom_strength_scale: 0.95,
                color_saturation_scale: 1.1,
                vignette_scale: 1.0,
            },
        }
    }
}

/// Per-effect multipliers and strength scales for a given mood.
///
/// Each probability multiplier is clamped by the resolver into `[0, 1]` after
/// being applied to the base probability in `parameter_descriptors.rs`.
#[derive(Clone, Copy, Debug)]
pub struct MoodBiases {
    /// Multiplier on base bloom enable probability.
    pub bloom: f64,
    /// Multiplier on base glow enhancement enable probability.
    pub glow: f64,
    /// Multiplier on base chromatic bloom enable probability.
    pub chromatic_bloom: f64,
    /// Multiplier on base perceptual blur enable probability.
    pub perceptual_blur: f64,
    /// Multiplier on base micro-contrast enable probability.
    pub micro_contrast: f64,
    /// Multiplier on base gradient map enable probability.
    pub gradient_map: f64,
    /// Multiplier on base cinematic color grade enable probability.
    pub color_grade: f64,
    /// Multiplier on base champleve enable probability.
    pub champleve: f64,
    /// Multiplier on base aether enable probability.
    pub aether: f64,
    /// Multiplier on base opalescence enable probability.
    pub opalescence: f64,
    /// Multiplier on base edge-luminance enable probability.
    pub edge_luminance: f64,
    /// Multiplier on base atmospheric-depth enable probability.
    pub atmospheric_depth: f64,
    /// Multiplier on base fine-texture enable probability.
    pub fine_texture: f64,
    /// Probability (0..1) that the multi-tier bloom pyramid runs.
    pub bloom_pyramid: f64,
    /// Probability (0..1) that the wide anamorphic lens flare runs.
    pub anamorphic_flare: f64,
    /// Probability (0..1) that god rays from hot highlights run.
    pub god_rays: f64,
    /// Probability (0..1) that bodies are rim-lit with doppler-tinted rings.
    pub rim_light: f64,
    /// Probability (0..1) that a procedural star field is composited.
    pub star_field: f64,
    /// Probability (0..1) that diffraction spikes run on bright pixels.
    pub diffraction_spikes: f64,
    /// Probability (0..1) that the airy-disc PSF replaces Gaussian body cores.
    pub airy_disc: f64,
    /// Probability (0..1) that the nebula background is tinted by the scene SPD.
    pub nebula_tint: f64,
    /// Probability (0..1) that the `OKLab` palette harmonizer is applied.
    pub palette_harmony: f64,
    /// Probability (0..1) that the warm-tinted glaze highlight pass runs.
    pub glaze: f64,
    /// Global strength multiplier applied to bloom / bloom-pyramid strength.
    pub bloom_strength_scale: f64,
    /// Global saturation multiplier applied to color grade.
    pub color_saturation_scale: f64,
    /// Global vignette intensity multiplier applied to color grade.
    pub vignette_scale: f64,
}

impl Default for MoodBiases {
    fn default() -> Self {
        Self {
            bloom: 1.0,
            glow: 1.0,
            chromatic_bloom: 1.0,
            perceptual_blur: 1.0,
            micro_contrast: 1.0,
            gradient_map: 1.0,
            color_grade: 1.0,
            champleve: 1.0,
            aether: 1.0,
            opalescence: 1.0,
            edge_luminance: 1.0,
            atmospheric_depth: 1.0,
            fine_texture: 1.0,
            bloom_pyramid: 0.0,
            anamorphic_flare: 0.0,
            god_rays: 0.0,
            rim_light: 0.0,
            star_field: 0.0,
            diffraction_spikes: 0.0,
            airy_disc: 0.0,
            nebula_tint: 0.0,
            palette_harmony: 0.0,
            glaze: 0.0,
            bloom_strength_scale: 1.0,
            color_saturation_scale: 1.0,
            vignette_scale: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mood_sample_is_deterministic() {
        let mut rng_a = Sha3RandomByteStream::new(&[0xAB, 0xCD], 100.0, 300.0, 300.0, 1.0);
        let mut rng_b = Sha3RandomByteStream::new(&[0xAB, 0xCD], 100.0, 300.0, 300.0, 1.0);
        assert_eq!(Mood::sample(&mut rng_a), Mood::sample(&mut rng_b));
    }

    #[test]
    fn mood_sample_covers_all_three() {
        let mut seen = [false; 3];
        for seed_byte in 0u8..=255 {
            let mut rng =
                Sha3RandomByteStream::new(&[seed_byte, 0x01, 0x02, 0x03], 100.0, 300.0, 300.0, 1.0);
            match Mood::sample(&mut rng) {
                Mood::Cinematic => seen[0] = true,
                Mood::Cosmic => seen[1] = true,
                Mood::Painterly => seen[2] = true,
            }
        }
        assert!(seen[0] && seen[1] && seen[2], "all three moods should appear over 256 seeds");
    }

    #[test]
    fn mood_from_cli_round_trip() {
        assert_eq!(Mood::from_cli("cinematic"), Some(Mood::Cinematic));
        assert_eq!(Mood::from_cli("Cosmic"), Some(Mood::Cosmic));
        assert_eq!(Mood::from_cli("painterly"), Some(Mood::Painterly));
        assert_eq!(Mood::from_cli("auto"), None);
        assert_eq!(Mood::from_cli(""), None);
    }

    #[test]
    fn biases_have_non_negative_values() {
        for mood in [Mood::Cinematic, Mood::Cosmic, Mood::Painterly] {
            let b = mood.biases();
            assert!(b.bloom >= 0.0 && b.glow >= 0.0 && b.fine_texture >= 0.0);
            assert!(b.bloom_pyramid >= 0.0 && b.star_field >= 0.0 && b.glaze >= 0.0);
            assert!(b.bloom_strength_scale > 0.0);
            assert!(b.color_saturation_scale > 0.0);
            assert!(b.vignette_scale > 0.0);
        }
    }
}
