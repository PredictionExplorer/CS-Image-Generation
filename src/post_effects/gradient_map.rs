//! Gradient mapping for luxury color palettes.
//!
//! This effect remaps the luminance values of the image through carefully
//! crafted gradient palettes to create stunning, professional color treatments.

use super::{PixelBuffer, PostEffect, PostEffectError, validate_buffer_shape};
use crate::oklab::{self, GamutMapMode};
use rayon::prelude::*;

/// Predefined luxury color palettes
#[derive(Clone, Debug)]
pub enum LuxuryPalette {
    /// Rich gold to deep purple gradient
    GoldPurple,
    /// Cosmic teal to vibrant pink
    CosmicTealPink,
    /// Warm amber to cool cyan
    AmberCyan,
    /// Deep indigo to bright gold
    IndigoGold,
    /// Ethereal blue to warm orange
    BlueOrange,

    // === MUSEUM-QUALITY ADDITIONS ===
    /// Venetian Renaissance: Deep crimson, burnt sienna, gold leaf, ultramarine
    VenetianRenaissance,
    /// Japanese Ukiyo-e: Prussian blue, vermillion, gold, ink black
    JapaneseUkiyoe,
    /// Art Nouveau: Jade green, peacock blue, burnished copper, cream
    ArtNouveau,
    /// Lunar Opal: Silver, moonstone blue, pale lavender, pearl white
    LunarOpal,
    /// Fire Opal: Deep ruby, flame orange, citrine yellow, rose gold
    FireOpal,
    /// Deep Ocean: Abyssal blue, bioluminescent teal, phosphorescent green, midnight indigo
    DeepOcean,
    /// Aurora Borealis: Emerald green, electric violet, ice blue, magenta
    AuroraBorealis,
    /// Molten Metal: Dark iron, cherry red heat, yellow-white, platinum
    MoltenMetal,
    /// Ancient Jade: Deep jade, celadon, seafoam, white jade
    AncientJade,
    /// Royal Amethyst: Deep purple, violet, lavender, silver
    RoyalAmethyst,
}

type PaletteStops = [(f64, f64, f64); 5];

// Deep purple -> rich gold.
const GOLD_PURPLE_STOPS: PaletteStops = [
    (0.15, 0.05, 0.25),
    (0.35, 0.15, 0.45),
    (0.55, 0.25, 0.50),
    (0.75, 0.45, 0.35),
    (0.95, 0.75, 0.35),
];

// Deep teal -> vibrant pink.
const COSMIC_TEAL_PINK_STOPS: PaletteStops = [
    (0.05, 0.20, 0.25),
    (0.10, 0.35, 0.45),
    (0.35, 0.50, 0.65),
    (0.75, 0.35, 0.60),
    (0.95, 0.55, 0.75),
];

// Warm amber -> cool cyan.
const AMBER_CYAN_STOPS: PaletteStops = [
    (0.25, 0.10, 0.05),
    (0.65, 0.35, 0.15),
    (0.85, 0.75, 0.45),
    (0.45, 0.75, 0.75),
    (0.15, 0.55, 0.70),
];

// Deep indigo -> bright gold.
const INDIGO_GOLD_STOPS: PaletteStops = [
    (0.05, 0.05, 0.20),
    (0.15, 0.20, 0.50),
    (0.35, 0.45, 0.65),
    (0.75, 0.60, 0.40),
    (1.00, 0.85, 0.40),
];

// Ethereal blue -> warm orange.
const BLUE_ORANGE_STOPS: PaletteStops = [
    (0.10, 0.15, 0.35),
    (0.20, 0.35, 0.65),
    (0.50, 0.55, 0.75),
    (0.85, 0.55, 0.35),
    (0.95, 0.65, 0.25),
];

// Inspired by Titian and Tintoretto: rich, warm, luxurious.
const VENETIAN_RENAISSANCE_STOPS: PaletteStops = [
    (0.12, 0.05, 0.08),
    (0.45, 0.12, 0.15),
    (0.55, 0.28, 0.18),
    (0.85, 0.65, 0.25),
    (0.25, 0.35, 0.65),
];

// Inspired by Hokusai and Hiroshige: bold, graphic, elegant.
const JAPANESE_UKIYOE_STOPS: PaletteStops = [
    (0.05, 0.08, 0.15),
    (0.08, 0.15, 0.38),
    (0.12, 0.35, 0.55),
    (0.75, 0.20, 0.18),
    (0.95, 0.75, 0.35),
];

// Inspired by Mucha and Klimt: organic, flowing, metallic.
const ART_NOUVEAU_STOPS: PaletteStops = [
    (0.15, 0.28, 0.22),
    (0.25, 0.45, 0.35),
    (0.18, 0.42, 0.52),
    (0.58, 0.35, 0.22),
    (0.92, 0.88, 0.75),
];

// Cool, ethereal, mystical, with opalescent shimmer.
const LUNAR_OPAL_STOPS: PaletteStops = [
    (0.25, 0.28, 0.35),
    (0.45, 0.52, 0.62),
    (0.65, 0.62, 0.72),
    (0.85, 0.88, 0.92),
    (0.95, 0.95, 0.98),
];

// Warm, intense, gem-like, with internal fire.
const FIRE_OPAL_STOPS: PaletteStops = [
    (0.22, 0.05, 0.08),
    (0.65, 0.08, 0.12),
    (0.88, 0.35, 0.15),
    (0.95, 0.75, 0.25),
    (0.92, 0.72, 0.52),
];

// Mysterious, bioluminescent, deep-water aesthetics.
const DEEP_OCEAN_STOPS: PaletteStops = [
    (0.02, 0.05, 0.15),
    (0.05, 0.12, 0.28),
    (0.08, 0.35, 0.45),
    (0.12, 0.55, 0.52),
    (0.25, 0.72, 0.45),
];

// Electric, dancing celestial phenomenon.
const AURORA_BOREALIS_STOPS: PaletteStops = [
    (0.08, 0.15, 0.25),
    (0.15, 0.52, 0.38),
    (0.22, 0.65, 0.72),
    (0.65, 0.28, 0.75),
    (0.85, 0.35, 0.72),
];

// Industrial, powerful forge aesthetics.
const MOLTEN_METAL_STOPS: PaletteStops = [
    (0.08, 0.08, 0.10),
    (0.25, 0.12, 0.10),
    (0.72, 0.18, 0.12),
    (0.95, 0.82, 0.35),
    (0.88, 0.90, 0.92),
];

// Serene, precious, Chinese imperial aesthetics.
const ANCIENT_JADE_STOPS: PaletteStops = [
    (0.08, 0.18, 0.15),
    (0.25, 0.42, 0.35),
    (0.45, 0.62, 0.52),
    (0.68, 0.82, 0.75),
    (0.88, 0.95, 0.92),
];

// Regal, mystical, crystalline gem.
const ROYAL_AMETHYST_STOPS: PaletteStops = [
    (0.15, 0.08, 0.22),
    (0.35, 0.15, 0.48),
    (0.55, 0.28, 0.65),
    (0.75, 0.52, 0.82),
    (0.85, 0.82, 0.88),
];

impl LuxuryPalette {
    /// Convert an integer index (0-14) to a palette variant.
    /// Useful for randomized palette selection.
    #[must_use]
    pub fn from_index(index: usize) -> Self {
        match index % 15 {
            // Modulo ensures we always get a valid palette
            0 => LuxuryPalette::GoldPurple,
            1 => LuxuryPalette::CosmicTealPink,
            2 => LuxuryPalette::AmberCyan,
            3 => LuxuryPalette::IndigoGold,
            4 => LuxuryPalette::BlueOrange,
            5 => LuxuryPalette::VenetianRenaissance,
            6 => LuxuryPalette::JapaneseUkiyoe,
            7 => LuxuryPalette::ArtNouveau,
            8 => LuxuryPalette::LunarOpal,
            9 => LuxuryPalette::FireOpal,
            10 => LuxuryPalette::DeepOcean,
            11 => LuxuryPalette::AuroraBorealis,
            12 => LuxuryPalette::MoltenMetal,
            13 => LuxuryPalette::AncientJade,
            14 => LuxuryPalette::RoyalAmethyst,
            _ => unreachable!("Modulo 15 ensures index is 0-14"),
        }
    }

    fn stops(&self) -> &'static PaletteStops {
        match self {
            LuxuryPalette::GoldPurple => &GOLD_PURPLE_STOPS,
            LuxuryPalette::CosmicTealPink => &COSMIC_TEAL_PINK_STOPS,
            LuxuryPalette::AmberCyan => &AMBER_CYAN_STOPS,
            LuxuryPalette::IndigoGold => &INDIGO_GOLD_STOPS,
            LuxuryPalette::BlueOrange => &BLUE_ORANGE_STOPS,
            LuxuryPalette::VenetianRenaissance => &VENETIAN_RENAISSANCE_STOPS,
            LuxuryPalette::JapaneseUkiyoe => &JAPANESE_UKIYOE_STOPS,
            LuxuryPalette::ArtNouveau => &ART_NOUVEAU_STOPS,
            LuxuryPalette::LunarOpal => &LUNAR_OPAL_STOPS,
            LuxuryPalette::FireOpal => &FIRE_OPAL_STOPS,
            LuxuryPalette::DeepOcean => &DEEP_OCEAN_STOPS,
            LuxuryPalette::AuroraBorealis => &AURORA_BOREALIS_STOPS,
            LuxuryPalette::MoltenMetal => &MOLTEN_METAL_STOPS,
            LuxuryPalette::AncientJade => &ANCIENT_JADE_STOPS,
            LuxuryPalette::RoyalAmethyst => &ROYAL_AMETHYST_STOPS,
        }
    }
}

/// Configuration for gradient mapping effect
#[derive(Clone, Debug)]
pub struct GradientMapConfig {
    /// Selected luxury palette
    pub palette: LuxuryPalette,
    /// Strength of the effect (0.0 = original, 1.0 = full gradient map)
    pub strength: f64,
    /// Preserve original hue to some degree
    pub hue_preservation: f64,
}

impl Default for GradientMapConfig {
    fn default() -> Self {
        Self { palette: LuxuryPalette::GoldPurple, strength: 0.55, hue_preservation: 0.25 }
    }
}

/// Gradient mapping post-effect
pub struct GradientMap {
    config: GradientMapConfig,
    enabled: bool,
}

impl GradientMap {
    /// Creates a new gradient map effect from the given configuration.
    #[must_use]
    pub fn new(config: GradientMapConfig) -> Self {
        Self { config, enabled: true }
    }

    /// Get color from palette at normalized position (0.0 to 1.0)
    fn sample_palette(&self, t: f64) -> (f64, f64, f64) {
        Self::interpolate_gradient(self.config.palette.stops(), t.clamp(0.0, 1.0))
    }

    /// Interpolate between gradient stop colors
    fn interpolate_gradient(colors: &[(f64, f64, f64)], t: f64) -> (f64, f64, f64) {
        let n = colors.len() - 1;
        let segment = (t * n as f64).min(n as f64 - 0.0001);
        let idx = segment.floor() as usize;
        let local_t = segment - idx as f64;

        let c0 = colors[idx];
        let c1 = colors[(idx + 1).min(n)];

        let (l0, c0_chroma, h0) = Self::rgb_to_oklch(c0.0, c0.1, c0.2);
        let (l1, c1_chroma, h1) = Self::rgb_to_oklch(c1.0, c1.1, c1.2);
        let hue = if c0_chroma < 1e-6 {
            h1
        } else if c1_chroma < 1e-6 {
            h0
        } else {
            Self::interpolate_hue_shortest_arc(h0, h1, local_t)
        };
        let lightness = l0 + (l1 - l0) * local_t;
        let chroma = c0_chroma + (c1_chroma - c0_chroma) * local_t;

        Self::oklch_to_rgb(lightness, chroma, hue)
    }

    fn rgb_to_oklch(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
        let (l, a, b_ch) = oklab::linear_srgb_to_oklab(r, g, b);
        let chroma = (a * a + b_ch * b_ch).sqrt();
        let hue = if chroma < 1e-10 { 0.0 } else { b_ch.atan2(a).to_degrees().rem_euclid(360.0) };
        (l, chroma, hue)
    }

    fn oklch_to_rgb(lightness: f64, chroma: f64, hue_deg: f64) -> (f64, f64, f64) {
        let hue_rad = hue_deg.to_radians();
        let a = chroma * hue_rad.cos();
        let b = chroma * hue_rad.sin();
        let (r, g, b_ch) = oklab::oklab_to_linear_srgb(lightness, a, b);
        GamutMapMode::PreserveHue.map_to_gamut(r, g, b_ch)
    }

    fn interpolate_hue_shortest_arc(start: f64, end: f64, t: f64) -> f64 {
        let delta = (end - start + 540.0).rem_euclid(360.0) - 180.0;
        (start + delta * t).rem_euclid(360.0)
    }
}

impl PostEffect for GradientMap {
    fn is_enabled(&self) -> bool {
        self.enabled && self.config.strength > 0.0
    }

    fn process(
        &self,
        input: &PixelBuffer,
        width: usize,
        height: usize,
    ) -> Result<PixelBuffer, PostEffectError> {
        validate_buffer_shape(self.name(), input.len(), width, height)?;
        if !self.is_enabled() {
            return Ok(input.clone());
        }

        let output: PixelBuffer = input
            .par_iter()
            .map(|&(r, g, b, a)| {
                if a <= 0.0 {
                    return (r, g, b, a);
                }

                // Convert to straight alpha
                let sr = r / a;
                let sg = g / a;
                let sb = b / a;

                // Calculate luminance
                let lum = crate::render::constants::rec709_luminance(sr, sg, sb);

                // Sample gradient at luminance
                let (gr, gg, gb) = self.sample_palette(lum);

                // Apply hue preservation if requested
                let (final_r, final_g, final_b) = if self.config.hue_preservation > 0.0 {
                    let (orig_l, orig_c, orig_h) = Self::rgb_to_oklch(sr, sg, sb);
                    let (grad_l, grad_c, grad_h) = Self::rgb_to_oklch(gr, gg, gb);
                    let preserve = self.config.hue_preservation.clamp(0.0, 1.0);
                    let blended_h = if orig_c < 1e-6 {
                        grad_h
                    } else if grad_c < 1e-6 {
                        orig_h
                    } else {
                        Self::interpolate_hue_shortest_arc(grad_h, orig_h, preserve)
                    };
                    let blended_c = grad_c * (1.0 - preserve) + orig_c * preserve;
                    let blended_l = grad_l * (1.0 - preserve * 0.35) + orig_l * preserve * 0.35;

                    Self::oklch_to_rgb(blended_l, blended_c, blended_h)
                } else {
                    (gr, gg, gb)
                };

                // Blend with original based on strength
                let strength = self.config.strength;
                let blended_r = sr * (1.0 - strength) + final_r * strength;
                let blended_g = sg * (1.0 - strength) + final_g * strength;
                let blended_b = sb * (1.0 - strength) + final_b * strength;

                // Convert back to premultiplied alpha
                ((blended_r * a).max(0.0), (blended_g * a).max(0.0), (blended_b * a).max(0.0), a)
            })
            .collect();

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolate_hue_shortest_arc_wraps_cleanly() {
        let hue = GradientMap::interpolate_hue_shortest_arc(350.0, 10.0, 0.5);
        assert!(hue.abs() < 1.0 || (hue - 360.0).abs() < 1.0);
    }

    #[test]
    fn test_palette_sampling_returns_in_gamut_rgb() {
        let gradient = GradientMap::new(GradientMapConfig {
            palette: LuxuryPalette::RoyalAmethyst,
            strength: 1.0,
            hue_preservation: 0.0,
        });
        let (r, g, b) = gradient.sample_palette(0.42);

        assert!((0.0..=1.0).contains(&r));
        assert!((0.0..=1.0).contains(&g));
        assert!((0.0..=1.0).contains(&b));
    }

    #[test]
    fn test_hue_preservation_keeps_neutral_input_stable() {
        let gradient = GradientMap::new(GradientMapConfig {
            palette: LuxuryPalette::AuroraBorealis,
            strength: 1.0,
            hue_preservation: 0.6,
        });
        let input = vec![(0.45, 0.45, 0.45, 1.0)];
        let output = gradient.process(&input, 1, 1).expect("gradient map process should succeed");
        let (r, g, b, _) = output[0];

        assert!((r - g).abs() < 0.25);
        assert!((g - b).abs() < 0.25);
    }

    #[test]
    fn test_gradient_map_rejects_invalid_buffer_shape() {
        let gradient = GradientMap::new(GradientMapConfig::default());
        let buffer: PixelBuffer = vec![(0.5, 0.5, 0.5, 1.0); 99];

        let err = gradient.process(&buffer, 10, 10).expect_err("mismatched buffer should fail");

        assert!(matches!(err, PostEffectError::InvalidBuffer { .. }));
        assert!(err.to_string().contains("GradientMap"));
        assert!(err.to_string().contains("buffer length"));
    }
}
