//! Proptest-based invariants for foundational components.
//!
//! These property tests do not attempt to exhaustively cover the pipeline; they
//! encode invariants that must hold for every randomized configuration, every
//! style bundle, and every post-effect instance that the resolver can produce.
//!
//! Each invariant is rendered in terms of *public* crate API only, so these
//! tests will catch regressions visible to library consumers as well as to the
//! internal rendering pipeline.

use proptest::prelude::*;
use three_body_problem::post_effects::{
    AtmosphericDepth, AtmosphericDepthConfig, ChromaticBloom, ChromaticBloomConfig,
    CinematicColorGrade, ColorGradeParams, EdgeLuminance, EdgeLuminanceConfig, FineTexture,
    FineTextureConfig, GlowEnhancement, GlowEnhancementConfig, GradientMap, GradientMapConfig,
    LensFlare, LensFlareConfig, MicroContrast, MicroContrastConfig, Opalescence, OpalescenceConfig,
    PerceptualBlur, PerceptualBlurConfig, PixelBuffer, PostEffect, Starfield, StarfieldConfig,
};
use three_body_problem::render::art_style::{ArtStyle, DriftCharacter};
use three_body_problem::render::context::RenderContext;
use three_body_problem::render::grade_presets::GradePreset;
use three_body_problem::render::hue_palette::{HuePaletteMode, hues_for_mode};
use three_body_problem::render::randomizable_config::RandomizableEffectConfig;
use three_body_problem::sim::Sha3RandomByteStream;

use nalgebra::Vector3;

const MIN_MASS: f64 = 100.0;
const MAX_MASS: f64 = 300.0;
const LOCATION: f64 = 300.0;
const VELOCITY: f64 = 1.0;

fn make_rng_from_u64(seed: u64) -> Sha3RandomByteStream {
    let bytes = seed.to_le_bytes();
    Sha3RandomByteStream::new(&bytes, MIN_MASS, MAX_MASS, LOCATION, VELOCITY)
}

fn sample_buffer(width: usize, height: usize) -> PixelBuffer {
    (0..(width * height))
        .map(|i| {
            let t = (i as f64) / ((width * height) as f64).max(1.0);
            (
                (0.30 + 0.6 * t).min(1.0),
                (0.20 + 0.5 * (1.0 - t)).min(1.0),
                (0.10 + 0.7 * t).min(1.0),
                0.85,
            )
        })
        .collect()
}

fn zero_alpha_buffer(width: usize, height: usize) -> PixelBuffer {
    (0..(width * height)).map(|_| (0.7, 0.2, 0.9, 0.0)).collect()
}

fn buffer_energy(buf: &PixelBuffer) -> f64 {
    buf.iter().map(|&(r, g, b, _)| r + g + b).sum::<f64>()
}

// ---------------------------------------------------------------------------
// Foundation invariants: art style, palettes, hue modes, grade presets.
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

    #[test]
    fn resolve_is_deterministic_under_cloned_rng(seed in any::<u64>()) {
        let mut rng_a = make_rng_from_u64(seed);
        let mut rng_b = make_rng_from_u64(seed);
        let cfg = RandomizableEffectConfig::default();
        let (r1, _) = cfg.resolve(&mut rng_a, 640, 360);
        let (r2, _) = cfg.resolve(&mut rng_b, 640, 360);
        prop_assert_eq!(r1.art_style, r2.art_style);
        prop_assert_eq!(r1.grade_preset, r2.grade_preset);
        prop_assert_eq!(r1.hue_palette_mode, r2.hue_palette_mode);
        prop_assert_eq!(r1.drift_character, r2.drift_character);
        prop_assert_eq!(r1.bloom_mode_choice, r2.bloom_mode_choice);
        prop_assert!((r1.hdr_scale - r2.hdr_scale).abs() < 1e-12);
        prop_assert!((r1.framing_zoom - r2.framing_zoom).abs() < 1e-12);
        prop_assert!((r1.starfield_strength - r2.starfield_strength).abs() < 1e-12);
    }

    #[test]
    fn resolved_config_is_in_safe_ranges(seed in any::<u64>()) {
        let mut rng = make_rng_from_u64(seed);
        let (r, _) = RandomizableEffectConfig::default().resolve(&mut rng, 320, 200);
        prop_assert!(r.hdr_scale >= 0.0 && r.hdr_scale <= 4.0,
            "hdr_scale out of range: {}", r.hdr_scale);
        prop_assert!(r.clip_black >= 0.0 && r.clip_black < 0.5);
        prop_assert!(r.clip_white > 0.5 && r.clip_white <= 1.0);
        prop_assert!(r.framing_zoom >= 1.0 && r.framing_zoom <= 2.0,
            "framing_zoom out of clamp: {}", r.framing_zoom);
        prop_assert!(r.vignette_offset_x.abs() <= 0.5);
        prop_assert!(r.vignette_offset_y.abs() <= 0.5);
        prop_assert!(r.starfield_strength >= 0.0 && r.starfield_strength <= 1.0);
        prop_assert!(r.lens_flare_strength >= 0.0 && r.lens_flare_strength <= 1.0);
    }

    #[test]
    fn grade_preset_tints_are_bounded(grade_index in 0usize..16) {
        let preset = GradePreset::from_index(grade_index);
        let params = preset.params();
        // Tints are additive offsets in linear RGB; each channel must stay in
        // a physically plausible delta range. Values outside ±0.5 would push
        // highlights/shadows into non-displayable territory.
        for &ch in params.shadow_tint.iter().chain(params.highlight_tint.iter()) {
            prop_assert!(ch.abs() <= 0.5, "tint delta out of range: {}", ch);
        }
        prop_assert!(params.palette_wave_strength >= 0.0 && params.palette_wave_strength <= 1.0);
        prop_assert!((0.0..=2.0).contains(&params.vibrance_bias),
            "vibrance_bias out of range: {}", params.vibrance_bias);
        prop_assert!((0.5..=2.0).contains(&params.tone_curve_bias),
            "tone_curve_bias out of range: {}", params.tone_curve_bias);
    }

    #[test]
    fn hues_for_mode_are_in_unit_interval(mode_index in 0usize..8, seed in any::<u64>()) {
        let modes = [
            HuePaletteMode::Triadic,
            HuePaletteMode::Complementary,
            HuePaletteMode::Analogous,
            HuePaletteMode::SplitComplementary,
            HuePaletteMode::TetradicSquare,
            HuePaletteMode::Monochromatic,
            HuePaletteMode::Duotone,
            HuePaletteMode::CosmicWarmCool,
        ];
        let mode = modes[mode_index];
        let mut rng = make_rng_from_u64(seed);
        let hues = hues_for_mode(mode, &mut rng);
        for h in hues {
            prop_assert!(h.is_finite());
            prop_assert!((0.0..360.0).contains(&h),
                "hue out of [0, 360): {} for mode {:?}", h, mode);
        }
    }

    #[test]
    fn art_style_pick_is_defined_for_all_rngs(seed in any::<u64>()) {
        let mut rng = make_rng_from_u64(seed);
        let style = ArtStyle::pick(&mut rng);
        let bundle = style.bundle();
        // Bundle derived fields must be internally consistent.
        prop_assert_eq!(bundle.style, style);
        // name() must return a non-empty, ASCII-ish identifier.
        let name = style.name();
        prop_assert!(!name.is_empty());        prop_assert!(name.chars().all(|c| c.is_ascii_graphic()));
        prop_assert!(bundle.hdr_scale_bias > 0.0 && bundle.hdr_scale_bias <= 3.0);
        // Drift character must be a valid enum variant.
        match bundle.drift {
            DriftCharacter::None
            | DriftCharacter::Linear
            | DriftCharacter::Brownian
            | DriftCharacter::Elliptical
            | DriftCharacter::Circular
            | DriftCharacter::Spiral => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Post-effect invariants: alpha=0 passthrough, buffer-size preservation.
// ---------------------------------------------------------------------------

fn run_effect(
    effect: &dyn PostEffect,
    input: &PixelBuffer,
    width: usize,
    height: usize,
) -> PixelBuffer {
    effect.process(input, width, height).expect("post effect should not fail on unit test inputs")
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 16, ..ProptestConfig::default() })]

    #[test]
    fn starfield_preserves_buffer_size(
        w in 16usize..48,
        h in 16usize..48,
        seed in any::<u64>(),
    ) {
        let cfg = StarfieldConfig {
            strength: 0.8,
            density: 400.0,
            min_brightness: 0.05,
            max_brightness: 0.9,
            max_radius: 1.5,
            warmth_bias: 0.0,
            seed,
            avoid_luminous_regions: 0.3,
        };
        let input = sample_buffer(w, h);
        let out = run_effect(&Starfield::new(cfg), &input, w, h);
        prop_assert_eq!(out.len(), input.len());
    }

    #[test]
    fn starfield_adds_light_on_empty_canvas(
        w in 32usize..64,
        h in 32usize..64,
        seed in any::<u64>(),
    ) {
        let cfg = StarfieldConfig {
            strength: 0.9,
            density: 600.0,
            min_brightness: 0.10,
            max_brightness: 0.95,
            max_radius: 1.2,
            warmth_bias: 0.0,
            seed,
            avoid_luminous_regions: 0.0,
        };
        // Use a completely dark, fully-opaque background; starfield should only
        // add non-negative light on top of it.
        let input: PixelBuffer = (0..(w * h)).map(|_| (0.0, 0.0, 0.0, 1.0)).collect();
        let out = run_effect(&Starfield::new(cfg), &input, w, h);
        prop_assert!(buffer_energy(&out) >= buffer_energy(&input));
    }

    #[test]
    fn lens_flare_never_darkens(
        w in 24usize..48,
        h in 24usize..48,
    ) {
        let cfg = LensFlareConfig {
            strength: 0.6,
            luminance_threshold: 0.3,
            ghost_count: 4,
            ghost_spread: 0.5,
            streak_strength: 0.25,
            streak_length: 0.3,
            streak_tint: [1.0, 0.95, 0.82],
            ghost_tint: [0.92, 0.96, 1.05],
        };
        let input = sample_buffer(w, h);
        let input_energy = buffer_energy(&input);
        let out = run_effect(&LensFlare::new(cfg), &input, w, h);
        prop_assert!(buffer_energy(&out) + 1e-9 >= input_energy,
            "lens flare should never reduce total energy");
    }

    #[test]
    fn effects_do_not_corrupt_zero_alpha_to_nan(
        w in 16usize..32,
        h in 16usize..32,
    ) {
        let input = zero_alpha_buffer(w, h);
        let effects: Vec<Box<dyn PostEffect>> = vec![
            Box::new(CinematicColorGrade::new(ColorGradeParams::default())),
            Box::new(GlowEnhancement::new(GlowEnhancementConfig::default())),
            Box::new(ChromaticBloom::new(ChromaticBloomConfig::default())),
            Box::new(PerceptualBlur::new(PerceptualBlurConfig::default())),
            Box::new(MicroContrast::new(MicroContrastConfig::default())),
            Box::new(EdgeLuminance::new(EdgeLuminanceConfig::default())),
            Box::new(AtmosphericDepth::new(AtmosphericDepthConfig::default())),
            Box::new(FineTexture::new(FineTextureConfig::default())),
            Box::new(Opalescence::new(OpalescenceConfig::default())),
            Box::new(GradientMap::new(GradientMapConfig::default())),
            Box::new(Starfield::new(StarfieldConfig {
                strength: 0.4,
                density: 200.0,
                min_brightness: 0.05,
                max_brightness: 0.8,
                max_radius: 1.0,
                warmth_bias: 0.0,
                seed: 7,
                avoid_luminous_regions: 0.1,
            })),
            Box::new(LensFlare::new(LensFlareConfig::default())),
        ];
        for effect in &effects {
            let out = run_effect(effect.as_ref(), &input, w, h);
            for &(r, g, b, a) in &out {
                prop_assert!(r.is_finite() && g.is_finite() && b.is_finite() && a.is_finite(),
                    "effect produced non-finite value at alpha=0");
            }
            prop_assert_eq!(out.len(), input.len());
        }
    }
}

// ---------------------------------------------------------------------------
// Monotone invariants: scaling strength never decreases energy on the effects
// that are supposed to be additive (glow, bloom-like operators).
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 12, ..ProptestConfig::default() })]

    #[test]
    fn glow_strength_is_monotone(
        w in 24usize..48,
        h in 24usize..48,
    ) {
        let input = sample_buffer(w, h);
        let low = GlowEnhancementConfig { strength: 0.2, ..GlowEnhancementConfig::default() };
        let high = GlowEnhancementConfig { strength: 0.8, ..GlowEnhancementConfig::default() };
        let out_low = run_effect(&GlowEnhancement::new(low), &input, w, h);
        let out_high = run_effect(&GlowEnhancement::new(high), &input, w, h);
        // Additive glow must not reduce total energy when strength goes up.
        prop_assert!(buffer_energy(&out_high) + 1e-6 >= buffer_energy(&out_low));
    }
}

#[test]
fn post_effect_configs_have_sensible_defaults() {
    let _ = ColorGradeParams::default();
    let _ = GlowEnhancementConfig::default();
    let _ = ChromaticBloomConfig::default();
    let _ = PerceptualBlurConfig::default();
    let _ = MicroContrastConfig::default();
    let _ = EdgeLuminanceConfig::default();
    let _ = AtmosphericDepthConfig::default();
    let _ = FineTextureConfig::default();
    let _ = OpalescenceConfig::default();
    let _ = GradientMapConfig::default();
    let _ = StarfieldConfig::default();
    let _ = LensFlareConfig::default();
}

// ---------------------------------------------------------------------------
// Framing invariants: the orbit centroid must land at the image center, and
// the world-to-pixel scale must be isotropic after aspect correction.
// ---------------------------------------------------------------------------
//
// The rendering pipeline always turns on aspect correction now, so these two
// invariants are the foundation of "images are centered" across every
// supported resolution. A regression here (e.g. accidentally asymmetric
// padding) would squish or off-center every future render.

fn make_positions(points: &[(f64, f64)]) -> Vec<Vec<Vector3<f64>>> {
    (0..3).map(|_| points.iter().map(|&(x, y)| Vector3::new(x, y, 0.0)).collect()).collect()
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

    #[test]
    fn centered_render_context_maps_bounds_midpoint_to_image_center(
        min_x in -500.0f64..500.0,
        max_x_offset in 1.0f64..400.0,
        min_y in -500.0f64..500.0,
        max_y_offset in 1.0f64..400.0,
        target_w in 64u32..2560,
        target_h in 64u32..2560,
        zoom in 1.0f64..2.0,
    ) {
        let positions = make_positions(&[
            (min_x, min_y),
            (min_x + max_x_offset, min_y + max_y_offset),
        ]);
        let ctx = RenderContext::new_with_framing(target_w, target_h, &positions, true, zoom);
        let bbox = ctx.bounds();
        let cx_world = f64::midpoint(bbox.min_x, bbox.max_x);
        let cy_world = f64::midpoint(bbox.min_y, bbox.max_y);
        let (px, py) = ctx.to_pixel(cx_world, cy_world);

        // The midpoint of the framing bbox must land within half a pixel of
        // the image center. Half a pixel is the quantization floor for a
        // world-to-pixel mapping done in f64 and cast to f32.
        prop_assert!(
            (f64::from(px) - f64::from(target_w) * 0.5).abs() <= 0.5,
            "center X off at {target_w}x{target_h} zoom {zoom}: px={px}, expected ~{}",
            f64::from(target_w) * 0.5
        );
        prop_assert!(
            (f64::from(py) - f64::from(target_h) * 0.5).abs() <= 0.5,
            "center Y off at {target_w}x{target_h} zoom {zoom}: py={py}, expected ~{}",
            f64::from(target_h) * 0.5
        );
    }

    #[test]
    fn proportional_scale_after_aspect_correction_is_isotropic(
        min_x in -500.0f64..500.0,
        max_x_offset in 1.0f64..400.0,
        min_y in -500.0f64..500.0,
        max_y_offset in 1.0f64..400.0,
        target_w in 64u32..2560,
        target_h in 64u32..2560,
    ) {
        let positions = make_positions(&[
            (min_x, min_y),
            (min_x + max_x_offset, min_y + max_y_offset),
        ]);
        let ctx = RenderContext::new_with_framing(target_w, target_h, &positions, true, 1.0);
        let bbox = ctx.bounds();

        let scale_x = bbox.width / f64::from(target_w);
        let scale_y = bbox.height / f64::from(target_h);
        let rel = (scale_x - scale_y).abs() / scale_x.max(scale_y).max(1e-12);

        // Aspect correction pads the shorter axis of the bounding box so the
        // world-to-pixel scale is identical on x and y. Any relative mismatch
        // above 1e-6 means the image will be subtly stretched.
        prop_assert!(
            rel <= 1e-6,
            "isotropy broken at {target_w}x{target_h}: scale_x={scale_x}, scale_y={scale_y}, rel={rel}",
        );
    }
}
