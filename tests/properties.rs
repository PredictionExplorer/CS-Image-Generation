//! Property-based invariants for curated render configuration.

use proptest::prelude::*;
use three_body_problem::render::grade_presets::GradePreset;
use three_body_problem::render::parameter_descriptors as pd;
use three_body_problem::render::randomizable_config::RandomizableEffectConfig;
use three_body_problem::sim::Sha3RandomByteStream;

const MIN_MASS: f64 = 100.0;
const MAX_MASS: f64 = 300.0;
const LOCATION: f64 = 300.0;
const VELOCITY: f64 = 1.0;

fn make_rng(seed: u64) -> Sha3RandomByteStream {
    Sha3RandomByteStream::new(&seed.to_le_bytes(), MIN_MASS, MAX_MASS, LOCATION, VELOCITY)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, ..ProptestConfig::default() })]

    #[test]
    fn resolve_is_deterministic(seed in any::<u64>()) {
        let cfg = RandomizableEffectConfig::default();
        let mut rng_a = make_rng(seed);
        let mut rng_b = make_rng(seed);
        let (a, _) = cfg.resolve(&mut rng_a, 640, 360);
        let (b, _) = cfg.resolve(&mut rng_b, 640, 360);

        prop_assert_eq!(a.composition_mode, b.composition_mode);
        prop_assert_eq!(a.finish_cluster, b.finish_cluster);
        prop_assert_eq!(a.grade_family, b.grade_family);
        prop_assert_eq!(a.grade_preset, b.grade_preset);
        prop_assert_eq!(a.gradient_map_palette, b.gradient_map_palette);
        prop_assert_eq!(a.framing_zoom.to_bits(), b.framing_zoom.to_bits());
        prop_assert_eq!(a.framing_shift_x.to_bits(), b.framing_shift_x.to_bits());
        prop_assert_eq!(a.framing_shift_y.to_bits(), b.framing_shift_y.to_bits());
        prop_assert_eq!(a.vignette_offset_x.to_bits(), b.vignette_offset_x.to_bits());
        prop_assert_eq!(a.vignette_offset_y.to_bits(), b.vignette_offset_y.to_bits());
        prop_assert_eq!(a.drift_scale_bias.to_bits(), b.drift_scale_bias.to_bits());
        prop_assert_eq!(a.drift_arc_bias.to_bits(), b.drift_arc_bias.to_bits());
        prop_assert_eq!(a.drift_eccentricity_bias.to_bits(), b.drift_eccentricity_bias.to_bits());
    }

    #[test]
    fn resolved_config_stays_in_safe_ranges(seed in any::<u64>()) {
        let mut rng = make_rng(seed);
        let (resolved, _) = RandomizableEffectConfig::default().resolve(&mut rng, 640, 360);

        prop_assert!((1.0..=1.20).contains(&resolved.framing_zoom));
        prop_assert!(resolved.framing_shift_x >= pd::FRAMING_SHIFT_X.min);
        prop_assert!(resolved.framing_shift_x <= pd::FRAMING_SHIFT_X.max);
        prop_assert!(resolved.framing_shift_y >= pd::FRAMING_SHIFT_Y.min);
        prop_assert!(resolved.framing_shift_y <= pd::FRAMING_SHIFT_Y.max);
        prop_assert!(resolved.vignette_offset_x >= pd::VIGNETTE_OFFSET_X.min);
        prop_assert!(resolved.vignette_offset_x <= pd::VIGNETTE_OFFSET_X.max);
        prop_assert!(resolved.vignette_offset_y >= pd::VIGNETTE_OFFSET_Y.min);
        prop_assert!(resolved.vignette_offset_y <= pd::VIGNETTE_OFFSET_Y.max);
        prop_assert!(resolved.gradient_map_palette <= pd::GRADIENT_MAP_PALETTE.max);
        prop_assert!(resolved.drift_scale_bias >= 0.85 && resolved.drift_scale_bias <= 1.15);
        prop_assert!(resolved.drift_arc_bias >= 0.80 && resolved.drift_arc_bias <= 1.20);
        prop_assert!(resolved.drift_eccentricity_bias >= 0.95 && resolved.drift_eccentricity_bias <= 1.08);
        prop_assert!(resolved.fine_texture_anisotropy >= 0.0 && resolved.fine_texture_anisotropy <= 1.0);
        prop_assert!(resolved.fine_texture_angle.is_finite());
        prop_assert!(resolved.champleve_cell_density > 0.0);
        prop_assert!(resolved.aether_filament_density > 0.0);
        prop_assert!(resolved.aether_iridescence_frequency > 0.0);
    }
}

#[test]
fn grade_preset_parameters_are_bounded() {
    for &preset in GradePreset::all() {
        let params = preset.params();
        for &channel in params.shadow_tint.iter().chain(params.highlight_tint.iter()) {
            assert!((-1.0..=1.0).contains(&channel));
        }
        assert!((0.0..=1.0).contains(&params.palette_wave_strength));
        assert!((0.0..=2.0).contains(&params.vibrance_bias));
        assert!((0.5..=2.0).contains(&params.tone_curve_bias));
    }
}
