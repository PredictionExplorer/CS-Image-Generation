//! Constraint rules for resolved randomizable effect configurations.

use super::ResolvedEffectConfig;
use crate::render::effect_randomizer::{
    RandomizationLog, RandomizationRecord, RandomizedParameter,
};

fn softness_stack_score(config: &ResolvedEffectConfig) -> f64 {
    let mut score = 0.0;
    if config.enable_bloom {
        score += 1.0;
    }
    if config.enable_chromatic_bloom {
        score += 1.05;
    }
    if config.enable_perceptual_blur {
        score += 1.0;
    }
    if config.enable_glow {
        score += 0.65;
    }
    if config.enable_atmospheric_depth {
        score += 0.45;
    }
    score
}

fn heavy_softness_count(config: &ResolvedEffectConfig) -> usize {
    usize::from(config.enable_bloom)
        + usize::from(config.enable_chromatic_bloom)
        + usize::from(config.enable_perceptual_blur)
}

fn cap_expensive_opalescence(config: &mut ResolvedEffectConfig, adjustments: &mut Vec<String>) {
    if config.enable_opalescence
        && config.opalescence_layers > 5
        && config.opalescence_strength > 0.30
    {
        let original_layers = config.opalescence_layers;
        config.opalescence_layers = 5;
        adjustments.push(format!(
            "Performance guard: Capped opalescence_layers ({} -> 5) at high strength ({:.2}) to prevent exponential cost",
            original_layers, config.opalescence_strength
        ));
    }
}

fn disable_isolated_chromatic_bloom(
    config: &mut ResolvedEffectConfig,
    adjustments: &mut Vec<String>,
) {
    if config.enable_chromatic_bloom && !config.enable_bloom {
        config.enable_chromatic_bloom = false;
        adjustments.push(
            "Quality guard: Disabled chromatic_bloom because base bloom was off, avoiding isolated prismatic haze"
                .to_string(),
        );
    }
}

fn soften_unsupported_atmospheric_depth(
    config: &mut ResolvedEffectConfig,
    adjustments: &mut Vec<String>,
) {
    if !config.enable_atmospheric_depth || config.enable_color_grade || config.enable_aether {
        return;
    }

    let original_strength = config.atmospheric_depth_strength;
    let original_desaturation = config.atmospheric_desaturation;

    config.atmospheric_depth_strength = config.atmospheric_depth_strength.min(0.12);
    config.atmospheric_desaturation = config.atmospheric_desaturation.min(0.16);

    if (config.atmospheric_depth_strength - original_strength).abs() > f64::EPSILON
        || (config.atmospheric_desaturation - original_desaturation).abs() > f64::EPSILON
    {
        adjustments.push(format!(
            "Quality guard: Softened atmospheric depth without color support (strength: {:.3} -> {:.3}, desaturation: {:.3} -> {:.3})",
            original_strength,
            config.atmospheric_depth_strength,
            original_desaturation,
            config.atmospheric_desaturation
        ));
    }
}

fn rebalance_gradient_map_without_color_grade(
    config: &mut ResolvedEffectConfig,
    adjustments: &mut Vec<String>,
) {
    if !config.enable_gradient_map || config.enable_color_grade {
        return;
    }

    let original_strength = config.gradient_map_strength;
    let original_hue_preservation = config.gradient_map_hue_preservation;

    config.gradient_map_strength = config.gradient_map_strength.min(0.35);
    config.gradient_map_hue_preservation = config.gradient_map_hue_preservation.max(0.50);

    if (config.gradient_map_strength - original_strength).abs() > f64::EPSILON
        || (config.gradient_map_hue_preservation - original_hue_preservation).abs() > f64::EPSILON
    {
        adjustments.push(format!(
            "Quality guard: Rebalanced gradient map toward source hues (strength: {:.3} -> {:.3}, hue preservation: {:.3} -> {:.3})",
            original_strength,
            config.gradient_map_strength,
            original_hue_preservation,
            config.gradient_map_hue_preservation
        ));
    }
}

fn break_heavy_softness_stack(config: &mut ResolvedEffectConfig, adjustments: &mut Vec<String>) {
    while heavy_softness_count(config) >= 2 && softness_stack_score(config) >= 1.85 {
        let score_before = softness_stack_score(config);
        if config.enable_perceptual_blur {
            config.enable_perceptual_blur = false;
            adjustments.push(format!(
                "Quality guard: Disabled perceptual_blur to break softness stack (score: {score_before:.2})"
            ));
        } else if config.enable_chromatic_bloom {
            config.enable_chromatic_bloom = false;
            adjustments.push(format!(
                "Quality guard: Disabled chromatic_bloom to break softness stack (score: {score_before:.2})"
            ));
        } else {
            break;
        }
    }
}

fn rescue_detail_for_softness_stack(
    config: &mut ResolvedEffectConfig,
    softness_score: f64,
    adjustments: &mut Vec<String>,
) {
    if softness_score < 1.75 {
        return;
    }

    let original = config.clone();

    config.enable_micro_contrast = true;
    config.enable_edge_luminance = true;
    config.dog_strength = config.dog_strength.min(0.25);
    config.dog_sigma_scale = config.dog_sigma_scale.min(0.0048);
    config.glow_strength = config.glow_strength.min(0.24);
    config.glow_radius_scale = config.glow_radius_scale.min(0.0028);
    config.chromatic_bloom_strength = config.chromatic_bloom_strength.min(0.28);
    config.chromatic_bloom_radius_scale = config.chromatic_bloom_radius_scale.min(0.0032);
    config.chromatic_bloom_separation_scale = config.chromatic_bloom_separation_scale.min(0.0008);
    config.perceptual_blur_strength = config.perceptual_blur_strength.min(0.30);
    config.micro_contrast_strength = config.micro_contrast_strength.max(0.34);
    config.micro_contrast_radius = config.micro_contrast_radius.min(4);
    config.edge_luminance_strength = config.edge_luminance_strength.max(0.26);
    config.edge_luminance_threshold = config.edge_luminance_threshold.min(0.18);
    config.edge_luminance_brightness_boost = config.edge_luminance_brightness_boost.max(0.34);

    if *config != original {
        adjustments.push(format!(
            "Quality guard: Tightened softness stack and enabled detail rescue (score: {:.2}, micro_contrast: {} -> {}, edge_luminance: {} -> {})",
            softness_score,
            original.enable_micro_contrast,
            config.enable_micro_contrast,
            original.enable_edge_luminance,
            config.enable_edge_luminance
        ));
    }
}

fn remove_extreme_softness_blur(
    config: &mut ResolvedEffectConfig,
    softness_score: f64,
    adjustments: &mut Vec<String>,
) {
    if softness_score >= 2.6 && config.enable_chromatic_bloom && config.enable_perceptual_blur {
        config.enable_perceptual_blur = false;
        adjustments.push(format!(
            "Quality guard: Disabled perceptual_blur inside an extreme softness stack (score: {softness_score:.2})"
        ));
    }
}

fn log_constraint_adjustments(log: &mut RandomizationLog, adjustments: Vec<String>) {
    if adjustments.is_empty() {
        return;
    }

    let mut adjustment_record =
        RandomizationRecord::new("render_constraints".to_string(), true, false);

    for adjustment in adjustments {
        adjustment_record.parameters.push(RandomizedParameter {
            name: "constraint".to_string(),
            value: adjustment,
            was_randomized: false,
            range_used: "N/A".to_string(),
        });
    }

    log.add_record(adjustment_record);
}

/// Apply render constraints to prevent pathological runtime and low-quality effect combinations.
///
/// Philosophy: Maximum exploration with minimum intervention.
/// - Keep the generative space broad while blocking combinations that predictably fail QA
/// - Prefer soft caps over hard disables unless a combination is consistently harmful
/// - Preserve deterministic resolution and logging for every adjustment
pub(super) fn apply_conflict_detection(
    mut config: ResolvedEffectConfig,
    log: &mut RandomizationLog,
) -> ResolvedEffectConfig {
    let mut adjustments = Vec::new();
    cap_expensive_opalescence(&mut config, &mut adjustments);
    disable_isolated_chromatic_bloom(&mut config, &mut adjustments);
    soften_unsupported_atmospheric_depth(&mut config, &mut adjustments);
    rebalance_gradient_map_without_color_grade(&mut config, &mut adjustments);
    let initial_softness_score = softness_stack_score(&config);
    rescue_detail_for_softness_stack(&mut config, initial_softness_score, &mut adjustments);
    break_heavy_softness_stack(&mut config, &mut adjustments);

    let softness_score = softness_stack_score(&config);
    rescue_detail_for_softness_stack(&mut config, softness_score, &mut adjustments);
    remove_extreme_softness_blur(&mut config, softness_score, &mut adjustments);
    log_constraint_adjustments(log, adjustments);

    config
}
