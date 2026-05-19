//! Color space conversions and utilities

use crate::render::constants::{
    BASE_HUE_DRIFT, HUE_DRIFT_SCALE, HUE_FULL_CIRCLE, HUE_WAVE_AMPLITUDE, OKLAB_CHROMA_BASE,
    OKLAB_CHROMA_BASE_BOOSTED, OKLAB_CHROMA_RANGE, OKLAB_CHROMA_RANGE_BOOSTED,
    OKLAB_CHROMA_WAVE_AMPLITUDE, OKLAB_CHROMA_WAVE_AMPLITUDE_BOOSTED, OKLAB_LIGHTNESS_BASE,
    OKLAB_LIGHTNESS_RANGE, OKLAB_LIGHTNESS_WAVE_AMPLITUDE,
};
use crate::render::randomizable_config::ResolvedEffectConfig;
use crate::sim::Sha3RandomByteStream;
use nalgebra::Vector3;
use tracing::info;

/// Type alias for `OKLab` color (L, a, b components)
pub type OklabColor = (f64, f64, f64);

/// Small random hue variation for visual interest
const HUE_DRIFT_JITTER: f64 = 0.1;
const BODY_COUNT: usize = 3;
#[cfg(test)]
const MAX_THEME_ATTEMPTS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ArtDirectionKind {
    CelestialOpal,
    DarkLuxuryJewelry,
    SolarCalligraphy,
    AlienBioluminescence,
    JapaneseInkAurora,
    GlassCathedral,
    CosmicBotanical,
    MoltenMyth,
    ArcticPrism,
    SacredNeonSigil,
    DeepSeaRelic,
    PlasmaSilk,
    BlackVelvetGold,
    RainbowOilFilm,
    AncientStarMap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MaterialFinish {
    StainedGlass,
    OpalPearl,
    MoltenMetal,
    NeonInk,
    SilkThread,
    Bioluminescent,
    CrystalFacet,
    AncientRelic,
    CosmicDust,
    PorcelainKintsugi,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CompositionMode {
    MacroJewelry,
    CelestialMap,
    DiagonalSweep,
    CenteredSigil,
    NegativeSpace,
    PanoramicRiver,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RareEvent {
    None,
    TotalEclipse,
    RainbowScar,
    GoldLeafManuscript,
    AlienXRay,
    SolarOverexposure,
    FrozenAurora,
    AstralCartography,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PaletteThemeKind {
    NoirGold,
    SolarFlare,
    AlienJewel,
    RoyalPlasma,
    ArcticAurora,
    RoseChrome,
    ToxicLuxury,
    PrismaticBlack,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PaletteTheme {
    kind: PaletteThemeKind,
    role_hues: [f64; BODY_COUNT],
    drift_scale: f64,
    wave_amplitude: f64,
    chroma_offsets: [f64; BODY_COUNT],
    lightness_offsets: [f64; BODY_COUNT],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct OrbitArtMetrics {
    speed_energy: f64,
    curvature_energy: f64,
    closeness_energy: f64,
    depth_energy: f64,
    aspect_energy: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CreativeProfile {
    pub(crate) art_direction: ArtDirectionKind,
    pub(crate) material_finish: MaterialFinish,
    pub(crate) composition_mode: CompositionMode,
    pub(crate) rare_event: RareEvent,
    palette: PaletteTheme,
    gradient_map_palette: usize,
    hue_wave_freq: f64,
    pub(crate) alpha_scale: f64,
    pub(crate) body_alpha_multipliers: [f64; BODY_COUNT],
    pub(crate) framing_zoom: f64,
    pub(crate) framing_offset: (f64, f64),
}

fn wrap_hue(hue: f64) -> f64 {
    hue.rem_euclid(HUE_FULL_CIRCLE)
}

#[cfg(test)]
fn hue_distance(a: f64, b: f64) -> f64 {
    let delta = (a - b).abs().rem_euclid(HUE_FULL_CIRCLE);
    delta.min(HUE_FULL_CIRCLE - delta)
}

fn is_blue_green_cluster(hue: f64) -> bool {
    (95.0..=250.0).contains(&wrap_hue(hue))
}

#[cfg(test)]
fn is_warm_sector(hue: f64) -> bool {
    let hue = wrap_hue(hue);
    hue <= 78.0 || hue >= 300.0
}

fn jitter(rng: &mut Sha3RandomByteStream, degrees: f64) -> f64 {
    (rng.next_f64() * 2.0 - 1.0) * degrees
}

#[cfg(test)]
fn pick_theme_kind(rng: &mut Sha3RandomByteStream) -> PaletteThemeKind {
    // Warm and warm/cool contrast themes are intentionally heavier than cool themes.
    const WEIGHTED_THEMES: &[PaletteThemeKind] = &[
        PaletteThemeKind::NoirGold,
        PaletteThemeKind::NoirGold,
        PaletteThemeKind::SolarFlare,
        PaletteThemeKind::SolarFlare,
        PaletteThemeKind::AlienJewel,
        PaletteThemeKind::RoyalPlasma,
        PaletteThemeKind::RoyalPlasma,
        PaletteThemeKind::ArcticAurora,
        PaletteThemeKind::RoseChrome,
        PaletteThemeKind::RoseChrome,
        PaletteThemeKind::ToxicLuxury,
        PaletteThemeKind::PrismaticBlack,
    ];
    let idx = (rng.next_f64() * WEIGHTED_THEMES.len() as f64).floor() as usize;
    WEIGHTED_THEMES[idx.min(WEIGHTED_THEMES.len() - 1)]
}

fn weighted_pick<T: Copy>(rng: &mut Sha3RandomByteStream, choices: &[T]) -> T {
    let idx = (rng.next_f64() * choices.len() as f64).floor() as usize;
    choices[idx.min(choices.len() - 1)]
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

impl OrbitArtMetrics {
    pub(crate) fn from_positions(positions: &[Vec<Vector3<f64>>]) -> Self {
        let mut speed_sum = 0.0;
        let mut speed_count = 0usize;
        let mut curvature_sum = 0.0;
        let mut curvature_count = 0usize;
        let mut z_min = f64::INFINITY;
        let mut z_max = f64::NEG_INFINITY;
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;

        for body in positions {
            for point in body {
                x_min = x_min.min(point[0]);
                x_max = x_max.max(point[0]);
                y_min = y_min.min(point[1]);
                y_max = y_max.max(point[1]);
                z_min = z_min.min(point[2]);
                z_max = z_max.max(point[2]);
            }

            for pair in body.windows(2) {
                speed_sum += (pair[1] - pair[0]).norm();
                speed_count += 1;
            }

            for triple in body.windows(3) {
                let v1 = triple[1] - triple[0];
                let v2 = triple[2] - triple[1];
                let denom = v1.norm() * v2.norm();
                if denom > 1e-12 {
                    let angle = (v1.dot(&v2) / denom).clamp(-1.0, 1.0).acos();
                    curvature_sum += angle / std::f64::consts::PI;
                    curvature_count += 1;
                }
            }
        }

        let mut close_sum = 0.0;
        let mut close_count = 0usize;
        if positions.len() >= BODY_COUNT {
            let steps = positions.iter().map(Vec::len).min().unwrap_or(0);
            for step in 0..steps {
                for a in 0..BODY_COUNT {
                    for b in a + 1..BODY_COUNT {
                        let distance = (positions[a][step] - positions[b][step]).norm();
                        close_sum += 1.0 / (1.0 + distance);
                        close_count += 1;
                    }
                }
            }
        }

        let average_speed = if speed_count > 0 { speed_sum / speed_count as f64 } else { 0.0 };
        let average_curvature =
            if curvature_count > 0 { curvature_sum / curvature_count as f64 } else { 0.0 };
        let average_closeness = if close_count > 0 { close_sum / close_count as f64 } else { 0.0 };
        let depth_span = if z_min.is_finite() && z_max.is_finite() { z_max - z_min } else { 0.0 };
        let width =
            if x_min.is_finite() && x_max.is_finite() { (x_max - x_min).abs() } else { 0.0 };
        let height =
            if y_min.is_finite() && y_max.is_finite() { (y_max - y_min).abs() } else { 0.0 };
        let aspect = width.max(height) / width.min(height).max(1e-9);

        Self {
            speed_energy: clamp01(average_speed / 0.15),
            curvature_energy: clamp01(average_curvature * 2.5),
            closeness_energy: clamp01(average_closeness * 3.0),
            depth_energy: clamp01(depth_span / (width.max(height).max(1e-9) * 0.8)),
            aspect_energy: clamp01((aspect - 1.0) / 2.5),
        }
    }
}

impl CreativeProfile {
    pub(crate) fn choose(rng: &mut Sha3RandomByteStream, positions: &[Vec<Vector3<f64>>]) -> Self {
        Self::from_metrics(rng, OrbitArtMetrics::from_positions(positions))
    }

    fn from_metrics(rng: &mut Sha3RandomByteStream, metrics: OrbitArtMetrics) -> Self {
        let art_direction = choose_art_direction(rng, metrics);
        let material_finish = choose_material_finish(rng, art_direction, metrics);
        let composition_mode = choose_composition_mode(rng, art_direction, metrics);
        let rare_event = choose_rare_event(rng, art_direction, metrics);
        let palette = PaletteTheme::from_art_direction(art_direction, rng);
        let gradient_map_palette = gradient_palette_for_direction(art_direction, rare_event, rng);
        let body_alpha_multipliers = alpha_multipliers_for_direction(art_direction, rare_event);
        let (framing_zoom, framing_offset) = framing_for_composition(composition_mode, rng);

        Self {
            art_direction,
            material_finish,
            composition_mode,
            rare_event,
            palette,
            gradient_map_palette,
            hue_wave_freq: hue_wave_frequency_for_direction(art_direction, rare_event, rng),
            alpha_scale: alpha_scale_for_composition(composition_mode, rare_event),
            body_alpha_multipliers,
            framing_zoom,
            framing_offset,
        }
    }

    pub(crate) fn apply_to_effect_config(self, config: &mut ResolvedEffectConfig) {
        config.gradient_map_palette = self.gradient_map_palette;
        config.enable_color_grade = true;
        config.enable_gradient_map = true;
        config.gradient_map_strength =
            (config.gradient_map_strength * profile_gradient_strength(self)).clamp(0.18, 0.58);
        config.gradient_map_hue_preservation = (config.gradient_map_hue_preservation
            * profile_hue_preservation(self))
        .clamp(0.25, 0.72);
        config.color_grade_strength =
            (config.color_grade_strength * profile_color_grade_strength(self)).clamp(0.45, 0.82);
        config.vibrance = (config.vibrance * profile_vibrance(self)).clamp(1.05, 1.55);
        config.clarity_strength =
            (config.clarity_strength * profile_clarity(self)).clamp(0.28, 0.72);
        config.tone_curve_strength =
            (config.tone_curve_strength * profile_tone_curve(self)).clamp(0.40, 0.90);
        config.vignette_strength =
            (config.vignette_strength * profile_vignette(self)).clamp(0.18, 0.72);
        config.hdr_scale = (config.hdr_scale * profile_hdr(self)).clamp(0.06, 0.24);
        config.composition_zoom = self.framing_zoom;
        config.composition_offset_x = self.framing_offset.0;
        config.composition_offset_y = self.framing_offset.1;

        match self.material_finish {
            MaterialFinish::StainedGlass => {
                config.enable_champleve = true;
                config.champleve_rim_intensity = config.champleve_rim_intensity.max(1.8);
                config.champleve_interference_amplitude =
                    config.champleve_interference_amplitude.max(0.55);
            }
            MaterialFinish::OpalPearl => {
                config.enable_opalescence = true;
                config.opalescence_strength = config.opalescence_strength.max(0.10);
                config.enable_perceptual_blur =
                    config.enable_perceptual_blur || self.rare_event == RareEvent::FrozenAurora;
            }
            MaterialFinish::MoltenMetal => {
                config.enable_bloom = true;
                config.enable_glow = true;
                config.enable_champleve = true;
                config.champleve_rim_warmth = config.champleve_rim_warmth.max(0.76);
            }
            MaterialFinish::NeonInk => {
                config.enable_edge_luminance = true;
                config.enable_micro_contrast = true;
                config.edge_luminance_strength = config.edge_luminance_strength.max(0.36);
                config.micro_contrast_strength = config.micro_contrast_strength.max(0.40);
            }
            MaterialFinish::SilkThread => {
                config.enable_fine_texture = true;
                config.enable_aether = true;
                config.aether_flow_alignment = config.aether_flow_alignment.max(0.78);
            }
            MaterialFinish::Bioluminescent => {
                config.enable_glow = true;
                config.enable_aether = true;
                config.glow_saturation_boost = config.glow_saturation_boost.max(0.30);
            }
            MaterialFinish::CrystalFacet => {
                config.enable_opalescence = true;
                config.enable_chromatic_bloom = true;
                config.enable_bloom = true;
                config.chromatic_bloom_strength = config.chromatic_bloom_strength.max(0.24);
            }
            MaterialFinish::AncientRelic => {
                config.enable_champleve = true;
                config.enable_fine_texture = true;
                config.champleve_rim_warmth = config.champleve_rim_warmth.max(0.72);
            }
            MaterialFinish::CosmicDust => {
                config.enable_atmospheric_depth = true;
                config.enable_glow = true;
                config.atmospheric_depth_strength = config.atmospheric_depth_strength.max(0.06);
            }
            MaterialFinish::PorcelainKintsugi => {
                config.enable_champleve = true;
                config.enable_fine_texture = true;
                config.champleve_rim_intensity = config.champleve_rim_intensity.max(2.0);
            }
        }

        match self.rare_event {
            RareEvent::None => {}
            RareEvent::TotalEclipse => {
                config.vignette_strength = config.vignette_strength.max(0.62);
                config.gradient_map_strength = config.gradient_map_strength.min(0.34);
                config.enable_edge_luminance = true;
            }
            RareEvent::RainbowScar => {
                config.enable_bloom = true;
                config.enable_chromatic_bloom = true;
                config.chromatic_bloom_strength = config.chromatic_bloom_strength.max(0.28);
                config.chromatic_bloom_separation_scale =
                    config.chromatic_bloom_separation_scale.max(0.0009);
            }
            RareEvent::GoldLeafManuscript | RareEvent::AstralCartography => {
                config.enable_champleve = true;
                config.enable_fine_texture = true;
                config.enable_edge_luminance = true;
                config.champleve_rim_warmth = config.champleve_rim_warmth.max(0.80);
            }
            RareEvent::AlienXRay => {
                config.enable_aether = true;
                config.enable_edge_luminance = true;
                config.aether_iridescence_amplitude = config.aether_iridescence_amplitude.max(0.70);
            }
            RareEvent::SolarOverexposure => {
                config.enable_bloom = true;
                config.enable_glow = true;
                config.blur_core_brightness = config.blur_core_brightness.max(12.0);
                config.hdr_scale = config.hdr_scale.max(0.18);
            }
            RareEvent::FrozenAurora => {
                config.enable_opalescence = true;
                config.enable_atmospheric_depth = true;
                config.atmospheric_desaturation = config.atmospheric_desaturation.max(0.10);
            }
        }

        if config.enable_chromatic_bloom {
            config.enable_bloom = true;
        }
    }
}

impl PaletteTheme {
    #[cfg(test)]
    fn choose(rng: &mut Sha3RandomByteStream) -> Self {
        let fallback_root = rng.next_f64() * HUE_FULL_CIRCLE;
        for _ in 0..MAX_THEME_ATTEMPTS {
            let kind = pick_theme_kind(rng);
            let root = rng.next_f64() * HUE_FULL_CIRCLE;
            let theme = Self::from_kind(kind, root, rng);
            if theme.quality_score() >= 0.78 {
                return theme;
            }
        }

        Self::from_kind(PaletteThemeKind::NoirGold, fallback_root, rng)
    }

    fn from_kind(kind: PaletteThemeKind, root: f64, rng: &mut Sha3RandomByteStream) -> Self {
        let root = wrap_hue(root);
        match kind {
            PaletteThemeKind::NoirGold => Self {
                kind,
                role_hues: [
                    wrap_hue(280.0 + jitter(rng, 16.0)),
                    wrap_hue(45.0 + jitter(rng, 12.0)),
                    wrap_hue(190.0 + jitter(rng, 10.0)),
                ],
                drift_scale: 0.88,
                wave_amplitude: 26.0,
                chroma_offsets: [0.00, 0.04, 0.02],
                lightness_offsets: [-0.05, 0.05, 0.02],
            },
            PaletteThemeKind::SolarFlare => Self {
                kind,
                role_hues: [
                    wrap_hue(8.0 + jitter(rng, 18.0)),
                    wrap_hue(36.0 + jitter(rng, 14.0)),
                    wrap_hue(214.0 + jitter(rng, 12.0)),
                ],
                drift_scale: 1.02,
                wave_amplitude: 32.0,
                chroma_offsets: [0.05, 0.02, 0.03],
                lightness_offsets: [0.02, 0.06, -0.02],
            },
            PaletteThemeKind::AlienJewel => Self {
                kind,
                role_hues: [
                    wrap_hue(154.0 + jitter(rng, 12.0)),
                    wrap_hue(310.0 + jitter(rng, 16.0)),
                    wrap_hue(22.0 + jitter(rng, 12.0)),
                ],
                drift_scale: 0.96,
                wave_amplitude: 30.0,
                chroma_offsets: [0.02, 0.05, 0.03],
                lightness_offsets: [-0.01, 0.05, 0.02],
            },
            PaletteThemeKind::RoyalPlasma => Self {
                kind,
                role_hues: [
                    wrap_hue(268.0 + jitter(rng, 16.0)),
                    wrap_hue(320.0 + jitter(rng, 16.0)),
                    wrap_hue(48.0 + jitter(rng, 10.0)),
                ],
                drift_scale: 0.92,
                wave_amplitude: 34.0,
                chroma_offsets: [0.03, 0.05, 0.02],
                lightness_offsets: [-0.02, 0.04, 0.06],
            },
            PaletteThemeKind::ArcticAurora => Self {
                kind,
                role_hues: [
                    wrap_hue(205.0 + jitter(rng, 12.0)),
                    wrap_hue(284.0 + jitter(rng, 16.0)),
                    wrap_hue(326.0 + jitter(rng, 14.0)),
                ],
                drift_scale: 0.82,
                wave_amplitude: 24.0,
                chroma_offsets: [0.00, 0.04, 0.05],
                lightness_offsets: [0.02, 0.03, 0.05],
            },
            PaletteThemeKind::RoseChrome => Self {
                kind,
                role_hues: [
                    wrap_hue(340.0 + jitter(rng, 14.0)),
                    wrap_hue(268.0 + jitter(rng, 12.0)),
                    wrap_hue(204.0 + jitter(rng, 10.0)),
                ],
                drift_scale: 0.86,
                wave_amplitude: 25.0,
                chroma_offsets: [0.05, 0.02, 0.01],
                lightness_offsets: [0.05, 0.02, 0.04],
            },
            PaletteThemeKind::ToxicLuxury => Self {
                kind,
                role_hues: [
                    wrap_hue(74.0 + jitter(rng, 10.0)),
                    wrap_hue(286.0 + jitter(rng, 14.0)),
                    wrap_hue(24.0 + jitter(rng, 12.0)),
                ],
                drift_scale: 0.90,
                wave_amplitude: 22.0,
                chroma_offsets: [0.05, 0.02, 0.03],
                lightness_offsets: [0.06, -0.03, 0.02],
            },
            PaletteThemeKind::PrismaticBlack => {
                let second = wrap_hue(root + 138.0 + jitter(rng, 12.0));
                let mut third = wrap_hue(root + 274.0 + jitter(rng, 12.0));
                if [root, second, third].iter().all(|&h| is_blue_green_cluster(h)) {
                    third = wrap_hue(32.0 + jitter(rng, 10.0));
                }
                Self {
                    kind,
                    role_hues: [root, second, third],
                    drift_scale: 1.12,
                    wave_amplitude: 42.0,
                    chroma_offsets: [0.00, 0.02, 0.06],
                    lightness_offsets: [-0.06, 0.01, 0.07],
                }
            }
        }
    }

    fn from_art_direction(direction: ArtDirectionKind, rng: &mut Sha3RandomByteStream) -> Self {
        let kind = match direction {
            ArtDirectionKind::CelestialOpal => PaletteThemeKind::RoseChrome,
            ArtDirectionKind::DarkLuxuryJewelry => PaletteThemeKind::NoirGold,
            ArtDirectionKind::SolarCalligraphy => PaletteThemeKind::SolarFlare,
            ArtDirectionKind::AlienBioluminescence => PaletteThemeKind::AlienJewel,
            ArtDirectionKind::JapaneseInkAurora => PaletteThemeKind::ArcticAurora,
            ArtDirectionKind::GlassCathedral => PaletteThemeKind::RoyalPlasma,
            ArtDirectionKind::CosmicBotanical => PaletteThemeKind::AlienJewel,
            ArtDirectionKind::MoltenMyth => PaletteThemeKind::SolarFlare,
            ArtDirectionKind::ArcticPrism => PaletteThemeKind::ArcticAurora,
            ArtDirectionKind::SacredNeonSigil => PaletteThemeKind::ToxicLuxury,
            ArtDirectionKind::DeepSeaRelic => PaletteThemeKind::AlienJewel,
            ArtDirectionKind::PlasmaSilk => PaletteThemeKind::RoyalPlasma,
            ArtDirectionKind::BlackVelvetGold => PaletteThemeKind::NoirGold,
            ArtDirectionKind::RainbowOilFilm => PaletteThemeKind::PrismaticBlack,
            ArtDirectionKind::AncientStarMap => PaletteThemeKind::NoirGold,
        };
        let root = rng.next_f64() * HUE_FULL_CIRCLE;
        Self::from_kind(kind, root, rng)
    }

    #[cfg(test)]
    fn quality_score(&self) -> f64 {
        let cool_count = self.role_hues.iter().filter(|&&h| is_blue_green_cluster(h)).count();
        let warm_count = self.role_hues.iter().filter(|&&h| is_warm_sector(h)).count();
        let min_distance = self
            .role_hues
            .iter()
            .enumerate()
            .flat_map(|(idx, &hue)| {
                self.role_hues.iter().skip(idx + 1).map(move |&other| hue_distance(hue, other))
            })
            .fold(HUE_FULL_CIRCLE, f64::min);
        let max_distance = self
            .role_hues
            .iter()
            .enumerate()
            .flat_map(|(idx, &hue)| {
                self.role_hues.iter().skip(idx + 1).map(move |&other| hue_distance(hue, other))
            })
            .fold(0.0_f64, f64::max);

        let has_analogous_pair_with_accent =
            min_distance < 35.0 && max_distance >= 130.0 && warm_count >= 2;
        let separation_score = if has_analogous_pair_with_accent {
            0.78
        } else {
            (min_distance / 105.0).clamp(0.0, 1.0)
        };
        let warm_score = if warm_count > 0 { 1.0 } else { 0.0 };
        let cool_balance_score = match cool_count {
            0 | 1 => 1.0,
            2 => 0.70,
            _ => 0.0,
        };
        let accent_score =
            if self.chroma_offsets.iter().any(|&offset| offset >= 0.05) { 1.0 } else { 0.75 };

        separation_score * 0.36
            + warm_score * 0.28
            + cool_balance_score * 0.26
            + accent_score * 0.10
    }

    fn hue_for_body(&self, body_index: usize) -> f64 {
        self.role_hues[body_index % BODY_COUNT]
    }
}

fn choose_art_direction(
    rng: &mut Sha3RandomByteStream,
    metrics: OrbitArtMetrics,
) -> ArtDirectionKind {
    if metrics.speed_energy > 0.82 && rng.next_f64() < 0.45 {
        return weighted_pick(
            rng,
            &[ArtDirectionKind::RainbowOilFilm, ArtDirectionKind::SolarCalligraphy],
        );
    }
    if metrics.closeness_energy > 0.78 && rng.next_f64() < 0.55 {
        return weighted_pick(
            rng,
            &[ArtDirectionKind::SolarCalligraphy, ArtDirectionKind::MoltenMyth],
        );
    }
    if metrics.curvature_energy > 0.72 && rng.next_f64() < 0.55 {
        return weighted_pick(
            rng,
            &[ArtDirectionKind::SacredNeonSigil, ArtDirectionKind::GlassCathedral],
        );
    }
    if metrics.depth_energy > 0.60 && rng.next_f64() < 0.45 {
        return weighted_pick(
            rng,
            &[ArtDirectionKind::CelestialOpal, ArtDirectionKind::DeepSeaRelic],
        );
    }
    if metrics.aspect_energy > 0.55 && rng.next_f64() < 0.45 {
        return weighted_pick(
            rng,
            &[ArtDirectionKind::PlasmaSilk, ArtDirectionKind::AncientStarMap],
        );
    }

    const ALL_DIRECTIONS: &[ArtDirectionKind] = &[
        ArtDirectionKind::CelestialOpal,
        ArtDirectionKind::DarkLuxuryJewelry,
        ArtDirectionKind::SolarCalligraphy,
        ArtDirectionKind::AlienBioluminescence,
        ArtDirectionKind::JapaneseInkAurora,
        ArtDirectionKind::GlassCathedral,
        ArtDirectionKind::CosmicBotanical,
        ArtDirectionKind::MoltenMyth,
        ArtDirectionKind::ArcticPrism,
        ArtDirectionKind::SacredNeonSigil,
        ArtDirectionKind::DeepSeaRelic,
        ArtDirectionKind::PlasmaSilk,
        ArtDirectionKind::BlackVelvetGold,
        ArtDirectionKind::RainbowOilFilm,
        ArtDirectionKind::AncientStarMap,
    ];
    weighted_pick(rng, ALL_DIRECTIONS)
}

fn choose_material_finish(
    rng: &mut Sha3RandomByteStream,
    direction: ArtDirectionKind,
    metrics: OrbitArtMetrics,
) -> MaterialFinish {
    match direction {
        ArtDirectionKind::CelestialOpal | ArtDirectionKind::ArcticPrism => {
            weighted_pick(rng, &[MaterialFinish::OpalPearl, MaterialFinish::CrystalFacet])
        }
        ArtDirectionKind::DarkLuxuryJewelry | ArtDirectionKind::BlackVelvetGold => {
            weighted_pick(rng, &[MaterialFinish::AncientRelic, MaterialFinish::MoltenMetal])
        }
        ArtDirectionKind::SolarCalligraphy | ArtDirectionKind::SacredNeonSigil => {
            weighted_pick(rng, &[MaterialFinish::NeonInk, MaterialFinish::MoltenMetal])
        }
        ArtDirectionKind::AlienBioluminescence | ArtDirectionKind::DeepSeaRelic => {
            weighted_pick(rng, &[MaterialFinish::Bioluminescent, MaterialFinish::CosmicDust])
        }
        ArtDirectionKind::GlassCathedral => MaterialFinish::StainedGlass,
        ArtDirectionKind::CosmicBotanical => MaterialFinish::Bioluminescent,
        ArtDirectionKind::MoltenMyth => MaterialFinish::MoltenMetal,
        ArtDirectionKind::JapaneseInkAurora => MaterialFinish::NeonInk,
        ArtDirectionKind::AncientStarMap => MaterialFinish::PorcelainKintsugi,
        ArtDirectionKind::PlasmaSilk => MaterialFinish::SilkThread,
        ArtDirectionKind::RainbowOilFilm => {
            if metrics.curvature_energy > 0.55 {
                MaterialFinish::CrystalFacet
            } else {
                MaterialFinish::OpalPearl
            }
        }
    }
}

fn choose_composition_mode(
    rng: &mut Sha3RandomByteStream,
    direction: ArtDirectionKind,
    metrics: OrbitArtMetrics,
) -> CompositionMode {
    if metrics.aspect_energy > 0.58 {
        return CompositionMode::PanoramicRiver;
    }
    match direction {
        ArtDirectionKind::DarkLuxuryJewelry | ArtDirectionKind::BlackVelvetGold => {
            weighted_pick(rng, &[CompositionMode::MacroJewelry, CompositionMode::NegativeSpace])
        }
        ArtDirectionKind::AncientStarMap | ArtDirectionKind::JapaneseInkAurora => {
            CompositionMode::CelestialMap
        }
        ArtDirectionKind::SacredNeonSigil | ArtDirectionKind::GlassCathedral => {
            CompositionMode::CenteredSigil
        }
        ArtDirectionKind::PlasmaSilk | ArtDirectionKind::SolarCalligraphy => {
            CompositionMode::DiagonalSweep
        }
        _ => weighted_pick(
            rng,
            &[
                CompositionMode::MacroJewelry,
                CompositionMode::CelestialMap,
                CompositionMode::DiagonalSweep,
                CompositionMode::NegativeSpace,
            ],
        ),
    }
}

fn choose_rare_event(
    rng: &mut Sha3RandomByteStream,
    direction: ArtDirectionKind,
    metrics: OrbitArtMetrics,
) -> RareEvent {
    let probability = (0.08 + metrics.closeness_energy * 0.05 + metrics.curvature_energy * 0.04)
        .clamp(0.06, 0.18);
    if rng.next_f64() >= probability {
        return RareEvent::None;
    }

    match direction {
        ArtDirectionKind::BlackVelvetGold => RareEvent::TotalEclipse,
        ArtDirectionKind::RainbowOilFilm => RareEvent::RainbowScar,
        ArtDirectionKind::AncientStarMap => RareEvent::AstralCartography,
        ArtDirectionKind::SolarCalligraphy | ArtDirectionKind::MoltenMyth => {
            weighted_pick(rng, &[RareEvent::SolarOverexposure, RareEvent::GoldLeafManuscript])
        }
        ArtDirectionKind::AlienBioluminescence | ArtDirectionKind::DeepSeaRelic => {
            RareEvent::AlienXRay
        }
        ArtDirectionKind::ArcticPrism | ArtDirectionKind::JapaneseInkAurora => {
            RareEvent::FrozenAurora
        }
        _ => weighted_pick(
            rng,
            &[
                RareEvent::RainbowScar,
                RareEvent::GoldLeafManuscript,
                RareEvent::AlienXRay,
                RareEvent::FrozenAurora,
            ],
        ),
    }
}

fn gradient_palette_for_direction(
    direction: ArtDirectionKind,
    rare_event: RareEvent,
    rng: &mut Sha3RandomByteStream,
) -> usize {
    if rare_event == RareEvent::RainbowScar {
        return weighted_pick(rng, &[1, 11, 14]);
    }
    match direction {
        ArtDirectionKind::CelestialOpal => weighted_pick(rng, &[8, 14, 0]),
        ArtDirectionKind::DarkLuxuryJewelry => weighted_pick(rng, &[0, 3, 12]),
        ArtDirectionKind::SolarCalligraphy => weighted_pick(rng, &[9, 12, 5]),
        ArtDirectionKind::AlienBioluminescence => weighted_pick(rng, &[1, 10, 11]),
        ArtDirectionKind::JapaneseInkAurora => weighted_pick(rng, &[6, 11, 3]),
        ArtDirectionKind::GlassCathedral => weighted_pick(rng, &[0, 2, 14]),
        ArtDirectionKind::CosmicBotanical => weighted_pick(rng, &[7, 13, 1]),
        ArtDirectionKind::MoltenMyth => weighted_pick(rng, &[9, 12, 5]),
        ArtDirectionKind::ArcticPrism => weighted_pick(rng, &[8, 11, 14]),
        ArtDirectionKind::SacredNeonSigil => weighted_pick(rng, &[11, 14, 1]),
        ArtDirectionKind::DeepSeaRelic => weighted_pick(rng, &[10, 13, 7]),
        ArtDirectionKind::PlasmaSilk => weighted_pick(rng, &[14, 1, 3]),
        ArtDirectionKind::BlackVelvetGold => weighted_pick(rng, &[0, 3, 12]),
        ArtDirectionKind::RainbowOilFilm => weighted_pick(rng, &[1, 11, 14]),
        ArtDirectionKind::AncientStarMap => weighted_pick(rng, &[5, 6, 0]),
    }
}

fn hue_wave_frequency_for_direction(
    direction: ArtDirectionKind,
    rare_event: RareEvent,
    rng: &mut Sha3RandomByteStream,
) -> f64 {
    let base = match direction {
        ArtDirectionKind::AncientStarMap | ArtDirectionKind::BlackVelvetGold => 1.4,
        ArtDirectionKind::PlasmaSilk | ArtDirectionKind::CelestialOpal => 2.0,
        ArtDirectionKind::SacredNeonSigil | ArtDirectionKind::RainbowOilFilm => 3.2,
        _ => 2.4,
    };
    let rare_boost = if rare_event == RareEvent::RainbowScar { 0.8 } else { 0.0 };
    base + rare_boost + rng.next_f64() * 1.4
}

fn alpha_multipliers_for_direction(
    direction: ArtDirectionKind,
    rare_event: RareEvent,
) -> [f64; BODY_COUNT] {
    let base = match direction {
        ArtDirectionKind::BlackVelvetGold | ArtDirectionKind::AncientStarMap => [0.72, 1.05, 1.28],
        ArtDirectionKind::SolarCalligraphy | ArtDirectionKind::MoltenMyth => [1.28, 1.10, 0.82],
        ArtDirectionKind::GlassCathedral | ArtDirectionKind::ArcticPrism => [0.92, 1.18, 1.02],
        ArtDirectionKind::PlasmaSilk => [0.78, 0.92, 1.20],
        _ => [1.0, 1.08, 0.94],
    };
    if rare_event == RareEvent::TotalEclipse {
        [base[0] * 0.70, base[1] * 0.82, base[2] * 1.35]
    } else {
        base
    }
}

fn framing_for_composition(
    composition: CompositionMode,
    rng: &mut Sha3RandomByteStream,
) -> (f64, (f64, f64)) {
    fn offset(rng: &mut Sha3RandomByteStream) -> f64 {
        (rng.next_f64() * 2.0 - 1.0) * 0.08
    }
    match composition {
        CompositionMode::MacroJewelry => (1.18 + rng.next_f64() * 0.12, (offset(rng), offset(rng))),
        CompositionMode::CelestialMap => {
            (0.82 + rng.next_f64() * 0.08, (offset(rng) * 0.6, offset(rng) * 0.6))
        }
        CompositionMode::DiagonalSweep => (1.02, (offset(rng) * 1.5, offset(rng) * 1.5)),
        CompositionMode::CenteredSigil => (1.05 + rng.next_f64() * 0.08, (0.0, 0.0)),
        CompositionMode::NegativeSpace => {
            (0.70 + rng.next_f64() * 0.10, (offset(rng), offset(rng)))
        }
        CompositionMode::PanoramicRiver => {
            (0.90 + rng.next_f64() * 0.08, (offset(rng) * 1.2, offset(rng) * 0.5))
        }
    }
}

fn alpha_scale_for_composition(composition: CompositionMode, rare_event: RareEvent) -> f64 {
    let base = match composition {
        CompositionMode::MacroJewelry => 1.16,
        CompositionMode::CelestialMap => 0.82,
        CompositionMode::DiagonalSweep => 1.04,
        CompositionMode::CenteredSigil => 1.08,
        CompositionMode::NegativeSpace => 0.68,
        CompositionMode::PanoramicRiver => 0.92,
    };
    if rare_event == RareEvent::SolarOverexposure { base * 1.18 } else { base }
}

fn profile_gradient_strength(profile: CreativeProfile) -> f64 {
    match profile.art_direction {
        ArtDirectionKind::AncientStarMap | ArtDirectionKind::BlackVelvetGold => 0.90,
        ArtDirectionKind::RainbowOilFilm | ArtDirectionKind::SacredNeonSigil => 1.28,
        ArtDirectionKind::GlassCathedral | ArtDirectionKind::CelestialOpal => 1.15,
        _ => 1.05,
    }
}

fn profile_hue_preservation(profile: CreativeProfile) -> f64 {
    match profile.art_direction {
        ArtDirectionKind::RainbowOilFilm | ArtDirectionKind::SacredNeonSigil => 0.76,
        ArtDirectionKind::AncientStarMap | ArtDirectionKind::BlackVelvetGold => 1.12,
        _ => 0.92,
    }
}

fn profile_color_grade_strength(profile: CreativeProfile) -> f64 {
    match profile.material_finish {
        MaterialFinish::NeonInk | MaterialFinish::MoltenMetal => 1.16,
        MaterialFinish::OpalPearl | MaterialFinish::CosmicDust => 0.96,
        _ => 1.06,
    }
}

fn profile_vibrance(profile: CreativeProfile) -> f64 {
    match profile.rare_event {
        RareEvent::RainbowScar | RareEvent::AlienXRay => 1.18,
        RareEvent::TotalEclipse => 0.92,
        _ => 1.06,
    }
}

fn profile_clarity(profile: CreativeProfile) -> f64 {
    match profile.material_finish {
        MaterialFinish::NeonInk | MaterialFinish::CrystalFacet | MaterialFinish::StainedGlass => {
            1.18
        }
        MaterialFinish::SilkThread | MaterialFinish::OpalPearl => 0.92,
        _ => 1.02,
    }
}

fn profile_tone_curve(profile: CreativeProfile) -> f64 {
    match profile.composition_mode {
        CompositionMode::NegativeSpace | CompositionMode::CelestialMap => 1.12,
        CompositionMode::MacroJewelry => 1.04,
        _ => 1.0,
    }
}

fn profile_vignette(profile: CreativeProfile) -> f64 {
    match profile.composition_mode {
        CompositionMode::NegativeSpace => 1.35,
        CompositionMode::MacroJewelry => 1.15,
        CompositionMode::CelestialMap => 0.88,
        _ => 1.0,
    }
}

fn profile_hdr(profile: CreativeProfile) -> f64 {
    match profile.rare_event {
        RareEvent::SolarOverexposure => 1.28,
        RareEvent::TotalEclipse => 0.86,
        _ => match profile.material_finish {
            MaterialFinish::MoltenMetal | MaterialFinish::NeonInk => 1.12,
            MaterialFinish::OpalPearl | MaterialFinish::SilkThread => 0.94,
            _ => 1.0,
        },
    }
}

fn generate_color_gradient_from_hue(
    rng: &mut Sha3RandomByteStream,
    length: usize,
    body_index: usize,
    base_hue: f64,
    base_hue_offset: f64,
    chroma_boost: bool,
    hue_wave_freq: f64,
    drift_scale: f64,
    wave_amplitude: f64,
    chroma_offset: f64,
    lightness_offset: f64,
) -> Vec<OklabColor> {
    let chroma_base = if chroma_boost { OKLAB_CHROMA_BASE_BOOSTED } else { OKLAB_CHROMA_BASE };
    let chroma_range = if chroma_boost { OKLAB_CHROMA_RANGE_BOOSTED } else { OKLAB_CHROMA_RANGE };
    let chroma_wave = if chroma_boost {
        OKLAB_CHROMA_WAVE_AMPLITUDE_BOOSTED
    } else {
        OKLAB_CHROMA_WAVE_AMPLITUDE
    };

    let mut colors = Vec::with_capacity(length);

    let ln_cache: Vec<f64> =
        (0..length).map(|i| if i > 0 { (i as f64).ln() } else { 0.0 }).collect();
    let wave_cache: Vec<f64> = (0..length)
        .map(|i| {
            let t = i as f64 / length.max(1) as f64;
            let phase_offset = body_index as f64 * 0.33 + rng.next_f64() * 0.1;
            ((phase_offset + t * hue_wave_freq) * std::f64::consts::TAU).sin()
        })
        .collect();

    let random_bits: Vec<u8> = (0..length).map(|_| rng.next_byte()).collect();
    let random_chromas: Vec<f64> = (0..length).map(|_| rng.next_f64()).collect();
    let random_lightnesses: Vec<f64> = (0..length).map(|_| rng.next_f64()).collect();

    for step in 0..length {
        let mut current_hue = base_hue
            + base_hue_offset * (1.0 + ln_cache[step]) * drift_scale
            + wave_cache[step] * wave_amplitude;

        if random_bits[step] & 1 == 0 {
            current_hue += HUE_DRIFT_JITTER;
        } else {
            current_hue -= HUE_DRIFT_JITTER;
        }
        current_hue = wrap_hue(current_hue);

        let wave_factor = wave_cache[step];
        let chroma = (chroma_base
            + random_chromas[step] * chroma_range
            + wave_factor * chroma_wave
            + body_index as f64 * 0.01
            + chroma_offset)
            .max(0.0);

        let lightness = (OKLAB_LIGHTNESS_BASE
            + random_lightnesses[step] * OKLAB_LIGHTNESS_RANGE
            + wave_factor * OKLAB_LIGHTNESS_WAVE_AMPLITUDE
            + body_index as f64 * 0.015
            + lightness_offset)
            .clamp(0.0, 1.0);

        let hue_rad = current_hue.to_radians();
        let a = chroma * hue_rad.cos();
        let b = chroma * hue_rad.sin();

        colors.push((lightness, a, b));
    }

    colors
}

/// Generate color gradient optimized for `OKLab` space.
///
/// Generates colors in `OKLCh` (cylindrical `OKLab`) for perceptually
/// uniform distribution. `chroma_boost` selects richer saturation
/// constants; `hue_wave_freq` controls per-seed color rhythm.
pub fn generate_color_gradient_oklab(
    rng: &mut Sha3RandomByteStream,
    length: usize,
    body_index: usize,
    base_hue_offset: f64,
    chroma_boost: bool,
    hue_wave_freq: f64,
) -> Vec<OklabColor> {
    let base_hue = rng.next_f64() * HUE_FULL_CIRCLE
        + body_index as f64 * 120.0
        + [0.0, 120.0, 240.0][body_index % BODY_COUNT];
    generate_color_gradient_from_hue(
        rng,
        length,
        body_index,
        base_hue,
        base_hue_offset,
        chroma_boost,
        hue_wave_freq,
        HUE_DRIFT_SCALE,
        HUE_WAVE_AMPLITUDE,
        0.0,
        0.0,
    )
}

/// Generate 3 color sequences + per-body alphas.
///
/// `chroma_boost`: use richer saturation constants.
/// `alpha_variation`: give each body a slightly different alpha for depth.
pub fn generate_body_color_sequences(
    rng: &mut Sha3RandomByteStream,
    length: usize,
    alpha_denom: usize,
    chroma_boost: bool,
    alpha_variation: bool,
) -> (Vec<Vec<OklabColor>>, Vec<f64>) {
    let profile = CreativeProfile::from_metrics(
        rng,
        OrbitArtMetrics {
            speed_energy: 0.5,
            curvature_energy: 0.5,
            closeness_energy: 0.5,
            depth_energy: 0.5,
            aspect_energy: 0.5,
        },
    );
    generate_body_color_sequences_with_profile(
        rng,
        length,
        alpha_denom,
        chroma_boost,
        alpha_variation,
        &profile,
    )
}

/// Generate 3 color sequences + per-body alphas from a full creative profile.
pub(crate) fn generate_body_color_sequences_with_profile(
    rng: &mut Sha3RandomByteStream,
    length: usize,
    alpha_denom: usize,
    chroma_boost: bool,
    alpha_variation: bool,
    profile: &CreativeProfile,
) -> (Vec<Vec<OklabColor>>, Vec<f64>) {
    let base_hue_offset = BASE_HUE_DRIFT;

    let palette = profile.palette;
    info!(
        "   => Art direction: {:?} / {:?} / {:?} / {:?}",
        profile.art_direction,
        profile.material_finish,
        profile.composition_mode,
        profile.rare_event
    );
    info!(
        "   => Color harmony: {:?} hues [{:.1}, {:.1}, {:.1}]",
        palette.kind, palette.role_hues[0], palette.role_hues[1], palette.role_hues[2]
    );

    let b1 = generate_color_gradient_from_hue(
        rng,
        length,
        0,
        palette.hue_for_body(0),
        base_hue_offset,
        chroma_boost,
        profile.hue_wave_freq,
        palette.drift_scale,
        palette.wave_amplitude,
        palette.chroma_offsets[0],
        palette.lightness_offsets[0],
    );
    let b2 = generate_color_gradient_from_hue(
        rng,
        length,
        1,
        palette.hue_for_body(1),
        base_hue_offset,
        chroma_boost,
        profile.hue_wave_freq,
        palette.drift_scale,
        palette.wave_amplitude,
        palette.chroma_offsets[1],
        palette.lightness_offsets[1],
    );
    let b3 = generate_color_gradient_from_hue(
        rng,
        length,
        2,
        palette.hue_for_body(2),
        base_hue_offset,
        chroma_boost,
        profile.hue_wave_freq,
        palette.drift_scale,
        palette.wave_amplitude,
        palette.chroma_offsets[2],
        palette.lightness_offsets[2],
    );

    let body_alphas = if alpha_variation {
        // Shuffle [13M, 15M, 17M] using the RNG for per-body depth hierarchy
        let mut denoms = [13_000_000.0_f64, 15_000_000.0, 17_000_000.0];
        for i in (1..3).rev() {
            let j = (rng.next_f64() * (i + 1) as f64).floor() as usize;
            denoms.swap(i, j);
        }
        let alphas = vec![
            (1.0 / denoms[0]) * profile.alpha_scale * profile.body_alpha_multipliers[0],
            (1.0 / denoms[1]) * profile.alpha_scale * profile.body_alpha_multipliers[1],
            (1.0 / denoms[2]) * profile.alpha_scale * profile.body_alpha_multipliers[2],
        ];
        info!(
            "   => Per-body alpha variation: {:.3e}, {:.3e}, {:.3e}",
            alphas[0], alphas[1], alphas[2]
        );
        alphas
    } else {
        let alpha_value = (1.0 / alpha_denom as f64) * profile.alpha_scale;
        info!("   => Profile body alpha: base 1/{alpha_denom} scaled to {alpha_value:.3e}");
        vec![
            alpha_value * profile.body_alpha_multipliers[0],
            alpha_value * profile.body_alpha_multipliers[1],
            alpha_value * profile.body_alpha_multipliers[2],
        ]
    };

    (vec![b1, b2, b3], body_alphas)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::Sha3RandomByteStream;

    fn hue_from_oklab((_, a, b): OklabColor) -> f64 {
        b.atan2(a).to_degrees().rem_euclid(HUE_FULL_CIRCLE)
    }

    fn sample_metric_positions(speed_scale: f64, close: bool) -> Vec<Vec<Vector3<f64>>> {
        (0..3)
            .map(|body| {
                (0..80)
                    .map(|step| {
                        let t = step as f64 * 0.08 * speed_scale + body as f64 * 2.0;
                        let radius =
                            if close { 0.18 + body as f64 * 0.02 } else { 1.0 + body as f64 };
                        Vector3::new(
                            radius * t.cos(),
                            radius * (t * 1.3).sin(),
                            (t * 0.7).sin() * speed_scale * 0.1,
                        )
                    })
                    .collect()
            })
            .collect()
    }

    fn circular_mean_hue(colors: &[OklabColor]) -> f64 {
        let (sum_cos, sum_sin) = colors.iter().fold((0.0, 0.0), |(sum_cos, sum_sin), &color| {
            let hue = hue_from_oklab(color).to_radians();
            (sum_cos + hue.cos(), sum_sin + hue.sin())
        });
        sum_sin.atan2(sum_cos).to_degrees().rem_euclid(HUE_FULL_CIRCLE)
    }

    fn assert_theme_quality(theme: PaletteTheme) {
        let cool_count = theme.role_hues.iter().filter(|&&hue| is_blue_green_cluster(hue)).count();
        let warm_count = theme.role_hues.iter().filter(|&&hue| is_warm_sector(hue)).count();
        let min_distance = theme
            .role_hues
            .iter()
            .enumerate()
            .flat_map(|(idx, &hue)| {
                theme.role_hues.iter().skip(idx + 1).map(move |&other| hue_distance(hue, other))
            })
            .fold(HUE_FULL_CIRCLE, f64::min);
        let max_distance = theme
            .role_hues
            .iter()
            .enumerate()
            .flat_map(|(idx, &hue)| {
                theme.role_hues.iter().skip(idx + 1).map(move |&other| hue_distance(hue, other))
            })
            .fold(0.0_f64, f64::max);
        let has_analogous_pair_with_accent =
            min_distance < 35.0 && max_distance >= 130.0 && warm_count >= 2;

        assert!(cool_count < 3, "{:?} produced an all-cool cluster", theme);
        assert!(warm_count > 0, "{:?} needs at least one warm anchor", theme);
        assert!(
            min_distance >= 35.0 || has_analogous_pair_with_accent,
            "{:?} roles are too tightly clustered without a far accent",
            theme
        );
        assert!(
            (0.70..=1.20).contains(&theme.drift_scale),
            "{:?} has unreasonable drift scale",
            theme
        );
        assert!(
            (20.0..=45.0).contains(&theme.wave_amplitude),
            "{:?} has unreasonable wave amplitude",
            theme
        );
        assert!(theme.quality_score() >= 0.78, "{:?} should pass the quality threshold", theme);
    }

    #[test]
    fn test_color_gradient_generation() {
        let mut rng = Sha3RandomByteStream::new(&[1, 2, 3, 4], 1.0, 1.0, 1.0, 1.0);
        let length = 100;
        let colors = generate_color_gradient_oklab(&mut rng, length, 0, BASE_HUE_DRIFT, false, 2.6);

        assert_eq!(colors.len(), length);
        for (l, a, b) in &colors {
            assert!(*l >= 0.0 && *l <= 1.0);
            assert!(*a >= -0.5 && *a <= 0.5);
            assert!(*b >= -0.5 && *b <= 0.5);
        }
    }

    #[test]
    fn test_color_gradient_chroma_boost() {
        let mut rng1 = Sha3RandomByteStream::new(&[1, 2, 3, 4], 1.0, 1.0, 1.0, 1.0);
        let mut rng2 = Sha3RandomByteStream::new(&[1, 2, 3, 4], 1.0, 1.0, 1.0, 1.0);

        let normal = generate_color_gradient_oklab(&mut rng1, 100, 0, BASE_HUE_DRIFT, false, 2.6);
        let boosted = generate_color_gradient_oklab(&mut rng2, 100, 0, BASE_HUE_DRIFT, true, 2.6);

        let avg_chroma = |cols: &[(f64, f64, f64)]| {
            cols.iter().map(|(_, a, b)| (a * a + b * b).sqrt()).sum::<f64>() / cols.len() as f64
        };
        assert!(
            avg_chroma(&boosted) > avg_chroma(&normal),
            "Boosted chroma should produce higher average saturation"
        );
    }

    #[test]
    fn test_body_color_sequences_uniform_alpha() {
        let mut rng = Sha3RandomByteStream::new(&[5, 6, 7, 8], 1.0, 1.0, 1.0, 1.0);
        let (colors, alphas) =
            generate_body_color_sequences(&mut rng, 50, 15_000_000, false, false);

        assert_eq!(colors.len(), 3);
        assert_eq!(alphas.len(), 3);
        for &a in &alphas {
            assert!(a > 0.0);
        }
        let unique: std::collections::HashSet<u64> = alphas.iter().map(|a| a.to_bits()).collect();
        assert!(unique.len() > 1, "creative profile should retain per-body depth hierarchy");
    }

    #[test]
    fn test_body_color_sequences_alpha_variation() {
        let mut rng = Sha3RandomByteStream::new(&[5, 6, 7, 8], 1.0, 1.0, 1.0, 1.0);
        let (_, alphas) = generate_body_color_sequences(&mut rng, 50, 15_000_000, false, true);

        assert_eq!(alphas.len(), 3);
        let unique: std::collections::HashSet<u64> = alphas.iter().map(|a| a.to_bits()).collect();
        assert!(unique.len() > 1, "alpha_variation should produce different per-body alphas");
    }

    #[test]
    fn test_color_generation_determinism() {
        let seed = [0x10, 0x00, 0x33];
        let steps = 200;

        let mut rng1 = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
        let (colors1, alphas1) =
            generate_body_color_sequences(&mut rng1, steps, 15_000_000, true, true);

        let mut rng2 = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
        let (colors2, alphas2) =
            generate_body_color_sequences(&mut rng2, steps, 15_000_000, true, true);

        for body in 0..3 {
            assert_eq!(
                alphas1[body].to_bits(),
                alphas2[body].to_bits(),
                "alpha for body {body} diverged"
            );
            for step in 0..steps {
                let (l1, a1, b1) = colors1[body][step];
                let (l2, a2, b2) = colors2[body][step];
                assert_eq!(l1.to_bits(), l2.to_bits(), "body {body} step {step} L diverged");
                assert_eq!(a1.to_bits(), a2.to_bits(), "body {body} step {step} a diverged");
                assert_eq!(b1.to_bits(), b2.to_bits(), "body {body} step {step} b diverged");
            }
        }
    }

    #[test]
    fn test_palette_theme_quality_rejects_all_cool_cluster() {
        let theme = PaletteTheme {
            kind: PaletteThemeKind::ArcticAurora,
            role_hues: [130.0, 180.0, 230.0],
            drift_scale: 0.8,
            wave_amplitude: 24.0,
            chroma_offsets: [0.0, 0.0, 0.0],
            lightness_offsets: [0.0, 0.0, 0.0],
        };

        assert!(
            theme.quality_score() < 0.78,
            "all green/cyan/blue palettes should fail the quality gate"
        );
    }

    #[test]
    fn test_palette_theme_choice_keeps_warm_or_balanced_roles() {
        for seed_val in 0u8..80 {
            let seed = [seed_val, 0x42, 0x91, 0x17];
            let mut rng = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
            let theme = PaletteTheme::choose(&mut rng);

            assert_theme_quality(theme);
        }
    }

    #[test]
    fn test_all_curated_palette_theme_templates_pass_quality_gate() {
        let kinds = [
            PaletteThemeKind::NoirGold,
            PaletteThemeKind::SolarFlare,
            PaletteThemeKind::AlienJewel,
            PaletteThemeKind::RoyalPlasma,
            PaletteThemeKind::ArcticAurora,
            PaletteThemeKind::RoseChrome,
            PaletteThemeKind::ToxicLuxury,
            PaletteThemeKind::PrismaticBlack,
        ];

        for (idx, kind) in kinds.iter().copied().enumerate() {
            let seed = [idx as u8, 0xBA, 0x5E, 0x11];
            let mut rng = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
            let theme = PaletteTheme::from_kind(kind, 142.0, &mut rng);

            assert_theme_quality(theme);
        }
    }

    #[test]
    fn test_generated_body_hues_follow_coherent_theme_roles() {
        for seed_val in 0u8..32 {
            let seed = [seed_val, 0x10, 0x20, 0x30];
            let mut rng = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
            let (colors, _) = generate_body_color_sequences(&mut rng, 160, 15_000_000, true, false);

            let mean_hues = [
                circular_mean_hue(&colors[0]),
                circular_mean_hue(&colors[1]),
                circular_mean_hue(&colors[2]),
            ];
            let cool_count = mean_hues.iter().filter(|&&hue| is_blue_green_cluster(hue)).count();
            let warm_count = mean_hues.iter().filter(|&&hue| is_warm_sector(hue)).count();
            let min_distance = mean_hues
                .iter()
                .enumerate()
                .flat_map(|(idx, &hue)| {
                    mean_hues.iter().skip(idx + 1).map(move |&other| hue_distance(hue, other))
                })
                .fold(HUE_FULL_CIRCLE, f64::min);
            let max_distance = mean_hues
                .iter()
                .enumerate()
                .flat_map(|(idx, &hue)| {
                    mean_hues.iter().skip(idx + 1).map(move |&other| hue_distance(hue, other))
                })
                .fold(0.0_f64, f64::max);
            let has_analogous_pair_with_accent =
                min_distance < 28.0 && max_distance >= 130.0 && warm_count >= 2;

            assert!(cool_count < 3, "seed {seed_val} produced all-cool body hues: {mean_hues:?}");
            assert!(warm_count > 0, "seed {seed_val} lost its warm body anchor: {mean_hues:?}");
            assert!(
                min_distance >= 28.0 || has_analogous_pair_with_accent,
                "seed {seed_val} body hues collapsed without a far accent: {mean_hues:?}"
            );
        }
    }

    #[test]
    fn test_palette_theme_fallback_is_quality_checked() {
        let seed = [0xFE, 0xED, 0xFA, 0xCE];
        let mut rng = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
        let theme = PaletteTheme::from_kind(PaletteThemeKind::NoirGold, 180.0, &mut rng);

        assert_theme_quality(theme);
    }

    #[test]
    fn test_orbit_art_metrics_respond_to_motion_and_closeness() {
        let slow_spread = OrbitArtMetrics::from_positions(&sample_metric_positions(0.08, false));
        let fast_spread = OrbitArtMetrics::from_positions(&sample_metric_positions(1.4, false));
        let fast_close = OrbitArtMetrics::from_positions(&sample_metric_positions(1.4, true));

        assert!(fast_spread.speed_energy > slow_spread.speed_energy);
        assert!(fast_close.closeness_energy > slow_spread.closeness_energy);
        assert!(fast_spread.depth_energy >= slow_spread.depth_energy);
    }

    #[test]
    fn test_creative_profile_selection_is_deterministic_for_same_orbit() {
        let positions = sample_metric_positions(1.2, false);
        let seed = [0x51, 0xAD, 0x10, 0x01];
        let mut rng_a = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
        let mut rng_b = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);

        let profile_a = CreativeProfile::choose(&mut rng_a, &positions);
        let profile_b = CreativeProfile::choose(&mut rng_b, &positions);

        assert_eq!(profile_a.art_direction, profile_b.art_direction);
        assert_eq!(profile_a.material_finish, profile_b.material_finish);
        assert_eq!(profile_a.composition_mode, profile_b.composition_mode);
        assert_eq!(profile_a.rare_event, profile_b.rare_event);
        assert_eq!(profile_a.gradient_map_palette, profile_b.gradient_map_palette);
        assert_eq!(profile_a.framing_zoom.to_bits(), profile_b.framing_zoom.to_bits());
        assert_eq!(profile_a.framing_offset.0.to_bits(), profile_b.framing_offset.0.to_bits());
        assert_eq!(profile_a.framing_offset.1.to_bits(), profile_b.framing_offset.1.to_bits());
    }

    #[test]
    fn test_creative_profiles_cover_many_art_directions() {
        let positions = sample_metric_positions(1.0, false);
        let mut seen = std::collections::HashSet::new();

        for seed_val in 0u16..256 {
            let seed = [(seed_val & 0xFF) as u8, (seed_val >> 8) as u8, 0x44, 0x99];
            let mut rng = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
            let profile = CreativeProfile::choose(&mut rng, &positions);
            seen.insert(profile.art_direction);
        }

        assert!(seen.len() >= 10, "creative profiles should explore many worlds: {seen:?}");
    }

    #[test]
    fn test_creative_profile_applies_effects_material_and_framing() {
        let profile = CreativeProfile {
            art_direction: ArtDirectionKind::RainbowOilFilm,
            material_finish: MaterialFinish::CrystalFacet,
            composition_mode: CompositionMode::MacroJewelry,
            rare_event: RareEvent::RainbowScar,
            palette: PaletteTheme {
                kind: PaletteThemeKind::PrismaticBlack,
                role_hues: [20.0, 170.0, 295.0],
                drift_scale: 1.0,
                wave_amplitude: 36.0,
                chroma_offsets: [0.05, 0.02, 0.06],
                lightness_offsets: [0.0, 0.02, 0.04],
            },
            gradient_map_palette: 11,
            hue_wave_freq: 3.4,
            alpha_scale: 1.1,
            body_alpha_multipliers: [1.0, 0.9, 1.2],
            framing_zoom: 1.22,
            framing_offset: (0.06, -0.04),
        };
        let mut config = ResolvedEffectConfig {
            width: 1920,
            height: 1080,
            gradient_map_strength: 0.30,
            gradient_map_hue_preservation: 0.55,
            color_grade_strength: 0.55,
            vibrance: 1.15,
            clarity_strength: 0.40,
            tone_curve_strength: 0.55,
            vignette_strength: 0.35,
            hdr_scale: 0.12,
            chromatic_bloom_strength: 0.20,
            chromatic_bloom_separation_scale: 0.0005,
            ..Default::default()
        };

        profile.apply_to_effect_config(&mut config);

        assert!(config.enable_gradient_map);
        assert!(config.enable_color_grade);
        assert!(config.enable_opalescence);
        assert!(config.enable_bloom);
        assert!(config.enable_chromatic_bloom);
        assert_eq!(config.gradient_map_palette, 11);
        assert_eq!(config.composition_zoom, 1.22);
        assert_eq!(config.composition_offset_x, 0.06);
        assert_eq!(config.composition_offset_y, -0.04);
        assert!(config.chromatic_bloom_strength >= 0.28);
        assert!(config.vibrance > 1.15);
    }
}
