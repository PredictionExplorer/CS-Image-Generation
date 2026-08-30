//! Public NFT trait metadata: `metadata/nft_traits.json`.
//!
//! This module turns the seed-resolved visual profile, the winning orbit, and
//! a set of deterministic physics analyses into the machine-readable trait
//! file consumed by the metadata server (see
//! `docs/augur-explorer-integration.md`). Everything in the file is a pure
//! function of the on-chain seed: anyone can re-run the pipeline and verify
//! every value byte-for-byte.
//!
//! Marketplace-facing attributes are emitted **ready to serve** (final
//! `trait_type` / `value` objects) so downstream consumers never re-derive or
//! re-name art facts. Raw measurements live in the `simulation` and
//! `generation` blocks.

use crate::app::AestheticSelection;
use crate::drift_config::ResolvedDriftConfig;
use crate::error::Result;
use crate::render::visual_profile::{
    CosmicSignatureParameters, ProjectionMode, StructureMode, SymmetryOp,
};
use crate::sim::{self, shift_bodies_to_com};
use crate::traits_analysis::{
    BraidSummary, Fate, chaos_coefficient_of_variation, chaos_index_from_cv, classify_fate,
    closest_approach, compute_braid, detect_syzygies,
};
use crate::{analysis, oklab, render, spectrum};
use serde::Serialize;
use serde_json::{Value, json};
use std::fs::File;
use std::io::BufWriter;
use std::sync::LazyLock;
use tracing::{info, warn};

/// Semantic version of the `nft_traits.json` schema. Consumers must check the
/// major component before trusting the file layout.
pub const SCHEMA_VERSION: &str = "1.0.0";

/// Extended-simulation factor for fate classification: the replay runs this
/// many recorded windows past the rendered one. Frozen; changing it would
/// change published `Fate` values.
pub const FATE_EXTENSION_FACTOR: usize = 8;

/// One marketplace-facing attribute, following the `OpenSea` metadata
/// conventions (`display_type` / `max_value` for numeric traits).
#[derive(Clone, Debug, Serialize)]
pub struct TraitAttribute {
    /// Attribute category shown in trait filters.
    pub trait_type: String,
    /// Optional display hint (`"number"`, `"date"`, ...).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_type: Option<String>,
    /// Attribute value (string or number).
    pub value: Value,
    /// Optional maximum for numeric traits (renders as `value / max`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_value: Option<u32>,
}

impl TraitAttribute {
    fn string(trait_type: &str, value: impl Into<String>) -> Self {
        Self {
            trait_type: trait_type.to_string(),
            display_type: None,
            value: Value::String(value.into()),
            max_value: None,
        }
    }

    fn number(trait_type: &str, value: u64, max_value: Option<u32>) -> Self {
        Self {
            trait_type: trait_type.to_string(),
            display_type: Some("number".to_string()),
            value: json!(value),
            max_value,
        }
    }
}

/// Global closest pairwise approach, serialized.
#[derive(Clone, Debug, Serialize)]
pub struct ClosestApproachBlock {
    /// Smallest pairwise separation in simulation units.
    pub distance: f64,
    /// The approaching body pair.
    pub pair: [usize; 2],
    /// Simulation step of the minimum.
    pub step: usize,
}

/// Braid-word summary, serialized.
#[derive(Clone, Debug, Serialize)]
pub struct BraidBlock {
    /// Artin word (possibly truncated), e.g. `"s1 s2' s1"`.
    pub word: String,
    /// Total crossings detected before truncation.
    pub crossings: usize,
    /// True when `word` was truncated.
    pub truncated: bool,
}

/// Long-term fate classification, serialized.
#[derive(Clone, Debug, Serialize)]
pub struct FateBlock {
    /// Outcome label: `"eternal_dance"` or `"ejection"`.
    pub outcome: String,
    /// Escaping body index, when the outcome is an ejection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub escaper: Option<usize>,
    /// Extended-window step at which the ejection confirmed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ejection_step: Option<usize>,
    /// Extended steps checked beyond the warm-up window.
    pub horizon_steps: usize,
}

/// Raw physics measurements of the winning orbit.
#[derive(Clone, Debug, Serialize)]
pub struct SimulationBlock {
    /// Body masses of the winning candidate.
    pub masses: [f64; 3],
    /// Total energy of the initial state (centre-of-mass frame).
    pub total_energy: f64,
    /// Magnitude of the total angular momentum (centre-of-mass frame).
    pub angular_momentum: f64,
    /// Production non-chaoticness score (higher = more regular).
    pub chaos_raw: f64,
    /// Scale-invariant chaos coefficient of variation.
    pub chaos_cv: f64,
    /// Bucketed 0-100 chaos index published as the `Chaos` attribute.
    pub chaos_index: u32,
    /// Equilateralness score in `[0, 1]`.
    pub equilateralness: f64,
    /// Global closest pairwise approach.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub closest_approach: Option<ClosestApproachBlock>,
    /// Number of three-body alignments in the rendered window.
    pub syzygy_count: usize,
    /// Braid word of the trajectory's crossing topology.
    pub braid: BraidBlock,
    /// Long-term fate classification.
    pub fate: FateBlock,
    /// Integrator identifier.
    pub integrator: String,
    /// Integration timestep.
    pub dt: f64,
    /// Recorded steps (equals the rendered window).
    pub steps: usize,
    /// Warm-up steps before recording.
    pub warmup_steps: usize,
    /// Escape-energy threshold used by fate classification.
    pub escape_threshold: f64,
}

/// One composed structure layer, serialized.
#[derive(Clone, Debug, Serialize)]
pub struct LayerBlock {
    /// Human-readable vocabulary name, e.g. `"Nebula Veil"`.
    pub vocabulary: String,
    /// Energy multiplier relative to the primary layer.
    pub alpha: f64,
}

/// Seed-resolved layer stack, serialized.
#[derive(Clone, Debug, Serialize)]
pub struct StructureBlock {
    /// Primary vocabulary display name.
    pub primary: String,
    /// Optional underlay layer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub underlay: Option<LayerBlock>,
    /// Optional accent layer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accent: Option<LayerBlock>,
    /// Stable machine label of the chosen stack.
    pub stack_label: String,
    /// Stable machine label of the seed's originally rolled stack.
    pub preferred_stack_label: String,
}

/// Rare seed-gated finishing effects, serialized.
#[derive(Clone, Debug, Serialize)]
pub struct FinishBlock {
    /// True when the prism (chromatic bloom) trait is active.
    pub prism: bool,
    /// Prism strength (0 disables).
    pub prism_strength: f64,
    /// True when diffraction spikes are active.
    pub diffraction_spikes: bool,
    /// Spike arm count, when active.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spike_arms: Option<u8>,
    /// True when the stardust field is active.
    pub stardust: bool,
    /// Stardust micro-dot count.
    pub stardust_count: u32,
    /// Halation strength (a common, non-rare finish; 0 disables).
    pub halation_strength: f64,
}

/// Palette genome summary, serialized.
#[derive(Clone, Debug, Serialize)]
pub struct PaletteBlock {
    /// Continuous genome fingerprint.
    pub fingerprint: String,
    /// Beauty-gate descriptor.
    pub gate: String,
    /// Bucketed family name published as the `Palette` attribute.
    pub family: String,
    /// Genome hue anchor in degrees.
    pub anchor_deg: f64,
    /// Genome hue dispersion in degrees.
    pub dispersion_deg: f64,
    /// Per-body base hues in `OKLCh` degrees.
    pub body_base_hues_deg: [f64; 3],
    /// Index of the most chromatic body.
    pub dominant_body: usize,
    /// Stellar class published as the `Spectral Class` attribute.
    pub spectral_class: String,
    /// Dominant wavelength (nm) backing the spectral class.
    pub dominant_wavelength_nm: f64,
}

/// Borda-selection context, serialized.
#[derive(Clone, Debug, Serialize)]
pub struct BordaBlock {
    /// Borda weight favouring chaotic orbits.
    pub chaos_weight: f64,
    /// Borda weight favouring equilateral orbits.
    pub equil_weight: f64,
    /// True when the weights were sampled rather than user-specified.
    pub weights_randomized: bool,
    /// Index of the winning candidate in its search.
    pub selected_index: usize,
    /// Total candidates evaluated across retries.
    pub total_candidates: usize,
    /// Bounded aesthetic retry searches used.
    pub retry_count: usize,
    /// Proxy aesthetic score of the accepted output.
    pub aesthetic_score: f64,
    /// Aesthetic score plus the stack-identity prior.
    pub selection_score: f64,
}

/// Camera drift parameters, serialized.
#[derive(Clone, Debug, Serialize)]
pub struct DriftBlock {
    /// Whether drift was applied.
    pub enabled: bool,
    /// Drift path style.
    pub mode: String,
    /// Drift magnitude multiplier.
    pub scale: f64,
    /// Fraction of one orbit swept.
    pub arc_fraction: f64,
    /// Drift ellipse eccentricity.
    pub orbit_eccentricity: f64,
    /// True when values were randomly generated.
    pub randomized: bool,
}

/// Output resolution, serialized.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ResolutionBlock {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
}

/// Seed-resolved generation context (the reproducibility certificate).
#[derive(Clone, Debug, Serialize)]
pub struct GenerationBlock {
    /// Visual profile identifier.
    pub visual_profile: String,
    /// Composed layer stack.
    pub structure: StructureBlock,
    /// Projection label (`"position"`, `"phase_portrait"`, ...).
    pub projection: String,
    /// Symmetry label (`"none"`, `"mirror_x"`, `"rot4"`, `"dih6"`).
    pub symmetry: String,
    /// True when the seed sampled extended wildcard ranges.
    pub wildcard: bool,
    /// Rare finishing effects.
    pub finishes: FinishBlock,
    /// Palette genome summary.
    pub palette: PaletteBlock,
    /// Borda-selection context.
    pub borda: BordaBlock,
    /// Camera drift parameters.
    pub drift: DriftBlock,
    /// Output resolution.
    pub resolution: ResolutionBlock,
}

/// Complete `metadata/nft_traits.json` payload.
#[derive(Clone, Debug, Serialize)]
pub struct NftTraitsFile {
    /// Schema version (semver); consumers must check the major component.
    pub schema_version: String,
    /// Canonical `0x`-prefixed lowercase seed.
    pub seed: String,
    /// Generator crate version that produced this file.
    pub pipeline_version: String,
    /// RFC 3339 timestamp of generation (informational, not deterministic).
    pub generated_at: String,
    /// Ready-to-serve marketplace attributes.
    pub attributes: Vec<TraitAttribute>,
    /// Art-derived description sentence(s); the metadata server appends
    /// provenance (round, imprint) from chain data.
    pub description_art: String,
    /// Raw physics measurements.
    pub simulation: SimulationBlock,
    /// Seed-resolved generation context.
    pub generation: GenerationBlock,
}

/// Everything the trait builder needs from the generation pipeline.
pub struct NftTraitsInputs<'a> {
    /// Hex seed without the `0x` prefix.
    pub seed_hex: &'a str,
    /// Seed-resolved visual parameters (after adaptive stack retargeting).
    pub parameters: &'a CosmicSignatureParameters,
    /// Winning orbit of the Borda + aesthetic selection.
    pub selection: &'a AestheticSelection,
    /// Resolved camera drift, when enabled.
    pub drift: Option<&'a ResolvedDriftConfig>,
    /// Drift mode label from the CLI.
    pub drift_mode: &'a str,
    /// Orbits evaluated per Borda search.
    pub num_sims: usize,
    /// Recorded simulation steps.
    pub num_steps: usize,
    /// Borda chaos weight.
    pub chaos_weight: f64,
    /// Borda equilateralness weight.
    pub equil_weight: f64,
    /// True when Borda weights were sampled.
    pub weights_randomized: bool,
    /// Escape-energy threshold for fate classification.
    pub escape_threshold: f64,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
}

/// Marketplace display name for a stroke vocabulary.
#[must_use]
pub fn vocabulary_display(mode: StructureMode) -> &'static str {
    match mode {
        StructureMode::TriangleWeb => "Triangle Web",
        StructureMode::OrbitRibbons => "Orbit Ribbons",
        StructureMode::Duet { .. } => "Duet",
        StructureMode::Spokes => "Spokes",
        StructureMode::TimeChords => "Time Chords",
        StructureMode::NebulaVeil => "Nebula Veil",
        StructureMode::HarmonicWeave => "Harmonic Weave",
        StructureMode::StippleConstellation => "Stipple Constellation",
        StructureMode::TangentCaustics => "Tangent Caustics",
    }
}

/// Marketplace display value for a symmetry operation (`None` when the seed
/// has no symmetry, so the attribute is omitted).
#[must_use]
pub fn symmetry_display(op: SymmetryOp) -> Option<String> {
    match op {
        SymmetryOp::None => None,
        SymmetryOp::MirrorX => Some("Mirror".to_string()),
        SymmetryOp::Rotational { k } => Some(format!("Mandala \u{d7}{k}")),
        SymmetryOp::Dihedral { k } => Some(format!("Rosette \u{d7}{k}")),
    }
}

/// Marketplace display value for a projection (`None` for plain position
/// space, so the attribute is omitted).
#[must_use]
pub fn projection_display(mode: ProjectionMode) -> Option<&'static str> {
    match mode {
        ProjectionMode::Position => None,
        ProjectionMode::PhasePortrait => Some("Phase Portrait"),
        ProjectionMode::CrossBraid => Some("Cross Braid"),
        ProjectionMode::Hodograph => Some("Hodograph"),
    }
}

/// Bucket the three body masses into a marketplace-facing balance class.
///
/// Thresholds are frozen: a max/min ratio at or below 1.25 is an
/// `"Equal Trio"`; otherwise the larger relative gap decides between a
/// dominant single body (`"Heavy Primary"`) and a close heavy pair above a
/// light third (`"Twin Binary"`).
#[must_use]
pub fn mass_balance(masses: [f64; 3]) -> &'static str {
    let mut sorted = masses;
    sorted.sort_by(f64::total_cmp);
    let [low, mid, high] = sorted;
    if high / low <= 1.25 {
        "Equal Trio"
    } else if high / mid >= mid / low {
        "Heavy Primary"
    } else {
        "Twin Binary"
    }
}

/// `(OKLab hue degrees, wavelength nm)` samples of the rendered spectral
/// locus, used to invert a hue back to its dominant wavelength.
static SPECTRAL_LOCUS: LazyLock<Vec<(f64, f64)>> = LazyLock::new(|| {
    (0..=160)
        .map(|i| {
            let nm = 380.0 + f64::from(i) * 2.0;
            let (r, g, b) = spectrum::wavelength_to_rgb(nm);
            let (_, a, lab_b) = oklab::linear_srgb_to_oklab(r, g, b);
            (lab_b.atan2(a).to_degrees().rem_euclid(360.0), nm)
        })
        .collect()
});

fn circular_hue_distance(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(360.0);
    d.min(360.0 - d)
}

/// Dominant wavelength (nm) whose rendered hue is nearest to `hue_deg`.
/// Non-spectral (magenta) hues resolve to the nearest locus end.
#[must_use]
pub fn dominant_wavelength_for_hue(hue_deg: f64) -> f64 {
    let hue = hue_deg.rem_euclid(360.0);
    SPECTRAL_LOCUS
        .iter()
        .copied()
        .min_by(|(h1, _), (h2, _)| {
            circular_hue_distance(hue, *h1).total_cmp(&circular_hue_distance(hue, *h2))
        })
        .map_or(550.0, |(_, nm)| nm)
}

/// Stellar spectral class for a dominant wavelength. The bands are an
/// artistic mapping of the visible spectrum onto the Morgan-Keenan
/// temperature sequence (O hottest / violet through M coolest / red) and are
/// frozen.
#[must_use]
pub fn spectral_class_for_wavelength(nm: f64) -> &'static str {
    if nm < 455.0 {
        "O"
    } else if nm < 485.0 {
        "B"
    } else if nm < 510.0 {
        "A"
    } else if nm < 565.0 {
        "F"
    } else if nm < 590.0 {
        "G"
    } else if nm < 625.0 {
        "K"
    } else {
        "M"
    }
}

fn structure_phrase(mode: StructureMode) -> &'static str {
    match mode {
        StructureMode::TriangleWeb => "a triangle web",
        StructureMode::OrbitRibbons => "orbit ribbons",
        StructureMode::Duet { .. } => "a duet of edges",
        StructureMode::Spokes => "centroid spokes",
        StructureMode::TimeChords => "time chords",
        StructureMode::NebulaVeil => "a nebula veil",
        StructureMode::HarmonicWeave => "a harmonic weave",
        StructureMode::StippleConstellation => "a stipple constellation",
        StructureMode::TangentCaustics => "tangent caustics",
    }
}

fn symmetry_phrase(op: SymmetryOp) -> Option<String> {
    match op {
        SymmetryOp::None => None,
        SymmetryOp::MirrorX => Some(", mirrored across the frame".to_string()),
        SymmetryOp::Rotational { k } => Some(format!(", folded into a {k}-fold mandala")),
        SymmetryOp::Dihedral { k } => Some(format!(", folded into a {k}-fold rosette")),
    }
}

fn projection_phrase(mode: ProjectionMode) -> Option<&'static str> {
    match mode {
        ProjectionMode::Position => None,
        ProjectionMode::PhasePortrait => Some(", plotted in phase-portrait space"),
        ProjectionMode::CrossBraid => Some(", plotted in cross-braid space"),
        ProjectionMode::Hodograph => Some(", plotted in hodograph space"),
    }
}

fn active_finishes(parameters: &CosmicSignatureParameters) -> Vec<&'static str> {
    let mut finishes = Vec::new();
    if parameters.prism_strength > 0.0 {
        finishes.push("Prism");
    }
    if parameters.spikes.enabled() {
        finishes.push("Diffraction Spikes");
    }
    if parameters.stardust.enabled() {
        finishes.push("Stardust");
    }
    finishes
}

fn finish_phrase(finishes: &[&'static str]) -> Option<String> {
    if finishes.is_empty() {
        return None;
    }
    let names: Vec<String> = finishes
        .iter()
        .map(|name| match *name {
            "Prism" => "prism dispersion".to_string(),
            other => other.to_lowercase(),
        })
        .collect();
    Some(format!(", crowned with {}", names.join(" and ")))
}

#[allow(clippy::too_many_arguments)]
fn build_description(
    parameters: &CosmicSignatureParameters,
    family: &str,
    chaos_index: u32,
    syzygy_count: usize,
    spectral_class: &str,
    fate: Fate,
) -> String {
    use std::fmt::Write as _;

    let stack = parameters.stack;
    let mut sentence =
        format!("Three bodies trace {} in {family} tones", structure_phrase(stack.primary));
    if let Some(underlay) = stack.underlay {
        let name = vocabulary_display(underlay.vocabulary).to_lowercase();
        let _ = write!(sentence, " over a {name} underlay");
    }
    if let Some(symmetry) = symmetry_phrase(parameters.symmetry) {
        sentence.push_str(&symmetry);
    }
    if let Some(projection) = projection_phrase(parameters.projection) {
        sentence.push_str(projection);
    }
    if let Some(finish) = finish_phrase(&active_finishes(parameters)) {
        sentence.push_str(&finish);
    }
    let fate_text = match fate {
        Fate::EternalDance => "eternal dance",
        Fate::Ejection => "ejection",
    };
    let _ = write!(
        sentence,
        ". Chaos {chaos_index}/100 across {syzygy_count} syzygies; spectral class \
         {spectral_class}; fate: {fate_text}."
    );
    sentence
}

/// Run the deterministic analyses and assemble the full trait file.
///
/// Re-simulates the winning candidate in raw physics space (a bit-identical
/// replay of the production recording) so syzygies, closest approach, braid
/// word, and chaos are measured on the physical orbit rather than the
/// projected or drifted render trajectory.
#[must_use]
pub fn build(inputs: &NftTraitsInputs<'_>) -> NftTraitsFile {
    let parameters = inputs.parameters;
    let selection = inputs.selection;

    info!("STAGE 8: NFT trait analysis (raw physics replay of the winning orbit)...");
    let raw_positions = sim::get_positions(selection.bodies.clone(), inputs.num_steps).positions;

    let masses = [selection.bodies[0].mass, selection.bodies[1].mass, selection.bodies[2].mass];
    let mut com_bodies = selection.bodies.clone();
    shift_bodies_to_com(&mut com_bodies);
    let total_energy = analysis::calculate_total_energy(&com_bodies);
    let angular_momentum = analysis::calculate_total_angular_momentum(&com_bodies).norm();

    let syzygies = detect_syzygies(&raw_positions);
    let approach = closest_approach(&raw_positions);
    let braid = compute_braid(&raw_positions);
    let chaos_cv = chaos_coefficient_of_variation(masses, &raw_positions);
    let chaos_index = chaos_index_from_cv(chaos_cv);
    drop(raw_positions);

    info!(
        "   => syzygies={} braid_crossings={} chaos_cv={:.3} chaos_index={}",
        syzygies.len(),
        braid.crossings,
        chaos_cv,
        chaos_index,
    );
    info!("   => Fate classification over {}x extended window...", FATE_EXTENSION_FACTOR);
    let fate = classify_fate(
        &selection.bodies,
        inputs.num_steps,
        FATE_EXTENSION_FACTOR,
        inputs.escape_threshold,
    );
    info!("   => fate={} escaper={:?}", fate.fate.label(), fate.escaper);

    let palette = render::color::current_palette_details().unwrap_or_else(|| {
        warn!("Palette details unavailable; emitting unresolved placeholders");
        render::color::PaletteDetails {
            fingerprint: "unresolved".to_string(),
            gate: "unresolved".to_string(),
            family: "Unresolved".to_string(),
            anchor_deg: 0.0,
            dispersion_deg: 0.0,
            body_base_hues_deg: [0.0; 3],
            dominant_body: 0,
        }
    });
    let dominant_hue = palette.body_base_hues_deg[palette.dominant_body];
    let dominant_wavelength_nm = dominant_wavelength_for_hue(dominant_hue);
    let spectral_class = spectral_class_for_wavelength(dominant_wavelength_nm);

    let mut attributes =
        vec![TraitAttribute::string("Structure", vocabulary_display(parameters.stack.primary))];
    if let Some(underlay) = parameters.stack.underlay {
        attributes
            .push(TraitAttribute::string("Underlay", vocabulary_display(underlay.vocabulary)));
    }
    if let Some(accent) = parameters.stack.accent {
        attributes.push(TraitAttribute::string("Accent", vocabulary_display(accent.vocabulary)));
    }
    if let Some(symmetry) = symmetry_display(parameters.symmetry) {
        attributes.push(TraitAttribute::string("Symmetry", symmetry));
    }
    if let Some(projection) = projection_display(parameters.projection) {
        attributes.push(TraitAttribute::string("Projection", projection));
    }
    for finish in active_finishes(parameters) {
        attributes.push(TraitAttribute::string("Finish", finish));
    }
    if parameters.wildcard {
        attributes.push(TraitAttribute::string("Wildcard", "Yes"));
    }
    attributes.push(TraitAttribute::string("Palette", palette.family.clone()));
    attributes.push(TraitAttribute::string("Spectral Class", spectral_class));
    attributes.push(TraitAttribute::string("Mass Balance", mass_balance(masses)));
    attributes.push(TraitAttribute::string(
        "Fate",
        match fate.fate {
            Fate::EternalDance => "Eternal Dance",
            Fate::Ejection => "Ejection",
        },
    ));
    attributes.push(TraitAttribute::number("Chaos", u64::from(chaos_index), Some(100)));
    attributes.push(TraitAttribute::number("Syzygies", syzygies.len() as u64, None));

    let description_art = build_description(
        parameters,
        &palette.family,
        chaos_index,
        syzygies.len(),
        spectral_class,
        fate.fate,
    );

    let BraidSummary { word, crossings, truncated } = braid;
    NftTraitsFile {
        schema_version: SCHEMA_VERSION.to_string(),
        seed: format!("0x{}", inputs.seed_hex.to_lowercase()),
        pipeline_version: env!("CARGO_PKG_VERSION").to_string(),
        generated_at: chrono::Local::now().to_rfc3339(),
        attributes,
        description_art,
        simulation: SimulationBlock {
            masses,
            total_energy,
            angular_momentum,
            chaos_raw: selection.result.chaos,
            chaos_cv,
            chaos_index,
            equilateralness: selection.result.equilateralness,
            closest_approach: approach.map(|a| ClosestApproachBlock {
                distance: a.distance,
                pair: [a.pair.0, a.pair.1],
                step: a.step,
            }),
            syzygy_count: syzygies.len(),
            braid: BraidBlock { word, crossings, truncated },
            fate: FateBlock {
                outcome: fate.fate.label().to_string(),
                escaper: fate.escaper,
                ejection_step: fate.ejection_step,
                horizon_steps: fate.horizon_steps,
            },
            integrator: "yoshida4".to_string(),
            dt: render::constants::DEFAULT_DT,
            steps: inputs.num_steps,
            warmup_steps: inputs.num_steps,
            escape_threshold: inputs.escape_threshold,
        },
        generation: GenerationBlock {
            visual_profile: render::visual_profile::COSMIC_SIGNATURE_PROFILE_NAME.to_string(),
            structure: StructureBlock {
                primary: vocabulary_display(parameters.stack.primary).to_string(),
                underlay: parameters.stack.underlay.map(|layer| LayerBlock {
                    vocabulary: vocabulary_display(layer.vocabulary).to_string(),
                    alpha: layer.alpha,
                }),
                accent: parameters.stack.accent.map(|layer| LayerBlock {
                    vocabulary: vocabulary_display(layer.vocabulary).to_string(),
                    alpha: layer.alpha,
                }),
                stack_label: parameters.stack.label(),
                preferred_stack_label: selection.preferred_stack.label(),
            },
            projection: parameters.projection.label().to_string(),
            symmetry: parameters.symmetry.label(),
            wildcard: parameters.wildcard,
            finishes: FinishBlock {
                prism: parameters.prism_strength > 0.0,
                prism_strength: parameters.prism_strength,
                diffraction_spikes: parameters.spikes.enabled(),
                spike_arms: parameters.spikes.enabled().then_some(parameters.spikes.arms),
                stardust: parameters.stardust.enabled(),
                stardust_count: parameters.stardust.count,
                halation_strength: parameters.halation_strength,
            },
            palette: PaletteBlock {
                fingerprint: palette.fingerprint,
                gate: palette.gate,
                family: palette.family,
                anchor_deg: palette.anchor_deg,
                dispersion_deg: palette.dispersion_deg,
                body_base_hues_deg: palette.body_base_hues_deg,
                dominant_body: palette.dominant_body,
                spectral_class: spectral_class.to_string(),
                dominant_wavelength_nm,
            },
            borda: BordaBlock {
                chaos_weight: inputs.chaos_weight,
                equil_weight: inputs.equil_weight,
                weights_randomized: inputs.weights_randomized,
                selected_index: selection.result.selected_index,
                total_candidates: inputs.num_sims * (selection.retry_count + 1),
                retry_count: selection.retry_count,
                aesthetic_score: selection.aesthetic.total,
                selection_score: selection.selection_score,
            },
            drift: inputs.drift.map_or(
                DriftBlock {
                    enabled: false,
                    mode: "none".to_string(),
                    scale: 0.0,
                    arc_fraction: 0.0,
                    orbit_eccentricity: 0.0,
                    randomized: false,
                },
                |drift| DriftBlock {
                    enabled: true,
                    mode: inputs.drift_mode.to_string(),
                    scale: drift.scale,
                    arc_fraction: drift.arc_fraction,
                    orbit_eccentricity: drift.orbit_eccentricity,
                    randomized: drift.was_randomized,
                },
            ),
            resolution: ResolutionBlock { width: inputs.width, height: inputs.height },
        },
    }
}

/// Serialize a trait file to `{seed_dir}/metadata/nft_traits.json`.
pub fn write(seed_dir: &str, file: &NftTraitsFile) -> Result<()> {
    let path = format!("{seed_dir}/metadata/nft_traits.json");
    let out = File::create(&path)?;
    serde_json::to_writer_pretty(BufWriter::new(out), file).map_err(std::io::Error::other)?;
    info!("   Saved NFT trait metadata => {path}");
    Ok(())
}

/// Build and write the trait file in one call (the pipeline entry point).
pub fn compute_and_write(seed_dir: &str, inputs: &NftTraitsInputs<'_>) -> Result<()> {
    let file = build(inputs);
    write(seed_dir, &file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocabulary_display_covers_every_mode() {
        let modes = [
            StructureMode::TriangleWeb,
            StructureMode::OrbitRibbons,
            StructureMode::Duet { dropped_edge: 1 },
            StructureMode::Spokes,
            StructureMode::TimeChords,
            StructureMode::NebulaVeil,
            StructureMode::HarmonicWeave,
            StructureMode::StippleConstellation,
            StructureMode::TangentCaustics,
        ];
        for mode in modes {
            assert!(!vocabulary_display(mode).is_empty());
        }
        assert_eq!(vocabulary_display(StructureMode::HarmonicWeave), "Harmonic Weave");
    }

    #[test]
    fn symmetry_display_matches_the_marketplace_vocabulary() {
        assert_eq!(symmetry_display(SymmetryOp::None), None);
        assert_eq!(symmetry_display(SymmetryOp::MirrorX).as_deref(), Some("Mirror"));
        assert_eq!(
            symmetry_display(SymmetryOp::Rotational { k: 4 }).as_deref(),
            Some("Mandala \u{d7}4")
        );
        assert_eq!(
            symmetry_display(SymmetryOp::Dihedral { k: 6 }).as_deref(),
            Some("Rosette \u{d7}6")
        );
    }

    #[test]
    fn projection_display_omits_position_space() {
        assert_eq!(projection_display(ProjectionMode::Position), None);
        assert_eq!(projection_display(ProjectionMode::PhasePortrait), Some("Phase Portrait"));
        assert_eq!(projection_display(ProjectionMode::CrossBraid), Some("Cross Braid"));
        assert_eq!(projection_display(ProjectionMode::Hodograph), Some("Hodograph"));
    }

    #[test]
    fn mass_balance_buckets_are_stable() {
        assert_eq!(mass_balance([200.0, 210.0, 220.0]), "Equal Trio");
        assert_eq!(mass_balance([100.0, 110.0, 290.0]), "Heavy Primary");
        assert_eq!(mass_balance([100.0, 280.0, 290.0]), "Twin Binary");
        // Order independence.
        assert_eq!(mass_balance([290.0, 100.0, 280.0]), "Twin Binary");
    }

    #[test]
    fn wavelength_inversion_roundtrips_within_tolerance() {
        for nm in [420.0, 470.0, 500.0, 550.0, 580.0, 610.0, 650.0] {
            let (r, g, b) = spectrum::wavelength_to_rgb(nm);
            let (_, a, lab_b) = oklab::linear_srgb_to_oklab(r, g, b);
            let hue = lab_b.atan2(a).to_degrees().rem_euclid(360.0);
            let recovered = dominant_wavelength_for_hue(hue);
            assert!(
                (recovered - nm).abs() <= 6.0,
                "wavelength {nm} nm recovered as {recovered} nm"
            );
        }
    }

    #[test]
    fn spectral_class_bands_follow_the_temperature_sequence() {
        assert_eq!(spectral_class_for_wavelength(430.0), "O");
        assert_eq!(spectral_class_for_wavelength(470.0), "B");
        assert_eq!(spectral_class_for_wavelength(500.0), "A");
        assert_eq!(spectral_class_for_wavelength(540.0), "F");
        assert_eq!(spectral_class_for_wavelength(575.0), "G");
        assert_eq!(spectral_class_for_wavelength(600.0), "K");
        assert_eq!(spectral_class_for_wavelength(660.0), "M");
    }

    #[test]
    fn trait_attribute_serialization_matches_marketplace_conventions() {
        let text = serde_json::to_string(&TraitAttribute::string("Structure", "Harmonic Weave"))
            .expect("serializable");
        assert_eq!(text, r#"{"trait_type":"Structure","value":"Harmonic Weave"}"#);

        let numeric = serde_json::to_string(&TraitAttribute::number("Chaos", 62, Some(100)))
            .expect("serializable");
        assert_eq!(
            numeric,
            r#"{"trait_type":"Chaos","display_type":"number","value":62,"max_value":100}"#
        );
    }
}
