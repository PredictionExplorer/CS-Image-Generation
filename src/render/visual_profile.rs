//! Visual profiles for the renderer.
//!
//! The production generator uses a single crisp `CosmicSignature` profile: no
//! randomized post-effect stack, a clean black field, saturated spectral colour,
//! and restrained HDR values that preserve thin luminous structure.
//!
//! Within that family identity each seed resolves a **layer stack** — a primary
//! stroke vocabulary plus optional underlay and accent layers at continuous
//! alphas — together with an orthogonal projection axis (position space or one
//! of several phase-space projections), a symmetry operation (none, mirror,
//! k-fold rotational, dihedral), per-vocabulary continuous tuning, and a small
//! set of rare seed-gated finishing traits (halation, prism dispersion,
//! diffraction spikes, stardust). A ~7% wildcard gate widens parameter ranges
//! for a minority of seeds; the aesthetic quality floor in `app` keeps those
//! extremes beautiful.
//!
//! Design principle: *sample wild, gate hard*. Every axis is continuous and
//! seed-derived; beauty is enforced by deterministic quality gates rather than
//! narrow presets.

use super::effect_randomizer::{RandomizationLog, RandomizationRecord};
use super::randomizable_config::ResolvedEffectConfig;
use crate::sim::Sha3RandomByteStream;
use smallvec::SmallVec;

/// Canonical name recorded in generation metadata for the default visual style.
pub const COSMIC_SIGNATURE_PROFILE_NAME: &str = "cosmic_signature";

/// Probability that a seed receives the rare prism (chromatic bloom) trait.
pub const PRISM_PROBABILITY: f64 = 0.05;

/// Probability that an `OrbitRibbons` seed layers a time-lagged echo band.
pub const RIBBON_ECHO_PROBABILITY: f64 = 0.60;

/// Probability that a seed composes a second (underlay) vocabulary layer.
pub const UNDERLAY_PROBABILITY: f64 = 0.40;

/// Probability that a seed adds a rare third (accent) vocabulary layer.
pub const ACCENT_PROBABILITY: f64 = 0.10;

/// Probability that a seed samples from extended "wildcard" parameter ranges.
pub const WILDCARD_PROBABILITY: f64 = 0.07;

/// Probability of the mirrored composition trait (within the symmetry table).
pub const MIRROR_PROBABILITY: f64 = 0.03;

/// Probability of a k-fold rotational (mandala) composition.
pub const ROTATIONAL_PROBABILITY: f64 = 0.07;

/// Probability of a dihedral (rosette) composition.
pub const DIHEDRAL_PROBABILITY: f64 = 0.05;

/// Probability that a seed renders a phase-space projection instead of position space.
pub const PHASE_PROJECTION_PROBABILITY: f64 = 0.08;

/// Probability of the rare diffraction-spike (astrophoto star cross) trait.
pub const SPIKE_PROBABILITY: f64 = 0.04;

/// Probability of the faint stardust background field.
pub const STARDUST_PROBABILITY: f64 = 0.10;

/// Probability that a seed receives the calligraphic width pulse: a slow
/// sinusoid along the timeline that swells and tapers stroke width (and, out
/// of phase, brightness) independently of orbital velocity.
pub const WIDTH_PULSE_PROBABILITY: f64 = 0.60;

/// RNG fork domain for layer-stack / trait sampling (keeps the legacy main
/// stream consumption byte-for-byte aligned).
const STRUCTURE_RNG_DOMAIN: &[u8] = b"cosmic-structure/v2";

/// Atomic stroke vocabulary: how three-body geometry becomes luminous marks.
///
/// Selected per seed as the primary layer of a [`LayerStack`]; every
/// vocabulary reuses the same crisp spectral splatter, so the family identity
/// (luminous structure on black) is preserved while the large-scale
/// composition changes dramatically between vocabularies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructureMode {
    /// Classic look: all three inter-body edges drawn each step.
    TriangleWeb,
    /// Each body paints its own trajectory as a continuous luminous ribbon,
    /// with optional decaying time-lagged echo bands (comet tails).
    OrbitRibbons,
    /// Two of the three edges drawn; `dropped_edge` (0..=2) is omitted.
    Duet {
        /// Index of the omitted edge (0 = body0-body1, 1 = body1-body2, 2 = body2-body0).
        dropped_edge: u8,
    },
    /// Lines from each body to the instantaneous triangle centroid.
    Spokes,
    /// String-art chords from each body to its own time-lagged future position,
    /// layered over a faint ribbon underlay (ruled-surface sheets).
    TimeChords,
    /// The triangle interior swept as translucent gauze: interpolated interior
    /// lines at very low alpha accumulate into continuous luminous veils with
    /// natural caustic folds where the triangle degenerates.
    NebulaVeil,
    /// Curved quadratic-Bezier chords between bodies (bowed toward/away from
    /// the third body) tessellated into organic woven lace.
    HarmonicWeave,
    /// Pointillist dots at a fixed time pitch along each body's path; slow
    /// passages cluster into dense bead curtains (Kepler's second law made
    /// visible), with periodic brighter pearls.
    StippleConstellation,
    /// Velocity tangent segments centered on each body; the orbit appears only
    /// as the envelope of its tangents (string-art caustics).
    TangentCaustics,
}

/// Coarse family used by the stack sampler's compatibility rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VocabularyFamily {
    /// Inter-body edge sets (web, duet).
    Edges,
    /// Per-body trail ribbons.
    Trails,
    /// Body-to-centroid spokes.
    Radial,
    /// Time-lagged chords.
    Chords,
    /// Swept-area veils.
    Veil,
    /// Curved weaves.
    Weave,
    /// Stippled dots.
    Dots,
    /// Velocity tangents.
    Tangent,
}

impl StructureMode {
    /// Stable identifier used for logging and generation metadata.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::TriangleWeb => "triangle_web",
            Self::OrbitRibbons => "orbit_ribbons",
            Self::Duet { .. } => "duet",
            Self::Spokes => "spokes",
            Self::TimeChords => "time_chords",
            Self::NebulaVeil => "nebula_veil",
            Self::HarmonicWeave => "harmonic_weave",
            Self::StippleConstellation => "stipple_constellation",
            Self::TangentCaustics => "tangent_caustics",
        }
    }

    /// Numeric index recorded in the randomization log.
    #[must_use]
    pub fn log_index(self) -> usize {
        match self {
            Self::TriangleWeb => 0,
            Self::OrbitRibbons => 1,
            Self::Duet { .. } => 2,
            Self::Spokes => 3,
            Self::TimeChords => 4,
            Self::NebulaVeil => 5,
            Self::HarmonicWeave => 6,
            Self::StippleConstellation => 7,
            Self::TangentCaustics => 8,
        }
    }

    /// Number of distinct vocabularies (for log range assertions).
    pub const COUNT: usize = 9;

    /// True for vocabularies whose primary geometry is per-body trails (sparse ink).
    #[must_use]
    pub fn is_ribbon_like(self) -> bool {
        matches!(self, Self::OrbitRibbons | Self::StippleConstellation)
    }

    /// Compatibility family used when composing layer stacks.
    #[must_use]
    pub fn family(self) -> VocabularyFamily {
        match self {
            Self::TriangleWeb | Self::Duet { .. } => VocabularyFamily::Edges,
            Self::OrbitRibbons => VocabularyFamily::Trails,
            Self::Spokes => VocabularyFamily::Radial,
            Self::TimeChords => VocabularyFamily::Chords,
            Self::NebulaVeil => VocabularyFamily::Veil,
            Self::HarmonicWeave => VocabularyFamily::Weave,
            Self::StippleConstellation => VocabularyFamily::Dots,
            Self::TangentCaustics => VocabularyFamily::Tangent,
        }
    }
}

/// Symmetry operation applied to every emitted stroke.
///
/// Generalizes the old rare `mirror` flag into a small composition algebra:
/// k-fold rotational replication produces mandala/rosette images from any base
/// vocabulary. Per-copy energy is divided by the fold count so total deposited
/// ink (and therefore exposure) stays stable across symmetry classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymmetryOp {
    /// No replication (the common case).
    None,
    /// Mirror every stroke across the vertical frame axis.
    MirrorX,
    /// `k` copies rotated uniformly about the frame center.
    Rotational {
        /// Fold count in `2..=6`.
        k: u8,
    },
    /// `k` rotations plus their mirror images (`2k` copies).
    Dihedral {
        /// Fold count in `{3, 4, 6}`.
        k: u8,
    },
}

impl SymmetryOp {
    /// Total stroke copies emitted per input stroke.
    #[must_use]
    pub fn fold_count(self) -> usize {
        match self {
            Self::None => 1,
            Self::MirrorX => 2,
            Self::Rotational { k } => usize::from(k.max(1)),
            Self::Dihedral { k } => usize::from(k.max(1)) * 2,
        }
    }

    /// Stable identifier used for logging.
    #[must_use]
    pub fn label(self) -> String {
        match self {
            Self::None => "none".to_string(),
            Self::MirrorX => "mirror_x".to_string(),
            Self::Rotational { k } => format!("rot{k}"),
            Self::Dihedral { k } => format!("dih{k}"),
        }
    }

    /// Numeric index recorded in the randomization log.
    #[must_use]
    pub fn log_index(self) -> usize {
        match self {
            Self::None => 0,
            Self::MirrorX => 1,
            Self::Rotational { k } => 10 + usize::from(k),
            Self::Dihedral { k } => 20 + usize::from(k),
        }
    }
}

/// Trajectory projection applied before rendering.
///
/// Position space is the classic look; the rare phase-space projections plot
/// mixed coordinates, producing Lissajous-like curve families impossible in
/// plain position space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionMode {
    /// Plain (x, y, z) position projection.
    Position,
    /// Per-body phase portrait: (x, `v_x` scaled, y).
    PhasePortrait,
    /// Cross-body braid: body i takes its y from body (i+1) % 3.
    CrossBraid,
    /// Hodograph: pure velocity space (`v_x`, `v_y`, z).
    Hodograph,
}

impl ProjectionMode {
    /// Stable identifier used for logging.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Position => "position",
            Self::PhasePortrait => "phase_portrait",
            Self::CrossBraid => "cross_braid",
            Self::Hodograph => "hodograph",
        }
    }

    /// Numeric index recorded in the randomization log.
    #[must_use]
    pub fn log_index(self) -> usize {
        match self {
            Self::Position => 0,
            Self::PhasePortrait => 1,
            Self::CrossBraid => 2,
            Self::Hodograph => 3,
        }
    }
}

/// One composed layer: a vocabulary drawn at a continuous alpha.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StackLayer {
    /// Stroke vocabulary for this layer.
    pub vocabulary: StructureMode,
    /// Energy multiplier relative to the primary layer (0, 1].
    pub alpha: f64,
}

/// Composed structure: a primary vocabulary plus optional underlay and accent.
///
/// Additive spectral accumulation makes layer order irrelevant; "underlay" and
/// "accent" simply name the two optional reduced-alpha slots. The legacy
/// hybrid modes (web+ribbons, web+spokes lace, comet echoes) are reachable
/// stack combinations rather than special cases.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerStack {
    /// Full-strength primary vocabulary.
    pub primary: StructureMode,
    /// Optional second vocabulary at continuous alpha.
    pub underlay: Option<StackLayer>,
    /// Optional rare third vocabulary at low alpha.
    pub accent: Option<StackLayer>,
}

impl LayerStack {
    /// Stack containing only a primary vocabulary.
    #[must_use]
    pub fn solo(primary: StructureMode) -> Self {
        Self { primary, underlay: None, accent: None }
    }

    /// Stack with a primary and one underlay layer.
    #[must_use]
    pub fn with_underlay(primary: StructureMode, vocabulary: StructureMode, alpha: f64) -> Self {
        Self { primary, underlay: Some(StackLayer { vocabulary, alpha }), accent: None }
    }

    /// All layers in drawing order (primary first, alpha 1.0).
    #[must_use]
    pub fn layers(&self) -> SmallVec<[StackLayer; 3]> {
        let mut layers = SmallVec::new();
        layers.push(StackLayer { vocabulary: self.primary, alpha: 1.0 });
        if let Some(underlay) = self.underlay {
            layers.push(underlay);
        }
        if let Some(accent) = self.accent {
            layers.push(accent);
        }
        layers
    }

    /// True when any layer uses the given family.
    #[must_use]
    pub fn contains_family(&self, family: VocabularyFamily) -> bool {
        self.layers().iter().any(|layer| layer.vocabulary.family() == family)
    }

    /// Human-readable stack descriptor, e.g. `nebula_veil+orbit_ribbons@0.22`.
    #[must_use]
    pub fn label(&self) -> String {
        use std::fmt::Write as _;
        let mut label = self.primary.label().to_string();
        for layer in [self.underlay, self.accent].into_iter().flatten() {
            let _ = write!(label, "+{}@{:.2}", layer.vocabulary.label(), layer.alpha);
        }
        label
    }
}

/// Rare diffraction-spike (astrophoto star cross) finishing trait.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpikeTraits {
    /// Streak energy multiplier; 0 disables the pass entirely.
    pub strength: f64,
    /// Number of spike arms (4 or 6).
    pub arms: u8,
    /// Base rotation of the spike cross in radians.
    pub angle: f64,
    /// Arm length relative to the output short edge.
    pub length_scale: f64,
    /// Fraction of the frame's bright percentile used as the source threshold.
    pub threshold_fraction: f64,
}

impl SpikeTraits {
    /// Disabled spikes (the common case).
    #[must_use]
    pub fn disabled() -> Self {
        Self { strength: 0.0, arms: 4, angle: 0.0, length_scale: 0.03, threshold_fraction: 0.65 }
    }

    /// True when the pass should run.
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.strength > 0.0
    }
}

/// Faint seeded stardust background field splatted beneath the structure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StardustTraits {
    /// Number of micro-dots; 0 disables the field.
    pub count: u32,
    /// Relative energy of each dot.
    pub brightness: f64,
    /// Deterministic sub-seed for dot placement.
    pub seed: u64,
    /// `OKLab` lightness center of the dust colors.
    pub lightness: f64,
    /// `OKLab` chroma of the dust colors (kept low: faint tinted dust).
    pub chroma: f64,
}

impl StardustTraits {
    /// Disabled stardust (the common case).
    #[must_use]
    pub fn disabled() -> Self {
        Self { count: 0, brightness: 0.0, seed: 0, lightness: 0.72, chroma: 0.03 }
    }

    /// True when the field should be splatted.
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.count > 0 && self.brightness > 0.0
    }
}

/// Seed-resolved scene traits consumed directly by the spectral accumulator.
#[derive(Clone, Copy, Debug)]
pub struct SceneTraits {
    /// Composed layer stack (primary + optional underlay/accent).
    pub stack: LayerStack,
    /// Global line weight multiplier (hairline seeds vs bold seeds).
    pub line_weight: f64,
    /// Trail-age exposure ramp; positive brightens late steps.
    pub age_ramp: f64,
    /// Per-edge energy asymmetry for web-style vocabularies (1.0 = symmetric).
    pub edge_energy: f64,
    /// Time lag, as a fraction of total steps, for chord/echo strokes.
    pub chord_lag_fraction: f64,
    /// Alpha scale of the time-lagged ribbon echo layer (0 disables it).
    pub ribbon_echo_alpha: f64,
    /// Number of decaying echo bands behind ribbon trails (1..=3).
    pub echo_layers: u8,
    /// Interior fill lines per step for the veil vocabulary.
    pub veil_fill_lines: u8,
    /// Bow factor of harmonic-weave Bezier chords in [-0.9, 0.9].
    pub weave_bow: f64,
    /// Time pitch between stipple dots as a fraction of total steps.
    pub stipple_pitch_fraction: f64,
    /// Every Nth stipple dot becomes a brighter pearl.
    pub stipple_pearl_every: u8,
    /// Tangent segment length scale for the caustics vocabulary.
    pub tangent_length: f64,
    /// Amplitude of the calligraphic width pulse along the timeline (0 disables).
    pub width_pulse_amp: f64,
    /// Width pulse frequency in full cycles over the timeline.
    pub width_pulse_freq: f64,
    /// Width pulse phase offset in turns (0..1).
    pub width_pulse_phase: f64,
    /// Symmetry operation applied to every stroke.
    pub symmetry: SymmetryOp,
    /// Rare diffraction-spike finishing trait.
    pub spikes: SpikeTraits,
    /// Faint stardust background field.
    pub stardust: StardustTraits,
}

impl Default for SceneTraits {
    fn default() -> Self {
        Self {
            stack: LayerStack::solo(StructureMode::TriangleWeb),
            line_weight: 1.0,
            age_ramp: 0.0,
            edge_energy: 1.0,
            chord_lag_fraction: 0.008,
            ribbon_echo_alpha: 0.0,
            echo_layers: 1,
            veil_fill_lines: 7,
            weave_bow: 0.45,
            stipple_pitch_fraction: 0.000_35,
            stipple_pearl_every: 6,
            tangent_length: 1.0,
            width_pulse_amp: 0.0,
            width_pulse_freq: 3.0,
            width_pulse_phase: 0.0,
            symmetry: SymmetryOp::None,
            spikes: SpikeTraits::disabled(),
            stardust: StardustTraits::disabled(),
        }
    }
}

/// Seed-varying aesthetic values that define the `CosmicSignature` look.
#[derive(Clone, Copy, Debug)]
pub struct CosmicSignatureParameters {
    /// Multiplier applied during spectral accumulation.
    pub hdr_scale: f64,
    /// Lower percentile used as the tonemapping black point.
    pub clip_black: f64,
    /// Upper percentile used as the tonemapping white point.
    pub clip_white: f64,
    /// Palette phase value reserved for deterministic colour evolution.
    pub palette_phase: f64,
    /// Per-edge energy asymmetry consumed by web-style vocabularies.
    pub edge_energy: f64,
    /// Display exposure key: < 1 renders darker/ember seeds, > 1 brighter/airier seeds.
    pub exposure_key: f64,
    /// Composed layer stack for this seed.
    pub stack: LayerStack,
    /// Trajectory projection axis.
    pub projection: ProjectionMode,
    /// Symmetry operation applied to every stroke.
    pub symmetry: SymmetryOp,
    /// True when this seed sampled extended wildcard ranges.
    pub wildcard: bool,
    /// Global line weight multiplier (range depends on the primary vocabulary).
    pub line_weight: f64,
    /// Trail-age exposure ramp (negative = early steps brighter).
    pub age_ramp: f64,
    /// Time lag, as a fraction of total steps, for chord/echo strokes.
    pub chord_lag_fraction: f64,
    /// Alpha scale of the ribbon echo layer (0 disables the pass).
    pub ribbon_echo_alpha: f64,
    /// Number of decaying echo bands behind ribbon trails (1..=3).
    pub echo_layers: u8,
    /// Interior fill lines per step for the veil vocabulary.
    pub veil_fill_lines: u8,
    /// Bow factor of harmonic-weave Bezier chords.
    pub weave_bow: f64,
    /// Time pitch between stipple dots as a fraction of total steps.
    pub stipple_pitch_fraction: f64,
    /// Every Nth stipple dot becomes a brighter pearl.
    pub stipple_pearl_every: u8,
    /// Tangent segment length scale for the caustics vocabulary.
    pub tangent_length: f64,
    /// Amplitude of the calligraphic width pulse along the timeline (0 disables).
    pub width_pulse_amp: f64,
    /// Width pulse frequency in full cycles over the timeline.
    pub width_pulse_freq: f64,
    /// Width pulse phase offset in turns (0..1).
    pub width_pulse_phase: f64,
    /// Subtle halation (tight `DoG` bloom) strength; 0 disables the pass entirely.
    pub halation_strength: f64,
    /// Halation inner sigma, relative to the output short edge.
    pub halation_radius_scale: f64,
    /// Halation outer/inner sigma ratio (controls halo softness).
    pub halation_softness: f64,
    /// Rare prism trait: chromatic bloom strength (0 disables).
    pub prism_strength: f64,
    /// Prism blur radius relative to the output short edge.
    pub prism_radius_scale: f64,
    /// Prism RGB channel separation relative to the output short edge.
    pub prism_separation_scale: f64,
    /// Prism luminance activation threshold.
    pub prism_threshold: f64,
    /// Rare diffraction-spike finishing trait.
    pub spikes: SpikeTraits,
    /// Faint stardust background field.
    pub stardust: StardustTraits,
}

#[derive(Clone, Copy, Debug)]
struct ModeIndependentParameters {
    hdr_scale: f64,
    clip_black: f64,
    clip_white: f64,
    palette_phase: f64,
    edge_energy: f64,
    exposure_key: f64,
}

/// Legacy main-stream rolls. Consumption order and count are frozen so that
/// pre-existing seed parameters (hdr scale, palette phase, halation, prism)
/// stay aligned; `legacy_mirror_gate` is consumed but superseded by the
/// forked symmetry table.
#[derive(Clone, Copy, Debug)]
struct ModeDependentRolls {
    line_weight: f64,
    age_ramp: f64,
    chord_lag_fraction: f64,
    ribbon_echo_gate: f64,
    ribbon_echo_alpha: f64,
    halation_gate: f64,
    halation_strength: f64,
    halation_radius: f64,
    halation_softness: f64,
    prism_gate: f64,
    prism_strength: f64,
    prism_radius: f64,
    prism_separation: f64,
    prism_threshold: f64,
    #[allow(dead_code)]
    legacy_mirror_gate: f64,
}

/// Rolls drawn from the forked `cosmic-structure/v2` stream. Every roll is
/// consumed unconditionally so the stream layout is independent of gates.
#[derive(Clone, Copy, Debug)]
struct ExtendedRolls {
    wildcard_gate: f64,
    underlay_gate: f64,
    underlay_pick: f64,
    underlay_alpha: f64,
    accent_gate: f64,
    accent_pick: f64,
    accent_alpha: f64,
    symmetry_roll: f64,
    symmetry_k: f64,
    projection_roll: f64,
    projection_variant: f64,
    echo_layers: f64,
    veil_lines: f64,
    weave_bow: f64,
    stipple_pitch: f64,
    stipple_pearl: f64,
    tangent_length: f64,
    spike_gate: f64,
    spike_strength: f64,
    spike_angle: f64,
    spike_arms: f64,
    spike_length: f64,
    stardust_gate: f64,
    stardust_count: f64,
    stardust_brightness: f64,
    stardust_seed: u64,
    stardust_lightness: f64,
    stardust_chroma: f64,
    // Width-pulse rolls are appended at the end of the forked stream so every
    // earlier trait keeps its pre-existing value for already-known seeds.
    width_pulse_gate: f64,
    width_pulse_amp: f64,
    width_pulse_freq: f64,
    width_pulse_phase: f64,
}

fn resolve_extended_rolls(rng: &mut Sha3RandomByteStream) -> ExtendedRolls {
    ExtendedRolls {
        wildcard_gate: rng.next_f64(),
        underlay_gate: rng.next_f64(),
        underlay_pick: rng.next_f64(),
        underlay_alpha: rng.next_f64(),
        accent_gate: rng.next_f64(),
        accent_pick: rng.next_f64(),
        accent_alpha: rng.next_f64(),
        symmetry_roll: rng.next_f64(),
        symmetry_k: rng.next_f64(),
        projection_roll: rng.next_f64(),
        projection_variant: rng.next_f64(),
        echo_layers: rng.next_f64(),
        veil_lines: rng.next_f64(),
        weave_bow: rng.next_f64(),
        stipple_pitch: rng.next_f64(),
        stipple_pearl: rng.next_f64(),
        tangent_length: rng.next_f64(),
        spike_gate: rng.next_f64(),
        spike_strength: rng.next_f64(),
        spike_angle: rng.next_f64(),
        spike_arms: rng.next_f64(),
        spike_length: rng.next_f64(),
        stardust_gate: rng.next_f64(),
        stardust_count: rng.next_f64(),
        stardust_brightness: rng.next_f64(),
        stardust_seed: rng.next_u64(),
        stardust_lightness: rng.next_f64(),
        stardust_chroma: rng.next_f64(),
        width_pulse_gate: rng.next_f64(),
        width_pulse_amp: rng.next_f64(),
        width_pulse_freq: rng.next_f64(),
        width_pulse_phase: rng.next_f64(),
    }
}

/// Mode-aware line weight range: ribbon-style vocabularies draw bolder strokes
/// so the sparse per-body trails read as luminous bands instead of hairlines,
/// while veil fill lines stay thin so the gauze reads as continuous surface.
///
/// All ranges sit well above 1.0: full-bodied strokes with visible width
/// gradients are the family identity, and hairline seeds read as defects.
fn line_weight_range(structure: StructureMode) -> (f64, f64) {
    match structure {
        StructureMode::TriangleWeb | StructureMode::Duet { .. } => (1.10, 2.10),
        StructureMode::OrbitRibbons => (1.70, 2.90),
        StructureMode::Spokes => (1.20, 2.00),
        StructureMode::TimeChords => (1.30, 2.40),
        StructureMode::NebulaVeil => (0.95, 1.70),
        StructureMode::HarmonicWeave => (1.20, 2.20),
        StructureMode::StippleConstellation => (1.70, 3.10),
        StructureMode::TangentCaustics => (1.10, 2.00),
    }
}

/// Mode-aware halation gate: sparse vocabularies always glow, dense webs less often.
fn halation_profile(structure: StructureMode) -> (f64, (f64, f64)) {
    match structure {
        StructureMode::OrbitRibbons => (1.0, (0.10, 0.18)),
        StructureMode::StippleConstellation => (1.0, (0.10, 0.20)),
        StructureMode::TimeChords | StructureMode::TangentCaustics => (0.60, (0.06, 0.15)),
        StructureMode::HarmonicWeave => (0.55, (0.05, 0.14)),
        StructureMode::Spokes => (0.50, (0.05, 0.14)),
        StructureMode::NebulaVeil => (0.35, (0.04, 0.12)),
        StructureMode::TriangleWeb | StructureMode::Duet { .. } => (0.40, (0.05, 0.14)),
    }
}

/// Sample the primary vocabulary with curated weights.
///
/// Consumption is one roll (plus one sub-roll for the duet edge), matching the
/// legacy layout so downstream main-stream rolls stay aligned.
fn resolve_structure_mode(rng: &mut Sha3RandomByteStream) -> StructureMode {
    let roll = rng.next_f64();
    if roll < 0.17 {
        StructureMode::TriangleWeb
    } else if roll < 0.31 {
        StructureMode::OrbitRibbons
    } else if roll < 0.39 {
        let dropped_edge = ((rng.next_f64() * 3.0).floor() as u8).min(2);
        StructureMode::Duet { dropped_edge }
    } else if roll < 0.45 {
        StructureMode::Spokes
    } else if roll < 0.57 {
        StructureMode::TimeChords
    } else if roll < 0.69 {
        StructureMode::NebulaVeil
    } else if roll < 0.81 {
        StructureMode::HarmonicWeave
    } else if roll < 0.91 {
        StructureMode::StippleConstellation
    } else {
        StructureMode::TangentCaustics
    }
}

/// Weighted layer-vocabulary table used for underlay/accent picks.
///
/// Veil leads: translucent gauze beneath line work is the highest-yield
/// combination. Duet never appears as a secondary layer (a missing edge reads
/// as a glitch under another structure).
const LAYER_VOCAB_TABLE: [(StructureMode, f64); 8] = [
    (StructureMode::NebulaVeil, 0.26),
    (StructureMode::TriangleWeb, 0.16),
    (StructureMode::OrbitRibbons, 0.15),
    (StructureMode::TimeChords, 0.11),
    (StructureMode::Spokes, 0.10),
    (StructureMode::StippleConstellation, 0.10),
    (StructureMode::HarmonicWeave, 0.07),
    (StructureMode::TangentCaustics, 0.05),
];

/// Pick a secondary-layer vocabulary, excluding the given families.
fn pick_layer_vocabulary(roll: f64, excluded: &[VocabularyFamily]) -> Option<StructureMode> {
    let candidates: SmallVec<[(StructureMode, f64); 8]> = LAYER_VOCAB_TABLE
        .iter()
        .copied()
        .filter(|(vocab, _)| !excluded.contains(&vocab.family()))
        .collect();
    let total: f64 = candidates.iter().map(|(_, w)| w).sum();
    if total <= 0.0 {
        return None;
    }
    let mut cursor = roll.clamp(0.0, 1.0) * total;
    for (vocab, weight) in &candidates {
        cursor -= weight;
        if cursor <= 0.0 {
            return Some(*vocab);
        }
    }
    candidates.last().map(|(vocab, _)| *vocab)
}

fn resolve_symmetry(rolls: &ExtendedRolls) -> SymmetryOp {
    let roll = rolls.symmetry_roll;
    let none_band = 1.0 - MIRROR_PROBABILITY - ROTATIONAL_PROBABILITY - DIHEDRAL_PROBABILITY;
    if roll < none_band {
        SymmetryOp::None
    } else if roll < none_band + MIRROR_PROBABILITY {
        SymmetryOp::MirrorX
    } else if roll < none_band + MIRROR_PROBABILITY + ROTATIONAL_PROBABILITY {
        // u8 cast: result of floor of value in [0, 5).
        let k = 2 + ((rolls.symmetry_k * 5.0).floor() as u8).min(4);
        SymmetryOp::Rotational { k }
    } else {
        let k = match (rolls.symmetry_k * 3.0).floor() as u8 {
            0 => 3,
            1 => 4,
            _ => 6,
        };
        SymmetryOp::Dihedral { k }
    }
}

fn resolve_projection(rolls: &ExtendedRolls) -> ProjectionMode {
    if rolls.projection_roll >= PHASE_PROJECTION_PROBABILITY {
        return ProjectionMode::Position;
    }
    match (rolls.projection_variant * 3.0).floor() as u8 {
        0 => ProjectionMode::PhasePortrait,
        1 => ProjectionMode::CrossBraid,
        _ => ProjectionMode::Hodograph,
    }
}

fn resolve_stack(primary: StructureMode, rolls: &ExtendedRolls, wildcard: bool) -> LayerStack {
    let mut stack = LayerStack::solo(primary);
    if rolls.underlay_gate < UNDERLAY_PROBABILITY {
        let excluded = [primary.family()];
        if let Some(vocab) = pick_layer_vocabulary(rolls.underlay_pick, &excluded) {
            let alpha_max = if wildcard { 0.65 } else { 0.45 };
            let alpha = log_lerp(0.08, alpha_max, rolls.underlay_alpha);
            stack.underlay = Some(StackLayer { vocabulary: vocab, alpha });
        }
    }
    if rolls.accent_gate < ACCENT_PROBABILITY {
        let mut excluded: SmallVec<[VocabularyFamily; 2]> = SmallVec::new();
        excluded.push(primary.family());
        if let Some(underlay) = stack.underlay {
            excluded.push(underlay.vocabulary.family());
        }
        if let Some(vocab) = pick_layer_vocabulary(rolls.accent_pick, &excluded) {
            let alpha = log_lerp(0.04, 0.16, rolls.accent_alpha);
            stack.accent = Some(StackLayer { vocabulary: vocab, alpha });
        }
    }
    stack
}

impl CosmicSignatureParameters {
    fn resolve(
        base: ModeIndependentParameters,
        rolls: ModeDependentRolls,
        ext: ExtendedRolls,
        structure: StructureMode,
    ) -> Self {
        let wildcard = ext.wildcard_gate < WILDCARD_PROBABILITY;
        let stack = resolve_stack(structure, &ext, wildcard);
        let symmetry = resolve_symmetry(&ext);
        let projection = resolve_projection(&ext);

        let (lw_min, lw_max) = line_weight_range(structure);
        let lw_max = if wildcard { lw_max * 1.35 } else { lw_max };
        let line_weight = lerp(lw_min, lw_max, rolls.line_weight);
        let age_span = if wildcard { 0.85 } else { 0.6 };
        let age_ramp = lerp(-age_span, age_span, rolls.age_ramp);
        let chord_lag_fraction = lerp(0.004, 0.020, rolls.chord_lag_fraction);

        let stack_has_trails = stack.contains_family(VocabularyFamily::Trails);
        let ribbon_echo_alpha = if stack_has_trails {
            if rolls.ribbon_echo_gate < RIBBON_ECHO_PROBABILITY {
                lerp(0.18, 0.42, rolls.ribbon_echo_alpha)
            } else {
                0.0
            }
        } else {
            0.0
        };
        // u8 cast: floor of a value in [0, 3) plus one.
        let echo_layers = 1 + ((ext.echo_layers * 3.0).floor() as u8).min(2);

        // u8 cast: floor of bounded rolls.
        let veil_max = if wildcard { 12.0 } else { 9.0 };
        let veil_fill_lines = (5.0 + ext.veil_lines * (veil_max - 5.0)).floor() as u8;
        let weave_bow = lerp(-0.85, 0.85, ext.weave_bow);
        let stipple_pitch_fraction = log_lerp(0.000_15, 0.000_60, ext.stipple_pitch);
        let stipple_pearl_every = 4 + ((ext.stipple_pearl * 6.0).floor() as u8).min(5);
        let tangent_length = lerp(0.55, 1.60, ext.tangent_length);

        let (width_pulse_amp, width_pulse_freq, width_pulse_phase) =
            if ext.width_pulse_gate < WIDTH_PULSE_PROBABILITY {
                let amp_max = if wildcard { 0.70 } else { 0.55 };
                (
                    lerp(0.18, amp_max, ext.width_pulse_amp),
                    lerp(2.0, 7.0, ext.width_pulse_freq),
                    ext.width_pulse_phase,
                )
            } else {
                (0.0, 3.0, 0.0)
            };

        let (halation_probability, (hal_min, hal_max)) = halation_profile(structure);
        let hal_max = if wildcard { (hal_max * 1.6).min(0.30) } else { hal_max };
        let (halation_strength, halation_radius_scale, halation_softness) =
            if rolls.halation_gate < halation_probability {
                let radius_range =
                    if structure.is_ribbon_like() { (0.0030, 0.0050) } else { (0.0026, 0.0046) };
                (
                    lerp(hal_min, hal_max, rolls.halation_strength),
                    lerp(radius_range.0, radius_range.1, rolls.halation_radius),
                    lerp(2.8, 4.0, rolls.halation_softness),
                )
            } else {
                (0.0, 0.0035, 3.2)
            };

        let (prism_strength, prism_radius_scale, prism_separation_scale, prism_threshold) =
            if rolls.prism_gate < PRISM_PROBABILITY {
                (
                    lerp(0.16, 0.30, rolls.prism_strength),
                    lerp(0.0035, 0.0050, rolls.prism_radius),
                    lerp(0.0008, 0.0014, rolls.prism_separation),
                    lerp(0.20, 0.26, rolls.prism_threshold),
                )
            } else {
                (0.0, 0.0042, 0.0011, 0.23)
            };

        let spikes = if ext.spike_gate < SPIKE_PROBABILITY {
            SpikeTraits {
                strength: lerp(0.10, 0.32, ext.spike_strength),
                arms: if ext.spike_arms < 0.62 { 4 } else { 6 },
                angle: ext.spike_angle * std::f64::consts::FRAC_PI_2,
                length_scale: lerp(0.020, 0.048, ext.spike_length),
                threshold_fraction: 0.65,
            }
        } else {
            SpikeTraits::disabled()
        };

        let stardust = if ext.stardust_gate < STARDUST_PROBABILITY {
            StardustTraits {
                // u32 cast: bounded count.
                count: log_lerp(800.0, 3200.0, ext.stardust_count).round() as u32,
                brightness: log_lerp(0.5, 1.8, ext.stardust_brightness),
                seed: ext.stardust_seed,
                lightness: lerp(0.58, 0.86, ext.stardust_lightness),
                chroma: lerp(0.01, 0.06, ext.stardust_chroma),
            }
        } else {
            StardustTraits::disabled()
        };

        Self {
            hdr_scale: base.hdr_scale,
            clip_black: base.clip_black,
            clip_white: base.clip_white,
            palette_phase: base.palette_phase,
            edge_energy: base.edge_energy,
            exposure_key: base.exposure_key,
            stack,
            projection,
            symmetry,
            wildcard,
            line_weight,
            age_ramp,
            chord_lag_fraction,
            ribbon_echo_alpha,
            echo_layers,
            veil_fill_lines,
            weave_bow,
            stipple_pitch_fraction,
            stipple_pearl_every,
            tangent_length,
            width_pulse_amp,
            width_pulse_freq,
            width_pulse_phase,
            halation_strength,
            halation_radius_scale,
            halation_softness,
            prism_strength,
            prism_radius_scale,
            prism_separation_scale,
            prism_threshold,
            spikes,
            stardust,
        }
    }

    /// Bundle the accumulation-facing traits for the renderer.
    #[must_use]
    pub fn scene_traits(&self) -> SceneTraits {
        SceneTraits {
            stack: self.stack,
            line_weight: self.line_weight,
            age_ramp: self.age_ramp,
            edge_energy: self.edge_energy,
            chord_lag_fraction: self.chord_lag_fraction,
            ribbon_echo_alpha: self.ribbon_echo_alpha,
            echo_layers: self.echo_layers,
            veil_fill_lines: self.veil_fill_lines,
            weave_bow: self.weave_bow,
            stipple_pitch_fraction: self.stipple_pitch_fraction,
            stipple_pearl_every: self.stipple_pearl_every,
            tangent_length: self.tangent_length,
            width_pulse_amp: self.width_pulse_amp,
            width_pulse_freq: self.width_pulse_freq,
            width_pulse_phase: self.width_pulse_phase,
            symmetry: self.symmetry,
            spikes: self.spikes,
            stardust: self.stardust,
        }
    }
}

fn resolve_base_parameters(rng: &mut Sha3RandomByteStream) -> ModeIndependentParameters {
    ModeIndependentParameters {
        // Floor sits at the energy level of the strongest reference seeds:
        // below ~0.19 the 1-exp(-E) tonemap renders strokes as faint hairlines
        // instead of saturated cores with luminous gradient skirts.
        hdr_scale: sample_range(rng, 0.195, 0.285),
        clip_black: sample_range(rng, 0.0045, 0.0085),
        clip_white: sample_range(rng, 0.9935, 0.9985),
        palette_phase: sample_range(rng, 0.0, 1.0),
        edge_energy: sample_range(rng, 0.92, 1.18),
        exposure_key: sample_range(rng, 0.85, 1.12),
    }
}

fn resolve_mode_rolls(rng: &mut Sha3RandomByteStream) -> ModeDependentRolls {
    ModeDependentRolls {
        line_weight: rng.next_f64(),
        age_ramp: rng.next_f64(),
        chord_lag_fraction: rng.next_f64(),
        ribbon_echo_gate: rng.next_f64(),
        ribbon_echo_alpha: rng.next_f64(),
        halation_gate: rng.next_f64(),
        halation_strength: rng.next_f64(),
        halation_radius: rng.next_f64(),
        halation_softness: rng.next_f64(),
        prism_gate: rng.next_f64(),
        prism_strength: rng.next_f64(),
        prism_radius: rng.next_f64(),
        prism_separation: rng.next_f64(),
        prism_threshold: rng.next_f64(),
        legacy_mirror_gate: rng.next_f64(),
    }
}

/// Fully resolved `CosmicSignature` profile plus metadata for logging.
#[derive(Clone, Debug)]
pub struct ResolvedVisualProfile {
    /// Human-readable profile identifier.
    pub name: &'static str,
    /// Core seed-varying visual parameters.
    pub parameters: CosmicSignatureParameters,
    /// Render/effect configuration consumed by the existing pipeline.
    pub effect_config: ResolvedEffectConfig,
    /// Metadata describing deterministic profile randomization.
    pub randomization_log: RandomizationLog,
    base_parameters: ModeIndependentParameters,
    mode_rolls: ModeDependentRolls,
    extended_rolls: ExtendedRolls,
    preferred_stack: LayerStack,
}

impl ResolvedVisualProfile {
    /// Resolve the default `CosmicSignature` profile for the output dimensions.
    pub fn cosmic_signature(rng: &mut Sha3RandomByteStream, width: u32, height: u32) -> Self {
        let base_parameters = resolve_base_parameters(rng);
        let preferred_structure = resolve_structure_mode(rng);
        let mode_rolls = resolve_mode_rolls(rng);
        let extended_rolls = resolve_extended_rolls(&mut rng.fork(STRUCTURE_RNG_DOMAIN));
        Self::from_resolved_parts(
            width,
            height,
            base_parameters,
            mode_rolls,
            extended_rolls,
            preferred_structure,
        )
    }

    fn from_resolved_parts(
        width: u32,
        height: u32,
        base_parameters: ModeIndependentParameters,
        mode_rolls: ModeDependentRolls,
        extended_rolls: ExtendedRolls,
        structure: StructureMode,
    ) -> Self {
        let parameters = CosmicSignatureParameters::resolve(
            base_parameters,
            mode_rolls,
            extended_rolls,
            structure,
        );
        let effect_config = effect_config_from_parameters(width, height, &parameters);
        let preferred_stack = parameters.stack;

        Self {
            name: COSMIC_SIGNATURE_PROFILE_NAME,
            parameters,
            effect_config,
            randomization_log: build_profile_log(&parameters, preferred_stack),
            base_parameters,
            mode_rolls,
            extended_rolls,
            preferred_stack,
        }
    }

    /// Return a copy of this profile retargeted to the stack selected by adaptive scoring.
    #[must_use]
    pub fn with_stack(&self, stack: LayerStack) -> Self {
        let mut parameters = CosmicSignatureParameters::resolve(
            self.base_parameters,
            self.mode_rolls,
            self.extended_rolls,
            stack.primary,
        );
        parameters.stack = stack;
        let effect_config = effect_config_from_parameters(
            self.effect_config.width,
            self.effect_config.height,
            &parameters,
        );

        Self {
            name: COSMIC_SIGNATURE_PROFILE_NAME,
            parameters,
            effect_config,
            randomization_log: build_profile_log(&parameters, self.preferred_stack),
            base_parameters: self.base_parameters,
            mode_rolls: self.mode_rolls,
            extended_rolls: self.extended_rolls,
            preferred_stack: self.preferred_stack,
        }
    }

    /// Layer stack originally rolled by the seed before adaptive selection.
    #[must_use]
    pub fn preferred_stack(&self) -> LayerStack {
        self.preferred_stack
    }
}

fn effect_config_from_parameters(
    width: u32,
    height: u32,
    parameters: &CosmicSignatureParameters,
) -> ResolvedEffectConfig {
    ResolvedEffectConfig {
        width,
        height,
        enable_bloom: parameters.halation_strength > 0.0,
        enable_glow: false,
        enable_chromatic_bloom: parameters.prism_strength > 0.0,
        enable_perceptual_blur: false,
        enable_micro_contrast: false,
        enable_gradient_map: false,
        enable_color_grade: false,
        enable_champleve: false,
        enable_aether: false,
        enable_opalescence: false,
        enable_edge_luminance: false,
        enable_atmospheric_depth: false,
        enable_fine_texture: false,
        blur_strength: 0.0,
        blur_radius_scale: 0.0,
        blur_core_brightness: 1.0,
        dog_strength: parameters.halation_strength,
        dog_sigma_scale: parameters.halation_radius_scale,
        dog_ratio: parameters.halation_softness,
        glow_strength: 0.0,
        glow_threshold: 1.0,
        glow_radius_scale: 0.0,
        glow_sharpness: 1.0,
        glow_saturation_boost: 0.0,
        chromatic_bloom_strength: parameters.prism_strength,
        chromatic_bloom_radius_scale: parameters.prism_radius_scale,
        chromatic_bloom_separation_scale: parameters.prism_separation_scale,
        chromatic_bloom_threshold: parameters.prism_threshold,
        perceptual_blur_strength: 0.0,
        color_grade_strength: 0.0,
        vignette_strength: 0.0,
        vignette_softness: 1.0,
        vibrance: 1.0,
        clarity_strength: 0.0,
        tone_curve_strength: 0.0,
        gradient_map_strength: 0.0,
        gradient_map_hue_preservation: 1.0,
        gradient_map_palette: 0,
        opalescence_strength: 0.0,
        opalescence_scale: 0.0,
        opalescence_layers: 1,
        champleve_flow_alignment: 0.0,
        champleve_interference_amplitude: 0.0,
        champleve_rim_intensity: 0.0,
        champleve_rim_warmth: 0.0,
        champleve_interior_lift: 0.0,
        aether_flow_alignment: 0.0,
        aether_scattering_strength: 0.0,
        aether_iridescence_amplitude: 0.0,
        aether_caustic_strength: 0.0,
        micro_contrast_strength: 0.0,
        micro_contrast_radius: 1,
        edge_luminance_strength: 0.0,
        edge_luminance_threshold: 1.0,
        edge_luminance_brightness_boost: 0.0,
        atmospheric_depth_strength: 0.0,
        atmospheric_desaturation: 0.0,
        atmospheric_darkening: 0.0,
        atmospheric_fog_color_r: 0.0,
        atmospheric_fog_color_g: 0.0,
        atmospheric_fog_color_b: 0.0,
        fine_texture_strength: 0.0,
        fine_texture_scale: 0.0,
        fine_texture_contrast: 0.0,
        hdr_scale: parameters.hdr_scale,
        clip_black: parameters.clip_black,
        clip_white: parameters.clip_white,
    }
}

fn sample_range(rng: &mut Sha3RandomByteStream, min: f64, max: f64) -> f64 {
    lerp(min, max, rng.next_f64())
}

fn lerp(min: f64, max: f64, t: f64) -> f64 {
    min + (max - min) * t
}

/// Log-uniform interpolation: equal probability per ratio rather than per unit.
fn log_lerp(min: f64, max: f64, t: f64) -> f64 {
    (min.ln() + (max.ln() - min.ln()) * t).exp()
}

fn layer_log_index(layer: Option<StackLayer>) -> usize {
    layer.map_or(StructureMode::COUNT, |l| l.vocabulary.log_index())
}

fn build_profile_log(
    parameters: &CosmicSignatureParameters,
    preferred_stack: LayerStack,
) -> RandomizationLog {
    let mode_count = StructureMode::COUNT;
    let mut record = RandomizationRecord::new(COSMIC_SIGNATURE_PROFILE_NAME, true, true);
    record.add_float("hdr_scale", parameters.hdr_scale, true, (0.195, 0.285));
    record.add_float("clip_black", parameters.clip_black, true, (0.0045, 0.0085));
    record.add_float("clip_white", parameters.clip_white, true, (0.9935, 0.9985));
    record.add_float("palette_phase", parameters.palette_phase, true, (0.0, 1.0));
    record.add_float("edge_energy", parameters.edge_energy, true, (0.92, 1.18));
    record.add_float("exposure_key", parameters.exposure_key, true, (0.85, 1.12));
    record.add_int(
        "preferred_primary",
        preferred_stack.primary.log_index(),
        true,
        (0, mode_count - 1),
    );
    record.add_int(
        "chosen_primary",
        parameters.stack.primary.log_index(),
        true,
        (0, mode_count - 1),
    );
    record.add_int(
        "underlay_vocab",
        layer_log_index(parameters.stack.underlay),
        true,
        (0, mode_count),
    );
    record.add_float(
        "underlay_alpha",
        parameters.stack.underlay.map_or(0.0, |l| l.alpha),
        true,
        (0.0, 0.65),
    );
    record.add_int("accent_vocab", layer_log_index(parameters.stack.accent), true, (0, mode_count));
    record.add_float(
        "accent_alpha",
        parameters.stack.accent.map_or(0.0, |l| l.alpha),
        true,
        (0.0, 0.16),
    );
    record.add_int("projection", parameters.projection.log_index(), true, (0, 3));
    record.add_int("symmetry", parameters.symmetry.log_index(), true, (0, 26));
    record.add_int("wildcard", usize::from(parameters.wildcard), true, (0, 1));
    record.add_float("line_weight", parameters.line_weight, true, (0.95, 4.20));
    record.add_float("age_ramp", parameters.age_ramp, true, (-0.85, 0.85));
    record.add_float("chord_lag_fraction", parameters.chord_lag_fraction, true, (0.004, 0.020));
    record.add_float("ribbon_echo_alpha", parameters.ribbon_echo_alpha, true, (0.0, 0.42));
    record.add_int("echo_layers", usize::from(parameters.echo_layers), true, (1, 3));
    record.add_int("veil_fill_lines", usize::from(parameters.veil_fill_lines), true, (5, 12));
    record.add_float("weave_bow", parameters.weave_bow, true, (-0.85, 0.85));
    record.add_float(
        "stipple_pitch_fraction",
        parameters.stipple_pitch_fraction,
        true,
        (0.000_15, 0.000_60),
    );
    record.add_int(
        "stipple_pearl_every",
        usize::from(parameters.stipple_pearl_every),
        true,
        (4, 9),
    );
    record.add_float("tangent_length", parameters.tangent_length, true, (0.55, 1.60));
    record.add_float("width_pulse_amp", parameters.width_pulse_amp, true, (0.0, 0.70));
    record.add_float("width_pulse_freq", parameters.width_pulse_freq, true, (2.0, 7.0));
    record.add_float("width_pulse_phase", parameters.width_pulse_phase, true, (0.0, 1.0));
    record.add_float("halation_strength", parameters.halation_strength, true, (0.0, 0.30));
    record.add_float("prism_strength", parameters.prism_strength, true, (0.0, 0.30));
    record.add_float("spike_strength", parameters.spikes.strength, true, (0.0, 0.32));
    record.add_int("spike_arms", usize::from(parameters.spikes.arms), true, (4, 6));
    record.add_int("stardust_count", parameters.stardust.count as usize, true, (0, 3200));
    record.add_float("stardust_brightness", parameters.stardust.brightness, true, (0.0, 1.8));

    let mut log = RandomizationLog::new();
    log.add_record(record);
    log
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng(seed: &[u8]) -> Sha3RandomByteStream {
        Sha3RandomByteStream::new(seed, 100.0, 300.0, 300.0, 1.0)
    }

    fn profiles(count: u32, salt: u8) -> impl Iterator<Item = ResolvedVisualProfile> {
        (0..count).map(move |s| {
            let seed = [(s & 0xff) as u8, (s >> 8) as u8, salt];
            let mut rng = make_rng(&seed);
            ResolvedVisualProfile::cosmic_signature(&mut rng, 640, 360)
        })
    }

    #[test]
    fn cosmic_signature_keeps_softening_legacy_effects_off() {
        let mut rng = make_rng(&[0x10, 0x00, 0x33]);
        let profile = ResolvedVisualProfile::cosmic_signature(&mut rng, 1920, 1080);
        let c = profile.effect_config;

        assert_eq!(c.enable_bloom, profile.parameters.halation_strength > 0.0);
        assert_eq!(c.enable_chromatic_bloom, profile.parameters.prism_strength > 0.0);
        assert!(!c.enable_glow);
        assert!(!c.enable_perceptual_blur);
        assert!(!c.enable_gradient_map);
        assert!(!c.enable_color_grade);
        assert!(!c.enable_champleve);
        assert!(!c.enable_aether);
        assert!(!c.enable_opalescence);
        assert!(!c.enable_edge_luminance);
        assert!(!c.enable_atmospheric_depth);
        assert!(!c.enable_fine_texture);
    }

    #[test]
    fn cosmic_signature_halation_is_gated_and_curated() {
        let mut enabled = 0usize;
        let mut ribbon_like = 0usize;
        let mut ribbon_like_with_halation = 0usize;
        let total = 256usize;
        for profile in profiles(total as u32, 0x77) {
            let p = profile.parameters;

            assert_eq!(profile.effect_config.enable_bloom, p.halation_strength > 0.0);
            if p.stack.primary.is_ribbon_like() {
                ribbon_like += 1;
                if p.halation_strength > 0.0 {
                    ribbon_like_with_halation += 1;
                }
            }
            if p.halation_strength > 0.0 {
                enabled += 1;
                assert!((0.04..=0.30).contains(&p.halation_strength));
                assert!((0.0026..=0.0050).contains(&p.halation_radius_scale));
                assert!((2.8..=4.0).contains(&p.halation_softness));
                assert_eq!(profile.effect_config.dog_strength, p.halation_strength);
            }
        }

        assert!(enabled > total / 5, "halation almost never enabled: {enabled}/{total}");
        assert!(enabled < total * 4 / 5, "halation enabled too often: {enabled}/{total}");
        assert!(ribbon_like > 0, "expected ribbon-like seeds in the sample");
        assert_eq!(
            ribbon_like, ribbon_like_with_halation,
            "every ribbon-like seed must receive halation"
        );
    }

    #[test]
    fn cosmic_signature_primaries_cover_all_vocabularies() {
        let mut seen = std::collections::HashSet::new();
        for profile in profiles(2048, 0x5A) {
            seen.insert(profile.parameters.stack.primary.log_index());
            if let StructureMode::Duet { dropped_edge } = profile.parameters.stack.primary {
                assert!(dropped_edge <= 2, "dropped edge out of range: {dropped_edge}");
            }
        }
        assert_eq!(
            seen.len(),
            StructureMode::COUNT,
            "all vocabularies should occur as primary across seeds: {seen:?}"
        );
    }

    #[test]
    fn layer_stacks_compose_within_curated_bounds() {
        let total = 2048usize;
        let mut with_underlay = 0usize;
        let mut with_accent = 0usize;
        let mut veil_underlay = 0usize;
        for profile in profiles(total as u32, 0x91) {
            let stack = profile.parameters.stack;
            if let Some(underlay) = stack.underlay {
                with_underlay += 1;
                assert_ne!(
                    underlay.vocabulary.family(),
                    stack.primary.family(),
                    "underlay must come from a different family: {}",
                    stack.label()
                );
                assert!(
                    (0.08..=0.65).contains(&underlay.alpha),
                    "underlay alpha out of range: {}",
                    underlay.alpha
                );
                if underlay.vocabulary == StructureMode::NebulaVeil {
                    veil_underlay += 1;
                }
            }
            if let Some(accent) = stack.accent {
                with_accent += 1;
                assert_ne!(accent.vocabulary.family(), stack.primary.family());
                if let Some(underlay) = stack.underlay {
                    assert_ne!(
                        accent.vocabulary.family(),
                        underlay.vocabulary.family(),
                        "accent must differ from underlay family: {}",
                        stack.label()
                    );
                }
                assert!((0.04..=0.16).contains(&accent.alpha));
            }
        }

        let underlay_rate = with_underlay as f64 / total as f64;
        let accent_rate = with_accent as f64 / total as f64;
        assert!(
            (0.30..=0.50).contains(&underlay_rate),
            "underlay rate drifted: {underlay_rate:.3}"
        );
        assert!((0.05..=0.16).contains(&accent_rate), "accent rate drifted: {accent_rate:.3}");
        assert!(veil_underlay > 0, "veil underlays should appear (highest-yield combo)");
    }

    #[test]
    fn symmetry_projection_and_wildcard_rates_stay_curated() {
        let total = 4096usize;
        let mut mirror = 0usize;
        let mut rotational = 0usize;
        let mut dihedral = 0usize;
        let mut phase = 0usize;
        let mut wildcard = 0usize;
        for profile in profiles(total as u32, 0xC3) {
            let p = profile.parameters;
            match p.symmetry {
                SymmetryOp::MirrorX => mirror += 1,
                SymmetryOp::Rotational { k } => {
                    rotational += 1;
                    assert!((2..=6).contains(&k), "rotational k out of range: {k}");
                }
                SymmetryOp::Dihedral { k } => {
                    dihedral += 1;
                    assert!(matches!(k, 3 | 4 | 6), "dihedral k out of range: {k}");
                }
                SymmetryOp::None => {}
            }
            if p.projection != ProjectionMode::Position {
                phase += 1;
            }
            if p.wildcard {
                wildcard += 1;
            }
        }

        let rate = |count: usize| count as f64 / total as f64;
        assert!((0.01..=0.06).contains(&rate(mirror)), "mirror rate: {}", rate(mirror));
        assert!((0.04..=0.11).contains(&rate(rotational)), "rotational rate: {}", rate(rotational));
        assert!((0.02..=0.09).contains(&rate(dihedral)), "dihedral rate: {}", rate(dihedral));
        assert!((0.05..=0.12).contains(&rate(phase)), "phase projection rate: {}", rate(phase));
        assert!((0.04..=0.11).contains(&rate(wildcard)), "wildcard rate: {}", rate(wildcard));
    }

    #[test]
    fn rare_finishing_traits_are_gated_and_curated() {
        let total = 4096usize;
        let mut prism = 0usize;
        let mut spikes = 0usize;
        let mut stardust = 0usize;
        for profile in profiles(total as u32, 0xE7) {
            let p = profile.parameters;

            if p.prism_strength > 0.0 {
                prism += 1;
                assert!((0.16..=0.30).contains(&p.prism_strength));
                assert!(profile.effect_config.enable_chromatic_bloom);
            } else {
                assert!(!profile.effect_config.enable_chromatic_bloom);
            }

            if p.spikes.enabled() {
                spikes += 1;
                assert!((0.10..=0.32).contains(&p.spikes.strength));
                assert!(matches!(p.spikes.arms, 4 | 6));
                assert!((0.0..=std::f64::consts::FRAC_PI_2).contains(&p.spikes.angle));
                assert!((0.020..=0.048).contains(&p.spikes.length_scale));
            }

            if p.stardust.enabled() {
                stardust += 1;
                assert!((800..=3200).contains(&p.stardust.count));
                assert!((0.5..=1.8).contains(&p.stardust.brightness));
                assert!((0.01..=0.06).contains(&p.stardust.chroma));
            }
        }

        for (name, count, min_rate, max_rate) in [
            ("prism", prism, 0.02, 0.10),
            ("spikes", spikes, 0.015, 0.08),
            ("stardust", stardust, 0.06, 0.15),
        ] {
            let rate = count as f64 / total as f64;
            assert!(count > 0, "{name} trait never appeared in {total} seeds");
            assert!((min_rate..=max_rate).contains(&rate), "{name} rate drifted: {rate:.4}");
        }
    }

    #[test]
    fn ribbon_primaries_resolve_bolder_line_weights() {
        let mut ribbon_min = f64::INFINITY;
        let mut web_max = f64::NEG_INFINITY;
        for profile in profiles(1024, 0x9C) {
            let p = profile.parameters;
            if p.wildcard {
                continue; // wildcard seeds intentionally exceed the curated caps
            }
            match p.stack.primary {
                StructureMode::OrbitRibbons => ribbon_min = ribbon_min.min(p.line_weight),
                StructureMode::TriangleWeb => web_max = web_max.max(p.line_weight),
                _ => {}
            }
        }
        assert!(
            ribbon_min >= 1.70,
            "orbit ribbons must draw bold strokes, found line weight {ribbon_min}"
        );
        assert!(web_max <= 2.10, "triangle web line weight exceeded mode cap: {web_max}");
    }

    #[test]
    fn echo_and_vocabulary_params_stay_in_curated_ranges() {
        let mut echo_seen = false;
        for profile in profiles(1024, 0x44) {
            let p = profile.parameters;

            assert!((0.004..=0.020).contains(&p.chord_lag_fraction));
            assert!((1..=3).contains(&p.echo_layers));
            assert!((5..=12).contains(&p.veil_fill_lines));
            assert!((-0.85..=0.85).contains(&p.weave_bow));
            assert!((0.000_15..=0.000_60).contains(&p.stipple_pitch_fraction));
            assert!((4..=9).contains(&p.stipple_pearl_every));
            assert!((0.55..=1.60).contains(&p.tangent_length));

            if p.stack.contains_family(VocabularyFamily::Trails) {
                if p.ribbon_echo_alpha > 0.0 {
                    echo_seen = true;
                    assert!((0.18..=0.42).contains(&p.ribbon_echo_alpha));
                }
            } else {
                assert_eq!(p.ribbon_echo_alpha, 0.0);
            }
        }
        assert!(echo_seen, "expected at least one trail seed with the echo layer");
    }

    #[test]
    fn cosmic_signature_is_deterministic_for_seed() {
        let mut rng_a = make_rng(&[0xCA, 0xFE]);
        let mut rng_b = make_rng(&[0xCA, 0xFE]);

        let a = ResolvedVisualProfile::cosmic_signature(&mut rng_a, 800, 450);
        let b = ResolvedVisualProfile::cosmic_signature(&mut rng_b, 800, 450);

        assert_eq!(a.parameters.hdr_scale.to_bits(), b.parameters.hdr_scale.to_bits());
        assert_eq!(a.parameters.palette_phase.to_bits(), b.parameters.palette_phase.to_bits());
        assert_eq!(a.parameters.stack, b.parameters.stack);
        assert_eq!(a.parameters.symmetry, b.parameters.symmetry);
        assert_eq!(a.parameters.projection, b.parameters.projection);
        assert_eq!(a.parameters.wildcard, b.parameters.wildcard);
        assert_eq!(a.parameters.line_weight.to_bits(), b.parameters.line_weight.to_bits());
        assert_eq!(a.parameters.weave_bow.to_bits(), b.parameters.weave_bow.to_bits());
        assert_eq!(a.parameters.spikes, b.parameters.spikes);
        assert_eq!(a.parameters.stardust, b.parameters.stardust);
    }

    #[test]
    fn extended_sampling_does_not_disturb_legacy_main_stream() {
        // The forked structure stream must leave main-stream consumption
        // identical to the legacy layout: base params + structure roll +
        // 15 mode rolls. A value drawn immediately after profile resolution
        // must match a manual replay of that exact consumption.
        let seed = [0x42, 0x77, 0x10];
        let mut rng_profile = make_rng(&seed);
        let _ = ResolvedVisualProfile::cosmic_signature(&mut rng_profile, 640, 360);
        let next_after_profile = rng_profile.next_f64();

        let mut rng_manual = make_rng(&seed);
        let _ = resolve_base_parameters(&mut rng_manual);
        let _ = resolve_structure_mode(&mut rng_manual);
        let _ = resolve_mode_rolls(&mut rng_manual);
        let next_manual = rng_manual.next_f64();

        assert_eq!(
            next_after_profile.to_bits(),
            next_manual.to_bits(),
            "profile resolution must not consume extra main-stream bytes"
        );
    }

    #[test]
    fn with_stack_retargets_primary_and_preserves_seed_rolls() {
        let mut rng = make_rng(&[0xAB, 0x01]);
        let profile = ResolvedVisualProfile::cosmic_signature(&mut rng, 640, 360);
        let fallback = LayerStack::solo(StructureMode::OrbitRibbons);
        let retargeted = profile.with_stack(fallback);

        assert_eq!(retargeted.parameters.stack, fallback);
        assert_eq!(retargeted.preferred_stack(), profile.preferred_stack());
        assert_eq!(
            retargeted.parameters.hdr_scale.to_bits(),
            profile.parameters.hdr_scale.to_bits()
        );
        let (lw_min, lw_max) = line_weight_range(StructureMode::OrbitRibbons);
        assert!(
            (lw_min..=lw_max * 1.35).contains(&retargeted.parameters.line_weight),
            "retargeted line weight must use the new primary's range"
        );
    }

    #[test]
    fn stack_labels_are_stable_and_descriptive() {
        let solo = LayerStack::solo(StructureMode::NebulaVeil);
        assert_eq!(solo.label(), "nebula_veil");

        let stacked = LayerStack {
            primary: StructureMode::TriangleWeb,
            underlay: Some(StackLayer { vocabulary: StructureMode::NebulaVeil, alpha: 0.25 }),
            accent: Some(StackLayer {
                vocabulary: StructureMode::StippleConstellation,
                alpha: 0.08,
            }),
        };
        assert_eq!(stacked.label(), "triangle_web+nebula_veil@0.25+stipple_constellation@0.08");
        assert_eq!(stacked.layers().len(), 3);
        assert!(stacked.contains_family(VocabularyFamily::Veil));
        assert!(stacked.contains_family(VocabularyFamily::Dots));
        assert!(!stacked.contains_family(VocabularyFamily::Tangent));
    }

    #[test]
    fn symmetry_fold_counts_match_definition() {
        assert_eq!(SymmetryOp::None.fold_count(), 1);
        assert_eq!(SymmetryOp::MirrorX.fold_count(), 2);
        assert_eq!(SymmetryOp::Rotational { k: 5 }.fold_count(), 5);
        assert_eq!(SymmetryOp::Dihedral { k: 4 }.fold_count(), 8);
        assert_eq!(SymmetryOp::Rotational { k: 3 }.label(), "rot3");
        assert_eq!(SymmetryOp::Dihedral { k: 6 }.label(), "dih6");
    }

    #[test]
    fn pick_layer_vocabulary_respects_family_exclusions() {
        for roll in [0.0, 0.13, 0.37, 0.62, 0.88, 0.999] {
            let picked = pick_layer_vocabulary(roll, &[VocabularyFamily::Veil])
                .expect("non-empty candidate set");
            assert_ne!(picked.family(), VocabularyFamily::Veil);
            assert!(!matches!(picked, StructureMode::Duet { .. }));
        }
        // Excluding everything yields no candidate.
        let all = [
            VocabularyFamily::Edges,
            VocabularyFamily::Trails,
            VocabularyFamily::Radial,
            VocabularyFamily::Chords,
            VocabularyFamily::Veil,
            VocabularyFamily::Weave,
            VocabularyFamily::Dots,
            VocabularyFamily::Tangent,
        ];
        assert_eq!(pick_layer_vocabulary(0.5, &all), None);
    }

    #[test]
    fn cosmic_signature_records_profile_randomization() {
        let mut rng = make_rng(&[1, 2, 3, 4]);
        let profile = ResolvedVisualProfile::cosmic_signature(&mut rng, 640, 360);

        assert_eq!(profile.name, COSMIC_SIGNATURE_PROFILE_NAME);
        assert_eq!(profile.randomization_log.effects.len(), 1);
        assert_eq!(profile.randomization_log.effects[0].effect_name, COSMIC_SIGNATURE_PROFILE_NAME);
        assert_eq!(profile.randomization_log.effects[0].parameters.len(), 34);
    }

    #[test]
    fn scene_traits_default_matches_classic_look() {
        let traits = SceneTraits::default();
        assert_eq!(traits.stack.primary, StructureMode::TriangleWeb);
        assert!(traits.stack.underlay.is_none());
        assert_eq!(traits.line_weight, 1.0);
        assert_eq!(traits.age_ramp, 0.0);
        assert_eq!(traits.edge_energy, 1.0);
        assert_eq!(traits.ribbon_echo_alpha, 0.0);
        assert_eq!(traits.symmetry, SymmetryOp::None);
        assert!(!traits.spikes.enabled());
        assert!(!traits.stardust.enabled());
    }

    #[test]
    fn cosmic_signature_finish_pipeline_matches_gated_traits() {
        let mut saw_empty = false;
        let mut saw_halation = false;

        for seed in 0u8..64 {
            let mut rng = make_rng(&[seed, 8, 7, 6]);
            let profile = ResolvedVisualProfile::cosmic_signature(&mut rng, 1280, 720);
            let render_config = crate::render::RenderConfig {
                hdr_scale: profile.effect_config.hdr_scale,
                bloom_mode: crate::render::BloomMode::Dog,
            };
            let effect_config = crate::render::build_effect_config_from_resolved(
                &profile.effect_config,
                &render_config,
                crate::render::FinishOutputMode::Still,
            );
            let pipeline = crate::render::effects::FinishEffectPipeline::new(effect_config);

            assert_eq!(pipeline.image_len(), 0);
            let expected = usize::from(profile.parameters.halation_strength > 0.0)
                + usize::from(profile.parameters.prism_strength > 0.0);
            assert_eq!(
                pipeline.trajectory_len(),
                expected,
                "trajectory chain length must match gated traits"
            );
            if profile.parameters.halation_strength > 0.0 {
                saw_halation = true;
            } else if expected == 0 {
                saw_empty = true;
            }
        }

        assert!(saw_empty, "expected at least one seed without finish passes");
        assert!(saw_halation, "expected at least one seed with halation");
    }

    #[test]
    fn log_lerp_is_log_uniform() {
        assert!((log_lerp(1.0, 100.0, 0.5) - 10.0).abs() < 1e-9);
        assert!((log_lerp(0.1, 10.0, 0.0) - 0.1).abs() < 1e-12);
        assert!((log_lerp(0.1, 10.0, 1.0) - 10.0).abs() < 1e-9);
    }
}
