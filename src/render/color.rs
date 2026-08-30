//! Procedural color generation for the spectral renderer.
//!
//! There are no preset palettes. Every palette is a point in a continuous
//! genome space sampled in `OKLCh`: hue anchor, hue dispersion (a log-uniform
//! continuum that passes smoothly through monochrome, analogous,
//! complementary, and triadic relationships without naming them), chroma and
//! lightness envelopes, and continuous temporal evolution rates.
//!
//! Beauty is enforced by a deterministic **gate**, not by curation: sampled
//! genomes must pass perceptual checks (minimum `OKLab` distance between
//! bodies, chroma energy floor/ceiling, lightness ladder, no red/green
//! opposition) or they are resampled from the forked stream; after a bounded
//! number of attempts the last candidate is deterministically repaired.
//! *Sample wild, gate hard.*

use crate::oklab::{max_display_p3_chroma_for_lh, oklch_to_oklab};
use crate::render::constants::{BASE_HUE_DRIFT, HUE_DRIFT_SCALE, HUE_FULL_CIRCLE};
use crate::sim::Sha3RandomByteStream;
use std::sync::{LazyLock, Mutex};
use tracing::info;

/// Type alias for `OKLab` color (L, a, b components)
pub type OklabColor = (f64, f64, f64);

/// Small random hue variation for visual interest
const HUE_DRIFT_JITTER: f64 = 0.1;
const COLOR_RNG_DOMAIN: &[u8] = b"cosmic-color/v3";
const GLOW_LIGHTNESS_FLOOR: f64 = 0.62;
const DOMINANT_CHROMA_FRACTION_FLOOR: f64 = 0.62;

/// Lower bound of the continuous per-body alpha multiplier (log-uniform).
const ALPHA_VARIATION_MIN: f64 = 0.55;
/// Upper bound of the continuous per-body alpha multiplier (log-uniform).
const ALPHA_VARIATION_MAX: f64 = 1.80;

/// Maximum genome resamples before the deterministic repair kicks in.
pub const MAX_PALETTE_ATTEMPTS: usize = 24;

/// Gate floor: minimum pairwise `OKLab` distance between mean body colors.
pub const GATE_MIN_BODY_DISTANCE: f64 = 0.085;
/// Gate floor on mean chroma (everything grey reads as mud).
pub const GATE_MIN_MEAN_CHROMA: f64 = 0.045;
/// Stricter chroma floor for near-monochrome palettes (hue cannot separate).
pub const GATE_MIN_MEAN_CHROMA_TIGHT: f64 = 0.070;
/// Gate ceiling on mean chroma (everything neon clips after tonemapping).
pub const GATE_MAX_MEAN_CHROMA: f64 = 0.30;
/// Dispersion below which a palette counts as near-monochrome for gating.
pub const GATE_TIGHT_DISPERSION: f64 = 40.0;
/// Lightness ladder floor for near-monochrome palettes.
pub const GATE_MIN_TIGHT_LIGHTNESS_SPAN: f64 = 0.10;
/// Lightness ladder floor for ordinary palettes.
pub const GATE_MIN_LIGHTNESS_SPAN: f64 = 0.04;

/// Continuous palette genome: every field is sampled, none is a preset.
#[derive(Clone, Copy, Debug)]
pub struct PaletteGenome {
    /// Hue anchor in degrees, uniform over the full circle.
    pub anchor: f64,
    /// Total hue span of the three bodies in degrees (log-uniform 8..300):
    /// the continuum from monochrome through analogous to triadic.
    pub dispersion: f64,
    /// Asymmetry of the middle body inside the span, in [-1, 1].
    pub skew: f64,
    /// Cusp fraction of the most chromatic body (log-uniform).
    pub chroma_peak: f64,
    /// Least-chromatic body's fraction of the peak.
    pub chroma_floor_ratio: f64,
    /// Center of the body lightness ladder.
    pub lightness_center: f64,
    /// Height of the body lightness ladder.
    pub lightness_span: f64,
    /// Scale of per-body hue journeys across the timeline.
    pub hue_journey_scale: f64,
    /// Frequency of the hue sway wave over the timeline.
    pub wave_freq: f64,
    /// Amplitude of the hue sway wave in degrees (log-uniform).
    pub wave_amp: f64,
    /// Strength of the secondary hue accent wave in degrees.
    pub accent_strength: f64,
    /// Amplitude scale of per-body lightness waves.
    pub lightness_wave: f64,
    /// Amplitude scale of per-body chroma waves.
    pub chroma_wave: f64,
    /// Gate attempts consumed before this genome passed (1 = first try).
    pub gate_attempts: usize,
    /// True when the bounded gate exhausted and deterministic repair ran.
    pub repaired: bool,
}

#[derive(Clone, Copy)]
struct BodyColorPlan {
    base_hue: f64,
    target_lightness: f64,
    lightness_range: f64,
    lightness_wave: f64,
    chroma_fraction: f64,
    chroma_noise: f64,
    chroma_wave: f64,
    hue_journey: f64,
    phase: f64,
    is_dominant: bool,
}

#[derive(Clone)]
struct PaletteSpec {
    genome: PaletteGenome,
    bodies: [BodyColorPlan; 3],
    palette_phase: f64,
}

static LAST_PALETTE_METADATA: LazyLock<Mutex<(String, String)>> =
    LazyLock::new(|| Mutex::new(("unresolved".to_string(), "unresolved".to_string())));

/// Return the continuous fingerprint and gate descriptor of the most recent
/// palette generation (replaces the old harmony/mood preset labels).
#[must_use]
pub fn current_palette_metadata() -> (String, String) {
    LAST_PALETTE_METADATA.lock().map_or_else(
        |_| ("unresolved".to_string(), "unresolved".to_string()),
        |metadata| metadata.clone(),
    )
}

/// Human-facing summary of the most recent resolved palette, consumed by the
/// NFT trait metadata pipeline.
#[derive(Clone, Debug)]
pub struct PaletteDetails {
    /// Continuous genome fingerprint (same string as the generation log).
    pub fingerprint: String,
    /// Beauty-gate descriptor (`gateN` / `gateN_repaired`).
    pub gate: String,
    /// Bucketed human-readable family name, e.g. `"Ember Triad"`.
    pub family: String,
    /// Genome hue anchor in degrees.
    pub anchor_deg: f64,
    /// Genome hue dispersion (total span) in degrees.
    pub dispersion_deg: f64,
    /// Per-body base hues in `OKLCh` degrees.
    pub body_base_hues_deg: [f64; 3],
    /// Index of the most chromatic (dominant) body.
    pub dominant_body: usize,
}

static LAST_PALETTE_DETAILS: LazyLock<Mutex<Option<PaletteDetails>>> =
    LazyLock::new(|| Mutex::new(None));

/// Return the detailed summary of the most recent palette generation, or
/// `None` when no palette has been resolved in this process yet.
#[must_use]
pub fn current_palette_details() -> Option<PaletteDetails> {
    LAST_PALETTE_DETAILS.lock().map_or(None, |details| details.clone())
}

/// Hue-sector names for the palette family, tuned to `OKLCh` hue landmarks.
const HUE_FAMILY_NAMES: [(f64, &str); 12] = [
    (20.0, "Rose"),
    (45.0, "Ember"),
    (75.0, "Amber"),
    (105.0, "Solar"),
    (140.0, "Aurora"),
    (170.0, "Jade"),
    (200.0, "Glacial"),
    (235.0, "Cerulean"),
    (270.0, "Sapphire"),
    (300.0, "Violet"),
    (330.0, "Nebular"),
    (360.0, "Orchid"),
];

/// Bucketed human name for a continuous palette genome: a hue-sector word
/// from the anchor plus a dispersion qualifier. The thresholds are frozen;
/// changing them would change published trait values.
#[must_use]
pub fn palette_family(anchor_deg: f64, dispersion_deg: f64) -> String {
    let hue = anchor_deg.rem_euclid(HUE_FULL_CIRCLE);
    let hue_name =
        HUE_FAMILY_NAMES.iter().find(|(upper, _)| hue < *upper).map_or("Orchid", |(_, name)| name);
    let spread = if dispersion_deg < 30.0 {
        "Mono"
    } else if dispersion_deg < 90.0 {
        "Analogous"
    } else if dispersion_deg < 200.0 {
        "Split"
    } else {
        "Triad"
    };
    format!("{hue_name} {spread}")
}

#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

#[inline]
fn log_lerp(min: f64, max: f64, t: f64) -> f64 {
    (min.ln() + (max.ln() - min.ln()) * t).exp()
}

#[inline]
fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn shuffle3(rng: &mut Sha3RandomByteStream, values: &mut [usize; 3]) {
    for i in (1..3).rev() {
        let j = (rng.next_f64() * (i + 1) as f64).floor() as usize;
        values.swap(i, j);
    }
}

fn is_red_sector(hue: f64) -> bool {
    let hue = hue.rem_euclid(HUE_FULL_CIRCLE);
    !(35.0..=335.0).contains(&hue)
}

fn is_green_sector(hue: f64) -> bool {
    let hue = hue.rem_euclid(HUE_FULL_CIRCLE);
    (92.0..=152.0).contains(&hue)
}

fn has_red_green_opposition(hues: [f64; 3]) -> bool {
    let has_red = hues.iter().any(|&hue| is_red_sector(hue));
    let has_green = hues.iter().any(|&hue| is_green_sector(hue));
    has_red && has_green
}

/// Rotate green-sector hues away from a red/green clash (deterministic repair).
fn repair_red_green_opposition(mut hues: [f64; 3]) -> [f64; 3] {
    if has_red_green_opposition(hues) {
        for hue in &mut hues {
            if is_green_sector(*hue) {
                *hue = (*hue + 55.0).rem_euclid(HUE_FULL_CIRCLE);
            }
        }
    }
    hues
}

/// Mean `OKLab` color a body plan integrates toward (cheap gate proxy).
fn plan_mean_color(plan: &BodyColorPlan) -> OklabColor {
    let lightness = plan.target_lightness.clamp(0.30, 0.94);
    let max_chroma = max_display_p3_chroma_for_lh(lightness, plan.base_hue);
    let chroma = (max_chroma * plan.chroma_fraction).min(max_chroma * 0.995);
    oklch_to_oklab(lightness, chroma, plan.base_hue)
}

fn oklab_distance(a: OklabColor, b: OklabColor) -> f64 {
    let dl = a.0 - b.0;
    let da = a.1 - b.1;
    let db = a.2 - b.2;
    (dl * dl + da * da + db * db).sqrt()
}

/// Perceptual quality gate. Returns `None` when the candidate passes, or a
/// short reason string used for diagnostics and tests.
fn gate_failure(genome: &PaletteGenome, bodies: &[BodyColorPlan; 3]) -> Option<&'static str> {
    let colors =
        [plan_mean_color(&bodies[0]), plan_mean_color(&bodies[1]), plan_mean_color(&bodies[2])];
    for color in &colors {
        if !(color.0.is_finite() && color.1.is_finite() && color.2.is_finite()) {
            return Some("non_finite");
        }
    }

    let min_distance = oklab_distance(colors[0], colors[1])
        .min(oklab_distance(colors[1], colors[2]))
        .min(oklab_distance(colors[0], colors[2]));
    if min_distance < GATE_MIN_BODY_DISTANCE {
        return Some("body_distance");
    }

    let mean_chroma = colors.iter().map(|(_, a, b)| (a * a + b * b).sqrt()).sum::<f64>() / 3.0;
    let chroma_floor = if genome.dispersion < GATE_TIGHT_DISPERSION {
        GATE_MIN_MEAN_CHROMA_TIGHT
    } else {
        GATE_MIN_MEAN_CHROMA
    };
    if mean_chroma < chroma_floor {
        return Some("chroma_floor");
    }
    if mean_chroma > GATE_MAX_MEAN_CHROMA {
        return Some("chroma_ceiling");
    }

    let lightness: Vec<f64> = bodies.iter().map(|b| b.target_lightness).collect();
    let span = lightness.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - lightness.iter().copied().fold(f64::INFINITY, f64::min);
    let span_floor = if genome.dispersion < GATE_TIGHT_DISPERSION {
        GATE_MIN_TIGHT_LIGHTNESS_SPAN
    } else {
        GATE_MIN_LIGHTNESS_SPAN
    };
    if span < span_floor {
        return Some("lightness_span");
    }

    let hues = [bodies[0].base_hue, bodies[1].base_hue, bodies[2].base_hue];
    if has_red_green_opposition(hues) {
        return Some("red_green");
    }

    None
}

/// Deterministic repair applied when the bounded gate exhausts: rotate hue
/// clashes apart, lift chroma above the mud floor, and widen the lightness
/// ladder. Guarantees termination with a displayable palette.
fn repair_palette(genome: &mut PaletteGenome, bodies: &mut [BodyColorPlan; 3]) {
    let hues =
        repair_red_green_opposition([bodies[0].base_hue, bodies[1].base_hue, bodies[2].base_hue]);
    for (plan, hue) in bodies.iter_mut().zip(hues) {
        plan.base_hue = hue;
    }

    // Spread hues apart until the pairwise distance floor holds.
    let spread = [0.0, 24.0, -24.0];
    loop {
        let colors =
            [plan_mean_color(&bodies[0]), plan_mean_color(&bodies[1]), plan_mean_color(&bodies[2])];
        let min_distance = oklab_distance(colors[0], colors[1])
            .min(oklab_distance(colors[1], colors[2]))
            .min(oklab_distance(colors[0], colors[2]));
        if min_distance >= GATE_MIN_BODY_DISTANCE {
            break;
        }
        for (plan, delta) in bodies.iter_mut().zip(spread) {
            plan.base_hue = (plan.base_hue + delta).rem_euclid(HUE_FULL_CIRCLE);
        }
        let lightness_targets = [0.78, 0.62, 0.44];
        for (plan, target) in bodies.iter_mut().zip(lightness_targets) {
            plan.target_lightness = lerp(plan.target_lightness, target, 0.5);
        }
        for plan in bodies.iter_mut() {
            plan.chroma_fraction = plan.chroma_fraction.max(0.60);
        }
    }

    for plan in bodies.iter_mut() {
        plan.chroma_fraction = plan.chroma_fraction.max(0.45);
    }
    genome.repaired = true;
}

/// Sample one genome + body plans from the stream (no gating).
fn sample_palette_candidate(
    rng: &mut Sha3RandomByteStream,
    chroma_boost: bool,
) -> (PaletteGenome, [BodyColorPlan; 3]) {
    let anchor = rng.next_f64() * HUE_FULL_CIRCLE;
    let dispersion = log_lerp(8.0, 300.0, rng.next_f64());
    let skew = rng.next_f64() * 2.0 - 1.0;
    let chroma_min = if chroma_boost { 0.34 } else { 0.26 };
    let chroma_peak = log_lerp(chroma_min, 0.97, rng.next_f64());
    let chroma_floor_ratio = lerp(0.25, 0.75, rng.next_f64());
    let lightness_center = lerp(0.46, 0.80, rng.next_f64());
    let lightness_span = lerp(0.10, 0.40, rng.next_f64());
    let hue_journey_scale = rng.next_f64();
    let wave_freq = lerp(0.6, 6.0, rng.next_f64());
    let wave_amp = log_lerp(3.0, 42.0, rng.next_f64());
    let accent_strength = lerp(4.0, 40.0, rng.next_f64());
    let lightness_wave = lerp(0.030, 0.160, rng.next_f64());
    let chroma_wave = lerp(0.020, 0.100, rng.next_f64());

    let genome = PaletteGenome {
        anchor,
        dispersion,
        skew,
        chroma_peak,
        chroma_floor_ratio,
        lightness_center,
        lightness_span,
        hue_journey_scale,
        wave_freq,
        wave_amp,
        accent_strength,
        lightness_wave,
        chroma_wave,
        gate_attempts: 1,
        repaired: false,
    };

    // Hue placement inside the dispersion span, with per-body jitter and
    // shuffled assignment so body 0 is not always the low-hue body.
    let offsets = [-0.5 * dispersion, 0.35 * skew * dispersion, 0.5 * dispersion];
    let jitters = [
        (rng.next_f64() - 0.5) * 20.0,
        (rng.next_f64() - 0.5) * 20.0,
        (rng.next_f64() - 0.5) * 20.0,
    ];
    let mut hue_order = [0usize, 1, 2];
    shuffle3(rng, &mut hue_order);

    let mut lightness_order = [0usize, 1, 2];
    let mut chroma_order = [0usize, 1, 2];
    shuffle3(rng, &mut lightness_order);
    shuffle3(rng, &mut chroma_order);

    let mid_jitter = (rng.next_f64() - 0.5) * 0.30;
    let lightness_values = [
        (lightness_center + 0.5 * lightness_span).clamp(0.34, 0.92),
        (lightness_center + mid_jitter * lightness_span).clamp(0.34, 0.92),
        (lightness_center - 0.5 * lightness_span).clamp(0.30, 0.88),
    ];
    let chroma_mid_t = rng.next_f64();
    let chroma_values = [
        chroma_peak,
        chroma_peak * lerp(chroma_floor_ratio, 1.0, chroma_mid_t),
        chroma_peak * chroma_floor_ratio,
    ];

    let journey_max = lerp(12.0, 110.0, hue_journey_scale) * (dispersion / 300.0).max(0.25);

    let mut plans = [BodyColorPlan {
        base_hue: 0.0,
        target_lightness: 0.65,
        lightness_range: 0.08,
        lightness_wave: 0.08,
        chroma_fraction: 0.6,
        chroma_noise: 0.08,
        chroma_wave: 0.04,
        hue_journey: 0.0,
        phase: 0.0,
        is_dominant: false,
    }; 3];

    for (body, plan) in plans.iter_mut().enumerate() {
        let hue_rank = hue_order.iter().position(|idx| *idx == body).unwrap_or(1);
        let lightness_rank = lightness_order.iter().position(|idx| *idx == body).unwrap_or(1);
        let chroma_rank = chroma_order.iter().position(|idx| *idx == body).unwrap_or(1);
        let is_dominant = chroma_rank == 0;
        let mut target_lightness = lightness_values[lightness_rank];
        if is_dominant {
            target_lightness = target_lightness.max(GLOW_LIGHTNESS_FLOOR);
        }

        *plan = BodyColorPlan {
            base_hue: (anchor + offsets[hue_rank] + jitters[hue_rank]).rem_euclid(HUE_FULL_CIRCLE),
            target_lightness,
            lightness_range: lerp(0.030, 0.110, rng.next_f64()),
            lightness_wave: lightness_wave * lerp(0.6, 1.4, rng.next_f64()),
            chroma_fraction: chroma_values[chroma_rank].clamp(0.10, 0.985),
            chroma_noise: lerp(0.04, 0.14, rng.next_f64()),
            chroma_wave: chroma_wave * lerp(0.6, 1.4, rng.next_f64()),
            hue_journey: (rng.next_f64() - 0.5) * 2.0 * journey_max,
            phase: rng.next_f64(),
            is_dominant,
        };
    }

    (genome, plans)
}

/// Sample genomes until the beauty gate passes (bounded, deterministic).
fn resolve_palette_spec(
    rng: &mut Sha3RandomByteStream,
    chroma_boost: bool,
    palette_phase: f64,
) -> PaletteSpec {
    let palette_phase = palette_phase.clamp(0.0, 1.0);

    let mut candidate = sample_palette_candidate(rng, chroma_boost);
    for attempt in 1..=MAX_PALETTE_ATTEMPTS {
        candidate.0.gate_attempts = attempt;
        if gate_failure(&candidate.0, &candidate.1).is_none() {
            let (genome, bodies) = candidate;
            return PaletteSpec { genome, bodies, palette_phase };
        }
        if attempt < MAX_PALETTE_ATTEMPTS {
            candidate = sample_palette_candidate(rng, chroma_boost);
        }
    }

    let (mut genome, mut bodies) = candidate;
    repair_palette(&mut genome, &mut bodies);
    PaletteSpec { genome, bodies, palette_phase }
}

/// Continuous numeric fingerprint of the resolved palette (for logs).
fn genome_fingerprint(genome: &PaletteGenome) -> String {
    format!(
        "h{:05.1}_d{:05.1}_s{:+.2}_c{:.2}x{:.2}_l{:.2}w{:.2}_f{:.2}a{:.1}",
        genome.anchor,
        genome.dispersion,
        genome.skew,
        genome.chroma_peak,
        genome.chroma_floor_ratio,
        genome.lightness_center,
        genome.lightness_span,
        genome.wave_freq,
        genome.wave_amp,
    )
}

fn gate_descriptor(genome: &PaletteGenome) -> String {
    if genome.repaired {
        format!("gate{}_repaired", genome.gate_attempts)
    } else {
        format!("gate{}", genome.gate_attempts)
    }
}

/// Generate color gradient optimized for `OKLab` space.
///
/// Generates colors in `OKLCh` (cylindrical `OKLab`) for perceptually
/// uniform distribution. `chroma_boost` selects richer saturation
/// floors; `hue_wave_freq` controls per-seed color rhythm.
pub fn generate_color_gradient_oklab(
    rng: &mut Sha3RandomByteStream,
    length: usize,
    body_index: usize,
    base_hue_offset: f64,
    chroma_boost: bool,
    hue_wave_freq: f64,
) -> Vec<OklabColor> {
    let palette = resolve_palette_spec(rng, chroma_boost, 0.5);
    generate_color_gradient_with_palette(
        rng,
        length,
        body_index,
        base_hue_offset,
        hue_wave_freq,
        &palette,
    )
}

fn generate_color_gradient_with_palette(
    rng: &mut Sha3RandomByteStream,
    length: usize,
    body_index: usize,
    base_hue_offset: f64,
    hue_wave_freq: f64,
    palette: &PaletteSpec,
) -> Vec<OklabColor> {
    let mut colors = Vec::with_capacity(length);

    let body = palette.bodies[body_index % palette.bodies.len()];
    let base_hue = body.base_hue;
    let phase_jitter = rng.next_f64() * 0.1 + palette.palette_phase + body.phase * 0.07;

    let ln_cache: Vec<f64> =
        (0..length).map(|i| if i > 0 { (i as f64).ln() } else { 0.0 }).collect();
    let wave_cache: Vec<f64> = (0..length)
        .map(|i| {
            let t = i as f64 / length.max(1) as f64;
            let phase_offset = body_index as f64 * 0.33 + phase_jitter + body.phase;
            ((phase_offset + t * hue_wave_freq) * std::f64::consts::TAU).sin()
        })
        .collect();
    let accent_cache: Vec<f64> = (0..length)
        .map(|i| {
            let t = i as f64 / length.max(1) as f64;
            let accent_phase = palette.palette_phase + body_index as f64 * 0.618 + body.phase;
            ((accent_phase + t * (hue_wave_freq * 0.37 + 0.71)) * std::f64::consts::TAU).sin()
        })
        .collect();

    let random_bits: Vec<u8> = (0..length).map(|_| rng.next_byte()).collect();
    let random_chromas: Vec<f64> = (0..length).map(|_| rng.next_f64()).collect();
    let random_lightnesses: Vec<f64> = (0..length).map(|_| rng.next_f64()).collect();

    for step in 0..length {
        let t = step as f64 / length.max(1) as f64;
        let journey = (smoothstep(t) - 0.5) * body.hue_journey;
        let mut current_hue = base_hue
            + journey
            + base_hue_offset * (1.0 + ln_cache[step]) * HUE_DRIFT_SCALE
            + wave_cache[step] * palette.genome.wave_amp
            + accent_cache[step] * palette.genome.accent_strength;

        if random_bits[step] & 1 == 0 {
            current_hue += HUE_DRIFT_JITTER;
        } else {
            current_hue -= HUE_DRIFT_JITTER;
        }
        current_hue = current_hue.rem_euclid(HUE_FULL_CIRCLE);

        let wave_factor = wave_cache[step];
        let accent_factor = accent_cache[step];
        let mut lightness = body.target_lightness
            + (random_lightnesses[step] - 0.5) * body.lightness_range
            + wave_factor * body.lightness_wave
            + accent_factor * 0.018;
        if body.is_dominant {
            lightness = lightness.max(GLOW_LIGHTNESS_FLOOR);
        }
        lightness = lightness.clamp(0.34, 0.94);

        let mut chroma_fraction = body.chroma_fraction
            + (random_chromas[step] - 0.5) * body.chroma_noise
            + wave_factor * body.chroma_wave
            + accent_factor.abs() * 0.035;
        if body.is_dominant {
            chroma_fraction = chroma_fraction.max(DOMINANT_CHROMA_FRACTION_FLOOR);
        }
        chroma_fraction = chroma_fraction.clamp(0.10, 0.985);
        let max_chroma = max_display_p3_chroma_for_lh(lightness, current_hue);
        let chroma_floor = if body.is_dominant { 0.055 } else { 0.026 };
        let mut chroma = max_chroma * chroma_fraction;
        if max_chroma > chroma_floor {
            chroma = chroma.max(chroma_floor);
        }
        chroma = chroma.min(max_chroma * 0.995);

        colors.push(oklch_to_oklab(lightness, chroma, current_hue));
    }

    colors
}

/// Generate 3 color sequences + per-body alphas.
///
/// `chroma_boost`: raise the chroma genome floor.
/// `alpha_variation`: give each body a continuously sampled alpha for depth.
pub fn generate_body_color_sequences(
    rng: &mut Sha3RandomByteStream,
    length: usize,
    alpha_denom: usize,
    chroma_boost: bool,
    alpha_variation: bool,
    palette_phase: f64,
) -> (Vec<Vec<OklabColor>>, Vec<f64>) {
    let mut rng = rng.fork(COLOR_RNG_DOMAIN);
    let base_hue_offset = BASE_HUE_DRIFT;

    let palette_phase = palette_phase.clamp(0.0, 1.0);
    let palette = resolve_palette_spec(&mut rng, chroma_boost, palette_phase);
    let hue_wave_freq = palette.genome.wave_freq + palette_phase * 0.45;
    let fingerprint = genome_fingerprint(&palette.genome);
    let gate = gate_descriptor(&palette.genome);
    if let Ok(mut metadata) = LAST_PALETTE_METADATA.lock() {
        *metadata = (fingerprint.clone(), gate.clone());
    }
    if let Ok(mut details) = LAST_PALETTE_DETAILS.lock() {
        let dominant_body = palette.bodies.iter().position(|plan| plan.is_dominant).unwrap_or(0);
        *details = Some(PaletteDetails {
            fingerprint: fingerprint.clone(),
            gate: gate.clone(),
            family: palette_family(palette.genome.anchor, palette.genome.dispersion),
            anchor_deg: palette.genome.anchor,
            dispersion_deg: palette.genome.dispersion,
            body_base_hues_deg: [
                palette.bodies[0].base_hue,
                palette.bodies[1].base_hue,
                palette.bodies[2].base_hue,
            ],
            dominant_body,
        });
    }
    info!("   => Palette genome {fingerprint} ({gate}) phase={:.3}", palette.palette_phase);

    let b1 = generate_color_gradient_with_palette(
        &mut rng,
        length,
        0,
        base_hue_offset,
        hue_wave_freq,
        &palette,
    );
    let b2 = generate_color_gradient_with_palette(
        &mut rng,
        length,
        1,
        base_hue_offset,
        hue_wave_freq,
        &palette,
    );
    let b3 = generate_color_gradient_with_palette(
        &mut rng,
        length,
        2,
        base_hue_offset,
        hue_wave_freq,
        &palette,
    );

    let body_alphas = if alpha_variation {
        // Continuous log-uniform multipliers: the wide range lets one body
        // genuinely dominate while another recedes, which combines with the
        // lightness/chroma hierarchy to give each seed a clear protagonist.
        let base = 1.0 / alpha_denom as f64;
        let (ln_min, ln_max) = (ALPHA_VARIATION_MIN.ln(), ALPHA_VARIATION_MAX.ln());
        let alphas: Vec<f64> =
            (0..3).map(|_| base * (ln_min + rng.next_f64() * (ln_max - ln_min)).exp()).collect();
        info!(
            "   => Per-body alpha variation: {:.3e}, {:.3e}, {:.3e}",
            alphas[0], alphas[1], alphas[2]
        );
        alphas
    } else {
        let alpha_value = 1.0 / alpha_denom as f64;
        info!("   => Uniform body alpha: 1/{alpha_denom} = {alpha_value:.3e}");
        vec![alpha_value; 3]
    };

    (vec![b1, b2, b3], body_alphas)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::Sha3RandomByteStream;

    fn avg_chroma(cols: &[(f64, f64, f64)]) -> f64 {
        cols.iter().map(|(_, a, b)| (a * a + b * b).sqrt()).sum::<f64>() / cols.len() as f64
    }

    fn spec_for_seed(seed: &[u8], phase: f64) -> PaletteSpec {
        let mut rng = Sha3RandomByteStream::new(seed, 100.0, 300.0, 300.0, 1.0);
        resolve_palette_spec(&mut rng, true, phase)
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
    fn test_body_color_sequences_uniform_alpha() {
        let mut rng = Sha3RandomByteStream::new(&[5, 6, 7, 8], 1.0, 1.0, 1.0, 1.0);
        let (colors, alphas) =
            generate_body_color_sequences(&mut rng, 50, 15_000_000, false, false, 0.5);

        assert_eq!(colors.len(), 3);
        assert_eq!(alphas.len(), 3);
        for &a in &alphas {
            assert_eq!(a, 1.0 / 15_000_000.0);
        }
    }

    #[test]
    fn test_body_color_sequences_alpha_variation() {
        let mut rng = Sha3RandomByteStream::new(&[5, 6, 7, 8], 1.0, 1.0, 1.0, 1.0);
        let (_, alphas) = generate_body_color_sequences(&mut rng, 50, 15_000_000, false, true, 0.5);

        assert_eq!(alphas.len(), 3);
        let unique: std::collections::HashSet<u64> = alphas.iter().map(|a| a.to_bits()).collect();
        assert!(unique.len() > 1, "alpha_variation should produce different per-body alphas");

        let base = 1.0 / 15_000_000.0;
        for &alpha in &alphas {
            let multiplier = alpha / base;
            assert!(
                (ALPHA_VARIATION_MIN..=ALPHA_VARIATION_MAX).contains(&multiplier),
                "alpha multiplier {multiplier} outside curated range"
            );
        }
    }

    #[test]
    fn test_alpha_variation_reaches_wide_spread_across_seeds() {
        let mut widest_ratio = 1.0f64;
        for seed in 0u8..48 {
            let mut rng = Sha3RandomByteStream::new(&[seed, 0x12, 0x9A], 1.0, 1.0, 1.0, 1.0);
            let (_, alphas) =
                generate_body_color_sequences(&mut rng, 16, 15_000_000, false, true, 0.5);
            let max = alphas.iter().copied().fold(0.0f64, f64::max);
            let min = alphas.iter().copied().fold(f64::INFINITY, f64::min);
            widest_ratio = widest_ratio.max(max / min);
        }
        assert!(
            widest_ratio > 1.8,
            "continuous alpha variation should produce strong hierarchies, widest={widest_ratio}"
        );
    }

    #[test]
    fn test_color_generation_determinism() {
        let seed = [0x10, 0x00, 0x33];
        let steps = 200;

        let mut rng1 = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
        let (colors1, alphas1) =
            generate_body_color_sequences(&mut rng1, steps, 15_000_000, true, true, 0.37);

        let mut rng2 = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
        let (colors2, alphas2) =
            generate_body_color_sequences(&mut rng2, steps, 15_000_000, true, true, 0.37);

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
    fn test_color_subseed_is_independent_of_parent_rng_position() {
        let seed = [0x44, 0x22, 0x11];
        let mut rng1 = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
        let mut rng2 = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
        for _ in 0..512 {
            rng2.next_byte();
        }

        let (colors1, alphas1) =
            generate_body_color_sequences(&mut rng1, 80, 15_000_000, true, true, 0.41);
        let (colors2, alphas2) =
            generate_body_color_sequences(&mut rng2, 80, 15_000_000, true, true, 0.41);

        assert_eq!(alphas1, alphas2);
        assert_eq!(colors1, colors2);
    }

    #[test]
    fn test_procedural_palettes_cover_all_hue_regions() {
        let mut hue_bins = std::collections::HashSet::new();

        for seed in 0u8..64 {
            let mut rng = Sha3RandomByteStream::new(&[seed, 0xA5, 0x5A], 100.0, 300.0, 300.0, 1.0);
            let (colors, _) = generate_body_color_sequences(
                &mut rng,
                32,
                15_000_000,
                true,
                false,
                f64::from(seed) / 63.0,
            );

            for body_colors in &colors {
                for &(_, a, b) in body_colors.iter().step_by(8) {
                    let hue = b.atan2(a).to_degrees().rem_euclid(HUE_FULL_CIRCLE);
                    hue_bins.insert((hue / 30.0).floor() as u8);
                }
            }
        }

        assert_eq!(hue_bins.len(), 12, "continuous palettes should occupy every hue bin");
    }

    #[test]
    fn test_procedural_palettes_stay_display_p3_safe() {
        for seed in 0u8..24 {
            let mut rng = Sha3RandomByteStream::new(&[0x33, seed, 0x77], 100.0, 300.0, 300.0, 1.0);
            let (colors, _) = generate_body_color_sequences(
                &mut rng,
                64,
                15_000_000,
                true,
                false,
                f64::from(seed) / 23.0,
            );

            for body_colors in &colors {
                for &(l, a, b) in body_colors {
                    let chroma = (a * a + b * b).sqrt();
                    let hue = b.atan2(a).to_degrees().rem_euclid(HUE_FULL_CIRCLE);
                    let max_chroma = max_display_p3_chroma_for_lh(l, hue);
                    assert!((0.34..=0.94).contains(&l), "lightness out of bounds: {l}");
                    assert!(
                        chroma <= max_chroma * 1.000_001,
                        "chroma {chroma} exceeds Display P3 cusp {max_chroma}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_palettes_keep_lightness_hierarchy_and_vividness() {
        let span = |values: &[f64]| {
            values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - values.iter().copied().fold(f64::INFINITY, f64::min)
        };

        for seed in 0u8..32 {
            let mut rng = Sha3RandomByteStream::new(&[0x81, seed, 0x19], 100.0, 300.0, 300.0, 1.0);
            let (colors, _) = generate_body_color_sequences(
                &mut rng,
                96,
                15_000_000,
                true,
                false,
                f64::from(seed) / 31.0,
            );

            let mean_l: Vec<f64> = colors
                .iter()
                .map(|body| body.iter().map(|(l, _, _)| *l).sum::<f64>() / body.len() as f64)
                .collect();
            let mean_c: Vec<f64> = colors
                .iter()
                .map(|body| {
                    body.iter().map(|(_, a, b)| (a * a + b * b).sqrt()).sum::<f64>()
                        / body.len() as f64
                })
                .collect();
            let max_c = mean_c.iter().copied().fold(0.0, f64::max);

            assert!(
                span(&mean_l) > 0.03,
                "lightness hierarchy collapsed for seed {seed}: {mean_l:?}"
            );
            assert!(max_c > 0.05, "palette should stay vivid for seed {seed}: {mean_c:?}");
        }
    }

    #[test]
    fn gate_enforces_min_body_separation_across_many_seeds() {
        for s in 0u32..512 {
            let seed = [(s & 0xff) as u8, (s >> 8) as u8, 0x3D, 0x91];
            let spec = spec_for_seed(&seed, f64::from(s % 97) / 97.0);
            let colors = [
                plan_mean_color(&spec.bodies[0]),
                plan_mean_color(&spec.bodies[1]),
                plan_mean_color(&spec.bodies[2]),
            ];
            let min_distance = oklab_distance(colors[0], colors[1])
                .min(oklab_distance(colors[1], colors[2]))
                .min(oklab_distance(colors[0], colors[2]));
            assert!(
                min_distance >= GATE_MIN_BODY_DISTANCE - 1e-9,
                "seed {s}: gate failed to enforce body separation ({min_distance:.4})"
            );
        }
    }

    #[test]
    fn gate_terminates_within_bounded_attempts() {
        let mut max_attempts = 0usize;
        let mut repaired = 0usize;
        for s in 0u32..1024 {
            let seed = [(s & 0xff) as u8, (s >> 8) as u8, 0x6B, 0x2F];
            let spec = spec_for_seed(&seed, 0.5);
            max_attempts = max_attempts.max(spec.genome.gate_attempts);
            if spec.genome.repaired {
                repaired += 1;
            }
            assert!(spec.genome.gate_attempts <= MAX_PALETTE_ATTEMPTS);
        }
        // The gate should almost always pass by sampling; repair is the
        // last-resort path and must stay rare.
        assert!(repaired * 50 < 1024, "deterministic repair triggered too often: {repaired}/1024");
        assert!(max_attempts >= 1);
    }

    #[test]
    fn dispersion_continuum_covers_monochrome_through_triadic() {
        let mut tight = 0usize;
        let mut wide = 0usize;
        let mut mid = 0usize;
        for s in 0u32..512 {
            let seed = [(s & 0xff) as u8, (s >> 8) as u8, 0x5a, 0xa5];
            let spec = spec_for_seed(&seed, f64::from(s % 97) / 97.0);
            let d = spec.genome.dispersion;
            assert!((8.0..=300.0).contains(&d), "dispersion {d} out of range");
            if d < 30.0 {
                tight += 1;
            } else if d > 200.0 {
                wide += 1;
            } else {
                mid += 1;
            }
        }
        assert!(tight > 0, "expected near-monochrome palettes in the continuum");
        assert!(wide > 0, "expected wide (triadic-like) palettes in the continuum");
        assert!(mid > 0, "expected mid-dispersion palettes in the continuum");
    }

    #[test]
    fn hue_anchor_distribution_is_roughly_uniform() {
        let bins = 12usize;
        let total = 1536u32;
        let mut hist = vec![0usize; bins];
        for s in 0..total {
            let seed = [(s & 0xff) as u8, (s >> 8) as u8, 0xC4, 0x51];
            let spec = spec_for_seed(&seed, 0.5);
            let idx = ((spec.genome.anchor / HUE_FULL_CIRCLE * bins as f64) as usize).min(bins - 1);
            hist[idx] += 1;
        }
        let expected = total as usize / bins;
        for (bin, &count) in hist.iter().enumerate() {
            assert!(
                count > expected / 3 && count < expected * 3,
                "hue anchor bin {bin} deviates from uniform: {count} vs ~{expected} ({hist:?})"
            );
        }
    }

    #[test]
    fn red_green_opposition_stays_rare_after_gating() {
        let mut clash_pairs = 0usize;
        let mut total_pairs = 0usize;
        for s in 0u32..1024 {
            let seed = [(s & 0xff) as u8, (s >> 8) as u8, 0xD1, 0x7A];
            let spec = spec_for_seed(&seed, f64::from(s % 113) / 113.0);
            let hues: Vec<f64> = spec.bodies.iter().map(|body| body.base_hue).collect();
            for i in 0..hues.len() {
                for j in i + 1..hues.len() {
                    total_pairs += 1;
                    if (is_red_sector(hues[i]) && is_green_sector(hues[j]))
                        || (is_green_sector(hues[i]) && is_red_sector(hues[j]))
                    {
                        clash_pairs += 1;
                    }
                }
            }
        }

        let rate = clash_pairs as f64 / total_pairs as f64;
        assert!(rate < 0.02, "red/green opposition pair rate too high after gate: {rate:.3}");
    }

    #[test]
    fn repair_always_produces_a_passing_palette() {
        // Construct degenerate plans (identical colors) and verify repair
        // separates them regardless of the starting hue.
        for hue in [0.0f64, 60.0, 122.0, 200.0, 310.0] {
            let mut genome = PaletteGenome {
                anchor: hue,
                dispersion: 8.0,
                skew: 0.0,
                chroma_peak: 0.30,
                chroma_floor_ratio: 0.5,
                lightness_center: 0.6,
                lightness_span: 0.0,
                hue_journey_scale: 0.0,
                wave_freq: 1.0,
                wave_amp: 5.0,
                accent_strength: 5.0,
                lightness_wave: 0.05,
                chroma_wave: 0.03,
                gate_attempts: MAX_PALETTE_ATTEMPTS,
                repaired: false,
            };
            let plan = BodyColorPlan {
                base_hue: hue,
                target_lightness: 0.6,
                lightness_range: 0.05,
                lightness_wave: 0.05,
                chroma_fraction: 0.15,
                chroma_noise: 0.05,
                chroma_wave: 0.03,
                hue_journey: 0.0,
                phase: 0.0,
                is_dominant: false,
            };
            let mut bodies = [plan, plan, plan];
            repair_palette(&mut genome, &mut bodies);

            assert!(genome.repaired);
            let colors = [
                plan_mean_color(&bodies[0]),
                plan_mean_color(&bodies[1]),
                plan_mean_color(&bodies[2]),
            ];
            let min_distance = oklab_distance(colors[0], colors[1])
                .min(oklab_distance(colors[1], colors[2]))
                .min(oklab_distance(colors[0], colors[2]));
            assert!(
                min_distance >= GATE_MIN_BODY_DISTANCE,
                "repair failed to separate bodies at hue {hue}: {min_distance:.4}"
            );
        }
    }

    #[test]
    fn palette_metadata_reports_fingerprint_and_gate() {
        let mut rng = Sha3RandomByteStream::new(&[0x99, 0x12], 100.0, 300.0, 300.0, 1.0);
        let _ = generate_body_color_sequences(&mut rng, 16, 15_000_000, true, true, 0.5);
        let (fingerprint, gate) = current_palette_metadata();
        assert!(
            fingerprint.starts_with('h'),
            "fingerprint should encode the anchor: {fingerprint}"
        );
        assert!(fingerprint.contains("_d"), "fingerprint should encode dispersion: {fingerprint}");
        assert!(gate.starts_with("gate"), "gate descriptor missing: {gate}");
    }

    #[test]
    fn chroma_boost_raises_population_chroma_peak() {
        // The boost contract is on the genome floor; compare population means
        // of the sampled chroma peak rather than noisy per-gradient chroma.
        // The beauty gate already resamples muddy genomes, so the residual
        // boost effect after gating is modest but must stay directional.
        let total = 512u32;
        let mut boosted_sum = 0.0;
        let mut plain_sum = 0.0;
        for s in 0..total {
            let seed = [(s & 0xff) as u8, (s >> 8) as u8, 0x42];
            let mut rng1 = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
            let mut rng2 = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
            plain_sum += resolve_palette_spec(&mut rng1, false, 0.5).genome.chroma_peak;
            boosted_sum += resolve_palette_spec(&mut rng2, true, 0.5).genome.chroma_peak;
        }
        assert!(
            boosted_sum > plain_sum + 0.005 * f64::from(total),
            "boosted chroma floor should raise the population mean: boosted={} plain={}",
            boosted_sum / f64::from(total),
            plain_sum / f64::from(total),
        );
        // Sanity: chroma usage stays meaningful in generated gradients.
        let mut rng = Sha3RandomByteStream::new(&[7, 7, 7], 1.0, 1.0, 1.0, 1.0);
        let colors = generate_color_gradient_oklab(&mut rng, 64, 0, BASE_HUE_DRIFT, true, 2.6);
        assert!(avg_chroma(&colors) > 0.02, "boosted gradients should stay colorful");
    }

    proptest::proptest! {
        #[test]
        fn proptest_palette_gate_holds_for_arbitrary_seeds(seed in proptest::collection::vec(0u8.., 1..16)) {
            let mut rng = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
            let spec = resolve_palette_spec(&mut rng, true, 0.5);

            // Terminates within bounds.
            proptest::prop_assert!(spec.genome.gate_attempts <= MAX_PALETTE_ATTEMPTS);

            // All mean colors finite and within the displayable envelope.
            for plan in &spec.bodies {
                let (l, a, b) = plan_mean_color(plan);
                proptest::prop_assert!(l.is_finite() && a.is_finite() && b.is_finite());
                let chroma = (a * a + b * b).sqrt();
                let hue = b.atan2(a).to_degrees().rem_euclid(HUE_FULL_CIRCLE);
                let cusp = max_display_p3_chroma_for_lh(l, hue);
                proptest::prop_assert!(chroma <= cusp * 1.000_001);
            }

            // Separation floor holds.
            let colors = [
                plan_mean_color(&spec.bodies[0]),
                plan_mean_color(&spec.bodies[1]),
                plan_mean_color(&spec.bodies[2]),
            ];
            let min_distance = oklab_distance(colors[0], colors[1])
                .min(oklab_distance(colors[1], colors[2]))
                .min(oklab_distance(colors[0], colors[2]));
            proptest::prop_assert!(min_distance >= GATE_MIN_BODY_DISTANCE - 1e-9);
        }

        #[test]
        fn proptest_generated_sequences_are_finite(seed in proptest::collection::vec(0u8.., 1..12)) {
            let mut rng = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
            let (colors, alphas) =
                generate_body_color_sequences(&mut rng, 24, 15_000_000, true, true, 0.5);
            for body in &colors {
                for &(l, a, b) in body {
                    proptest::prop_assert!(l.is_finite() && a.is_finite() && b.is_finite());
                }
            }
            for &alpha in &alphas {
                proptest::prop_assert!(alpha.is_finite() && alpha > 0.0);
            }
        }
    }
}
