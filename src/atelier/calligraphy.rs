//! Tidal Calligraphy: open, tapered strokes whose spines are actual body history.
//!
//! These are designed ribbons, not a freely moving cloth sheet. Every central
//! surface point lies on its body's source curve. Cross-sectional shaping and
//! fine fibers give that curve a textile interpretation without changing it.
//! Detail is anchored to cumulative source arc length, never to a free-running
//! animation clock, so the material does not boil as the history window moves.

use super::{BodySample, Material, OrbitSeries, Scene, SilkResult, Strand, Triangle, V3, Vertex};
use serde::{Deserialize, Serialize};

const FIBER_TINTS: usize = 16;

/// Geometry and material controls for three calligraphic history ribbons.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CalligraphyConfig {
    /// Shared recent-history window, as a fraction of the recorded source.
    pub history_fraction: f64,
    /// Optional shared world-space arc budget; zero keeps the fixed-time window.
    pub history_arc_length: f64,
    /// Longitudinal intervals for a fully grown hero ribbon.
    pub segments: usize,
    /// Across-width intervals; use an even count to include the exact spine.
    pub across_segments: usize,
    /// Nominal full width in the source's fixed four-unit world normalization.
    pub ribbon_width: f64,
    /// Per-body width hierarchy, preserving original body identities.
    pub body_widths: [f64; 3],
    /// Fraction of a ribbon's length devoted to its long fading tail.
    pub tail_taper: f64,
    /// Fraction devoted to the short, pointed leading tip.
    pub head_taper: f64,
    /// Extra breadth at slow passages, and narrowing at fast passages.
    pub fan_strength: f64,
    /// Additional narrowing in strongly curved source passages.
    pub turn_slimming: f64,
    /// Additional narrowing when the bodies are near one another.
    pub proximity_slimming: f64,
    /// Maximum full width relative to the visible spine's arc length.
    pub arc_width_limit: f64,
    /// Maximum product of local geometric curvature and full ribbon width.
    pub curvature_width_limit: f64,
    /// Smooth opening-rate limit near the trailing and leading tips.
    pub width_slope_limit: f64,
    /// Rounded shoulder length relative to nominal width/opening rate.
    pub width_transition: f64,
    /// Gentle transverse arch depth, expressed as a fraction of full width.
    pub camber: f64,
    /// Shallow transverse pleat depth, as a fraction of full width.
    pub corrugation_amplitude: f64,
    /// Number of smooth transverse corrugation cycles across the full width.
    pub corrugation_count: f64,
    /// Source arc length over which the diagonal pleat phase advances one turn.
    pub corrugation_pitch: f64,
    /// Fixed per-body roll around transported tangents; never an auto-spin.
    pub roll_degrees: [f64; 3],
    /// Gentle roll amplitude anchored to source arc length, in degrees.
    pub roll_amplitude_degrees: f64,
    /// Source arc length for one broad roll cycle.
    pub roll_pitch: f64,
    /// Fixed phase offset of each body's broad roll, in degrees.
    pub roll_phase_degrees: [f64; 3],
    /// Optical-depth multiplier for the delicate web between visible fibers.
    pub surface_opacity: f64,
    /// Fine interior companion fibers per body.
    pub fiber_count: usize,
    /// Fine fringe fibers on each side of each ribbon.
    pub fringe_count: usize,
    /// Longitudinal intervals used by each fine fiber.
    pub fiber_segments: usize,
    /// Nominal world-space fiber radius before tapering.
    pub fiber_radius: f64,
    /// Independent mean optical depth of the individual fibers.
    pub fiber_optical_depth: f64,
    /// Independent reflected-light multiplier for fibers.
    pub fiber_brightness: f64,
    /// Independent grazing reflection strength of fibers.
    pub fiber_sheen: f64,
    /// Independent roughness of the fine fiber highlights.
    pub fiber_roughness: f64,
    /// Optional radiance multiplier for explicitly luminous fiber treatments.
    pub fiber_emission: f64,
    /// Restrained, stable warm/cool tint variation between fibers.
    pub fiber_tint_variation: f64,
    /// Stable fractional radius variation between individual fibers.
    pub fiber_radius_variation: f64,
    /// Smooth monotone bunching and spreading within the ribbon width.
    pub fiber_fan_strength: f64,
    /// Source arc length for one internal fiber-fan cycle.
    pub fiber_fan_pitch: f64,
    /// Fine ordered wandering relative to the available inter-fiber spacing.
    pub fiber_wander: f64,
    /// Maximum removed tail fraction for independently tapered fibers.
    pub fiber_length_variation: f64,
    /// Additional fringe reach relative to the ribbon's half-width.
    pub fringe_spread: f64,
    /// Extra fringe arch depth relative to full width.
    pub fringe_lift: f64,
    /// Strength of smooth fringe reach and lift changes along the source arc.
    pub fringe_breathing: f64,
    /// Number of differently phased, fine fringe packets per side.
    pub fringe_bundles: usize,
    /// Source arc length for one broad fringe fan cycle.
    pub fringe_pitch: f64,
    /// Stable artistic-detail seed; does not change source paths or their clocks.
    pub detail_seed: u64,
    /// Radius of the tiny source-position glints at the leading tips.
    pub tip_glint_radius: f64,
    /// Add one restrained, non-overlapping older history window per body.
    pub echoes: bool,
    /// Width multiplier for optional older strokes.
    pub echo_width: f64,
    /// Optical-depth multiplier for optional older strokes.
    pub echo_opacity: f64,
    /// Three distinct, coordinated fabric materials in original body order.
    pub materials: [Material; 3],
}

impl Default for CalligraphyConfig {
    fn default() -> Self {
        Self {
            history_fraction: 0.14,
            history_arc_length: 0.0,
            segments: 1024,
            across_segments: 32,
            ribbon_width: 0.26,
            body_widths: [1.0, 0.90, 0.78],
            tail_taper: 0.42,
            head_taper: 0.10,
            fan_strength: 0.42,
            turn_slimming: 0.50,
            proximity_slimming: 0.28,
            arc_width_limit: 0.12,
            curvature_width_limit: 0.60,
            width_slope_limit: 0.65,
            width_transition: 0.28,
            camber: 0.065,
            corrugation_amplitude: 0.024,
            corrugation_count: 2.75,
            corrugation_pitch: 2.1,
            roll_degrees: [12.0, -24.0, 34.0],
            roll_amplitude_degrees: 25.0,
            roll_pitch: 8.0,
            roll_phase_degrees: [0.0, 120.0, 240.0],
            surface_opacity: 0.45,
            fiber_count: 64,
            fringe_count: 8,
            fiber_segments: 384,
            fiber_radius: 0.00045,
            fiber_optical_depth: 0.65,
            fiber_brightness: 1.08,
            fiber_sheen: 0.78,
            fiber_roughness: 0.22,
            fiber_emission: 0.0,
            fiber_tint_variation: 0.18,
            fiber_radius_variation: 0.35,
            fiber_fan_strength: 0.28,
            fiber_fan_pitch: 3.2,
            fiber_wander: 0.45,
            fiber_length_variation: 0.18,
            fringe_spread: 0.55,
            fringe_lift: 0.025,
            fringe_breathing: 0.65,
            fringe_bundles: 3,
            fringe_pitch: 3.6,
            detail_seed: 7331,
            tip_glint_radius: 0.0014,
            echoes: false,
            echo_width: 0.22,
            echo_opacity: 0.18,
            materials: default_materials(),
        }
    }
}

impl CalligraphyConfig {
    /// Airier, longer strokes with finer webs and a denser filament fringe.
    #[must_use]
    pub fn gossamer() -> Self {
        let mut config = Self {
            history_fraction: 0.21,
            ribbon_width: 0.14,
            tail_taper: 0.55,
            head_taper: 0.14,
            fan_strength: 0.56,
            camber: 0.045,
            corrugation_amplitude: 0.017,
            fiber_count: 96,
            fringe_count: 12,
            fiber_radius: 0.00031,
            fringe_spread: 0.80,
            ..Self::default()
        };
        for material in &mut config.materials {
            material.optical_depth *= 0.62;
            material.sheen = 0.56;
        }
        config
    }

    /// Shorter, broader brush fans with restrained fibers and larger open turns.
    #[must_use]
    pub fn silk_fans() -> Self {
        Self {
            history_fraction: 0.085,
            ribbon_width: 0.38,
            tail_taper: 0.48,
            head_taper: 0.16,
            fan_strength: 0.58,
            arc_width_limit: 0.14,
            camber: 0.10,
            corrugation_amplitude: 0.035,
            corrugation_count: 1.75,
            fiber_count: 48,
            fringe_count: 6,
            ..Self::default()
        }
    }
}

fn default_materials() -> [Material; 3] {
    [
        Material {
            front_color: V3::new(0.78, 0.73, 0.62),
            back_color: V3::new(0.19, 0.16, 0.31),
            optical_depth: V3::new(0.40, 0.45, 0.54),
            roughness: 0.29,
            anisotropy: 0.82,
            sheen: 0.48,
            fiber_frequency: 420.0,
            fiber_strength: 0.09,
            ..Material::default()
        },
        Material {
            front_color: V3::new(0.54, 0.66, 0.73),
            back_color: V3::new(0.12, 0.20, 0.29),
            optical_depth: V3::new(0.43, 0.34, 0.29),
            roughness: 0.31,
            anisotropy: 0.84,
            sheen: 0.46,
            fiber_frequency: 460.0,
            fiber_strength: 0.08,
            ..Material::default()
        },
        Material {
            front_color: V3::new(0.69, 0.47, 0.29),
            back_color: V3::new(0.24, 0.13, 0.22),
            optical_depth: V3::new(0.26, 0.40, 0.57),
            roughness: 0.30,
            anisotropy: 0.80,
            sheen: 0.43,
            fiber_frequency: 390.0,
            fiber_strength: 0.085,
            ..Material::default()
        },
    ]
}

#[derive(Clone, Copy)]
struct Section {
    position: V3,
    tangent: V3,
    across: V3,
    normal: V3,
    arc: f64,
    width: f64,
    age_coordinate: f64,
}

struct Ribbon {
    sections: Vec<Section>,
    nominal_width: f64,
    segment_fraction: f64,
}

/// Build a high-detail frame using only available, genuine source history.
///
/// `time` is the current recorded-source fraction, not an animation phase. At
/// zero, verified source prelude is used when available; otherwise three tiny
/// source-position glints introduce the strokes. No earlier samples are invented,
/// wrapped, or drawn from the future.
pub fn scene(source: &OrbitSeries, time: f64, config: &CalligraphyConfig) -> SilkResult<Scene> {
    validate(time, config)?;
    let mut result = Scene { materials: config.materials.to_vec(), ..Scene::default() };
    for material in &mut result.materials {
        material.optical_depth *= config.surface_opacity;
    }
    for material in &config.materials {
        let mean_depth =
            (material.optical_depth.x + material.optical_depth.y + material.optical_depth.z) / 3.0;
        for variant in 0..FIBER_TINTS {
            let variation = (2.0 * variant as f64 / (FIBER_TINTS - 1) as f64 - 1.0)
                * config.fiber_tint_variation;
            let tint =
                V3::new(1.0 + 0.55 * variation, 1.0 - 0.08 * variation, 1.0 - 0.45 * variation);
            let mut fiber = material.clone();
            fiber.front_color = material.front_color.hadamard(tint) * config.fiber_brightness;
            fiber.back_color = material.back_color.hadamard(tint) * config.fiber_brightness;
            fiber.optical_depth = if mean_depth > 1e-12 {
                material.optical_depth * (config.fiber_optical_depth / mean_depth)
            } else {
                V3::new(
                    config.fiber_optical_depth,
                    config.fiber_optical_depth,
                    config.fiber_optical_depth,
                )
            };
            fiber.roughness = config.fiber_roughness;
            fiber.anisotropy = 0.94;
            fiber.sheen = config.fiber_sheen;
            fiber.emission = fiber.front_color * config.fiber_emission;
            fiber.fiber_strength = 0.0;
            result.materials.push(fiber);
        }
    }
    if config.echoes {
        for material in &config.materials {
            let mut echo = material.clone();
            echo.optical_depth *= config.echo_opacity;
            echo.sheen *= 0.55;
            result.materials.push(echo);
        }
    }
    for body in 0..3 {
        let current = source
            .sample_body(body, time)
            .ok_or("Calligraphy could not sample the current source position")?;
        let start = history_start(source, body, time, config)?;
        if let Some(ribbon) = make_ribbon(source, body, start, time, 1.0, config)? {
            append_surface(&mut result, &ribbon, body, config)?;
            append_fibers(&mut result, &ribbon, body, config);
        }
        append_glint(
            &mut result,
            current,
            config.tip_glint_radius,
            3 + body * FIBER_TINTS + FIBER_TINTS / 2,
        );
        if config.echoes && start > source.history_start_fraction() {
            let end = start;
            let start = history_start(source, body, end, config)?;
            if let Some(ribbon) = make_ribbon(source, body, start, end, config.echo_width, config)?
            {
                append_surface(&mut result, &ribbon, 3 + 3 * FIBER_TINTS + body, config)?;
            }
        }
    }
    Ok(result)
}

fn validate(time: f64, config: &CalligraphyConfig) -> SilkResult<()> {
    if !time.is_finite() || !(0.0..=1.0).contains(&time) {
        return Err("Calligraphy time must lie within the recorded source interval [0,1]".into());
    }
    if !(0.0..=1.0).contains(&config.history_fraction) || config.history_fraction == 0.0 {
        return Err("Calligraphy history_fraction must be positive and at most one".into());
    }
    if config.segments < 4
        || config.across_segments < 2
        || !config.across_segments.is_multiple_of(2)
        || config.fiber_segments < 4
        || config.fringe_bundles == 0
    {
        return Err(
            "Calligraphy needs at least four length intervals and an even cross-section".into()
        );
    }
    let requested_vertices = config
        .segments
        .checked_add(1)
        .and_then(|count| count.checked_mul(config.across_segments.checked_add(1)?))
        .and_then(|count| count.checked_mul(if config.echoes { 6 } else { 3 }));
    if requested_vertices.is_none_or(|count| u32::try_from(count).is_err()) {
        return Err("Calligraphy geometry exceeds the scene index range".into());
    }
    let positive = [
        config.ribbon_width,
        config.tail_taper,
        config.head_taper,
        config.arc_width_limit,
        config.curvature_width_limit,
        config.width_slope_limit,
        config.width_transition,
        config.corrugation_pitch,
        config.roll_pitch,
        config.fiber_fan_pitch,
        config.fringe_pitch,
        config.echo_width,
    ];
    if positive.iter().any(|value| !value.is_finite() || *value <= 0.0)
        || config.body_widths.iter().any(|value| !value.is_finite() || *value <= 0.0)
        || config.roll_degrees.iter().any(|value| !value.is_finite())
        || config.roll_phase_degrees.iter().any(|value| !value.is_finite())
        || !config.roll_amplitude_degrees.is_finite()
    {
        return Err("Calligraphy widths, shaping scales and rolls must be finite".into());
    }
    let nonnegative = [
        config.camber,
        config.corrugation_amplitude,
        config.corrugation_count,
        config.fiber_radius,
        config.fringe_spread,
        config.fringe_lift,
        config.tip_glint_radius,
        config.echo_opacity,
        config.history_arc_length,
        config.surface_opacity,
        config.fiber_optical_depth,
        config.fiber_brightness,
        config.fiber_sheen,
        config.fiber_emission,
    ];
    if nonnegative.iter().any(|value| !value.is_finite() || *value < 0.0)
        || config.tail_taper > 1.0
        || config.head_taper > 1.0
        || [config.fan_strength, config.turn_slimming, config.proximity_slimming]
            .iter()
            .any(|value| !value.is_finite() || !(0.0..1.0).contains(value))
    {
        return Err(
            "Calligraphy shaping and material strengths are outside their valid range".into()
        );
    }
    if [
        config.fiber_tint_variation,
        config.fiber_radius_variation,
        config.fiber_wander,
        config.fiber_length_variation,
        config.fringe_breathing,
        config.fiber_roughness,
    ]
    .iter()
    .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || !config.fiber_fan_strength.is_finite()
        || !(0.0..0.5).contains(&config.fiber_fan_strength)
        || config.fiber_length_variation >= 0.8
    {
        return Err(
            "Calligraphy fiber variations must be finite, bounded, and retain an open width order"
                .into(),
        );
    }
    for material in &config.materials {
        for color in
            [material.front_color, material.back_color, material.optical_depth, material.emission]
        {
            if !color.is_finite() || color.x < 0.0 || color.y < 0.0 || color.z < 0.0 {
                return Err("Calligraphy materials require finite nonnegative linear colors".into());
            }
        }
        if !material.roughness.is_finite()
            || !(0.0..=1.0).contains(&material.roughness)
            || !material.anisotropy.is_finite()
            || !(0.0..=1.0).contains(&material.anisotropy)
            || !material.sheen.is_finite()
            || material.sheen < 0.0
            || !material.fiber_frequency.is_finite()
            || material.fiber_frequency < 0.0
            || !material.fiber_strength.is_finite()
            || !(0.0..=1.0).contains(&material.fiber_strength)
        {
            return Err(
                "Calligraphy material parameters must be finite and physically bounded".into()
            );
        }
    }
    Ok(())
}

fn smoothstep(value: f64) -> f64 {
    let t = value.clamp(0.0, 1.0);
    t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
}

fn smooth_cap(value: f64, limit: f64) -> f64 {
    if value <= 0.0 || limit <= 0.0 {
        return 0.0;
    }
    let (smaller, larger) = if value <= limit { (value, limit) } else { (limit, value) };
    let ratio = smaller / larger;
    let square = ratio * ratio;
    let fourth = square * square;
    // (a^-8+b^-8)^(-1/8): smooth for positive inputs, never exceeds either cap.
    smaller / (1.0 + fourth * fourth).sqrt().sqrt().sqrt()
}

fn rounded_opening(distance: f64, slope: f64, rounding: f64) -> f64 {
    // C1 at the tip, with derivative bounded by slope and no straight cone corner.
    slope * distance * distance / (distance + rounding)
}

fn taper(coordinate: f64, config: &CalligraphyConfig) -> f64 {
    smoothstep(coordinate / config.tail_taper) * smoothstep((1.0 - coordinate) / config.head_taper)
}

fn history_start(
    source: &OrbitSeries,
    body: usize,
    end: f64,
    config: &CalligraphyConfig,
) -> SilkResult<f64> {
    let earliest = (end - config.history_fraction).max(source.history_start_fraction());
    if config.history_arc_length == 0.0 || end <= earliest {
        return Ok(earliest);
    }
    let head = source.sample_body(body, end).ok_or("Unavailable Calligraphy arc-budget head")?;
    let first =
        source.sample_body(body, earliest).ok_or("Unavailable Calligraphy arc-budget tail")?;
    if head.arc_length - first.arc_length <= config.history_arc_length {
        return Ok(earliest);
    }
    let target = head.arc_length - config.history_arc_length;
    let (mut low, mut high) = (earliest, end);
    for _ in 0..40 {
        let middle = (low + high) * 0.5;
        let sample =
            source.sample_body(body, middle).ok_or("Unavailable Calligraphy arc-budget sample")?;
        if sample.arc_length < target {
            low = middle;
        } else {
            high = middle;
        }
    }
    Ok(high)
}

fn make_ribbon(
    source: &OrbitSeries,
    body: usize,
    start: f64,
    end: f64,
    width_multiplier: f64,
    config: &CalligraphyConfig,
) -> SilkResult<Option<Ribbon>> {
    if end <= start {
        return Ok(None);
    }
    let first = source.sample_body(body, start).ok_or("Unavailable Calligraphy history start")?;
    let last = source.sample_body(body, end).ok_or("Unavailable Calligraphy history end")?;
    let length = last.arc_length - first.arc_length;
    if length <= 1e-9 {
        return Ok(None);
    }
    let growth = if config.history_arc_length > 0.0 {
        (length / config.history_arc_length).clamp(0.0, 1.0)
    } else {
        ((end - start) / config.history_fraction).clamp(0.0, 1.0)
    };
    let segments = ((config.segments as f64 * growth).ceil() as usize).max(4);
    let nominal_width = config.ribbon_width * config.body_widths[body] * width_multiplier;
    let mut sections = Vec::with_capacity(segments + 1);
    for index in 0..=segments {
        let u = index as f64 / segments as f64;
        let fraction = if index == segments { end } else { start + (end - start) * u };
        let sample = source.sample_body(body, fraction).ok_or("Unavailable Calligraphy history")?;
        let roll_phase = std::f64::consts::TAU * sample.arc_length / config.roll_pitch
            + config.roll_phase_degrees[body].to_radians();
        let roll = config.roll_degrees[body] + config.roll_amplitude_degrees * roll_phase.sin();
        let (sin_roll, cos_roll) = roll.to_radians().sin_cos();
        let fan = 1.0 + config.fan_strength * (1.0 - 2.0 * sample.speed);
        let turn = 1.0 - config.turn_slimming * sample.curvature;
        let clearance = 1.0 - config.proximity_slimming * sample.proximity;
        let width =
            smooth_cap(nominal_width * fan * turn * clearance, length * config.arc_width_limit);
        sections.push(Section {
            position: sample.position,
            tangent: sample.tangent,
            across: sample.normal * cos_roll + sample.binormal * sin_roll,
            normal: sample.binormal * cos_roll - sample.normal * sin_roll,
            arc: sample.arc_length,
            width,
            age_coordinate: u,
        });
    }
    // A wide ribbon cannot follow a tight turn cleanly. Limit its width using
    // world-space curvature, independently of normalized artistic measurements.
    for index in 0..sections.len() {
        let left = index.saturating_sub(1);
        let right = (index + 1).min(sections.len() - 1);
        let distance = sections[right].arc - sections[left].arc;
        if distance > 1e-10 {
            let curvature = (sections[right].tangent - sections[left].tangent).length() / distance;
            if curvature > 1e-8 {
                sections[index].width =
                    smooth_cap(sections[index].width, config.curvature_width_limit / curvature);
            }
        }
    }
    // Rounded opening profiles replace the former piecewise-linear forward/back
    // minima. Conservative soft caps preserve clearance while removing visible
    // shoulder corners, independent of tessellation density.
    let rounding = smooth_cap(nominal_width, length * config.arc_width_limit)
        * config.width_transition
        / config.width_slope_limit;
    for section in &mut sections {
        let from_tail = (section.arc - first.arc_length).max(0.0);
        let from_head = (last.arc_length - section.arc).max(0.0);
        let tailed = section.width * taper(section.age_coordinate, config);
        section.width = smooth_cap(
            smooth_cap(tailed, rounded_opening(from_tail, config.width_slope_limit, rounding)),
            rounded_opening(from_head, config.width_slope_limit, rounding),
        );
    }
    Ok(Some(Ribbon { sections, nominal_width, segment_fraction: growth }))
}

fn surface_point(section: Section, across: f64, config: &CalligraphyConfig) -> V3 {
    if across == 0.0 {
        return section.position;
    }
    let envelope = (1.0 - across * across).max(0.0);
    let phase = std::f64::consts::TAU * section.arc / config.corrugation_pitch;
    let corrugation = config.corrugation_amplitude
        * envelope
        * envelope
        * ((std::f64::consts::PI * config.corrugation_count * across + phase).cos() - phase.cos());
    let depth = section.width * (-config.camber * across * across + corrugation);
    // At across==0 both shaping terms are exactly zero: the actual source curve
    // remains an explicit, inspectable centerline of the finished surface.
    section.position + section.across * (0.5 * section.width * across) + section.normal * depth
}

fn append_surface(
    scene: &mut Scene,
    ribbon: &Ribbon,
    material: usize,
    config: &CalligraphyConfig,
) -> SilkResult<()> {
    let base = scene.vertices.len();
    let columns = config.across_segments + 1;
    let count = ribbon.sections.len() * columns;
    scene.vertices.try_reserve(count)?;
    for &section in &ribbon.sections {
        for column in 0..columns {
            let v = 2.0 * column as f64 / config.across_segments as f64 - 1.0;
            scene.vertices.push(Vertex {
                position: surface_point(section, v, config),
                normal: section.normal,
                tangent: section.tangent,
                uv: [section.arc, (v + 1.0) * 0.5],
            });
        }
    }
    for row in 0..ribbon.sections.len() {
        for column in 0..columns {
            let index = base + row * columns + column;
            let previous = base + row.saturating_sub(1) * columns + column;
            let next = base + (row + 1).min(ribbon.sections.len() - 1) * columns + column;
            let left = base + row * columns + column.saturating_sub(1);
            let right = base + row * columns + (column + 1).min(columns - 1);
            let length_direction =
                scene.vertices[next].position - scene.vertices[previous].position;
            let width_direction = scene.vertices[right].position - scene.vertices[left].position;
            let normal = length_direction.cross(width_direction).normalized();
            if normal.length_squared() > 0.5 {
                scene.vertices[index].normal = normal;
            }
            let tangent = length_direction.normalized();
            if tangent.length_squared() > 0.5 {
                scene.vertices[index].tangent = tangent;
            }
        }
    }
    scene.triangles.try_reserve((ribbon.sections.len() - 1) * config.across_segments * 2)?;
    for row in 0..ribbon.sections.len() - 1 {
        for column in 0..config.across_segments {
            let a = base + row * columns + column;
            let b = a + columns;
            for indices in [[a, b, b + 1], [a, b + 1, a + 1]] {
                let [first, second, third] = indices.map(|index| scene.vertices[index].position);
                if (second - first).cross(third - first).length_squared() > 1e-26 {
                    scene.triangles.push(Triangle {
                        indices: [
                            u32::try_from(indices[0])?,
                            u32::try_from(indices[1])?,
                            u32::try_from(indices[2])?,
                        ],
                        material,
                    });
                }
            }
        }
    }
    Ok(())
}

fn interpolate_section(ribbon: &Ribbon, coordinate: f64) -> Section {
    let index = coordinate * (ribbon.sections.len() - 1) as f64;
    let first = index.floor() as usize;
    let second = (first + 1).min(ribbon.sections.len() - 1);
    let t = index - first as f64;
    let a = ribbon.sections[first];
    let b = ribbon.sections[second];
    let tangent = a.tangent.lerp(b.tangent, t).normalized();
    let across_hint = a.across.lerp(b.across, t);
    let across = (across_hint - tangent * across_hint.dot(tangent)).normalized();
    Section {
        position: a.position.lerp(b.position, t),
        tangent,
        across,
        normal: tangent.cross(across).normalized(),
        arc: a.arc * (1.0 - t) + b.arc * t,
        width: interpolated_width(ribbon, first, t),
        age_coordinate: coordinate,
    }
}

fn width_derivative(ribbon: &Ribbon, index: usize) -> f64 {
    if index == 0 || index + 1 == ribbon.sections.len() {
        return 0.0;
    }
    let left = ribbon.sections[index].width - ribbon.sections[index - 1].width;
    let right = ribbon.sections[index + 1].width - ribbon.sections[index].width;
    if left * right <= 0.0 { 0.0 } else { 2.0 * left * right / (left + right) }
}

fn interpolated_width(ribbon: &Ribbon, index: usize, t: f64) -> f64 {
    let next = (index + 1).min(ribbon.sections.len() - 1);
    let a = ribbon.sections[index].width;
    let b = ribbon.sections[next].width;
    let da = width_derivative(ribbon, index);
    let db = width_derivative(ribbon, next);
    let t2 = t * t;
    let t3 = t2 * t;
    ((2.0 * t3 - 3.0 * t2 + 1.0) * a
        + (t3 - 2.0 * t2 + t) * da
        + (-2.0 * t3 + 3.0 * t2) * b
        + (t3 - t2) * db)
        .max(0.0)
}

fn mix_bits(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn unit_variation(seed: u64) -> f64 {
    (mix_bits(seed) >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

struct FiberSpec {
    across: f64,
    fringe_rank: f64,
    phase: f64,
    group_phase: f64,
    tail_offset: f64,
    radius_scale: f64,
    material: usize,
}

fn fiber_spec(
    body: usize,
    ordinal: usize,
    across: f64,
    fringe_rank: f64,
    packet: usize,
    config: &CalligraphyConfig,
) -> FiberSpec {
    let body_seed = config.detail_seed ^ (body as u64).wrapping_mul(0xd1b5_4a32_d192_ed03);
    let seed = body_seed ^ (ordinal as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
    FiberSpec {
        across,
        fringe_rank,
        phase: std::f64::consts::TAU * unit_variation(seed),
        group_phase: std::f64::consts::TAU
            * unit_variation(body_seed ^ (packet as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)),
        tail_offset: config.fiber_length_variation
            * unit_variation(seed ^ 0x1327)
            * (if fringe_rank > 0.0 { 1.0 } else { 0.35 }),
        radius_scale: 1.0
            + config.fiber_radius_variation * (2.0 * unit_variation(seed ^ 0x7193) - 1.0),
        material: 3 + body * FIBER_TINTS + (mix_bits(seed ^ 0x5637) % FIBER_TINTS as u64) as usize,
    }
}

fn append_fibers(scene: &mut Scene, ribbon: &Ribbon, body: usize, config: &CalligraphyConfig) {
    let segments =
        ((config.fiber_segments as f64 * ribbon.segment_fraction).ceil() as usize).max(4);
    for fiber in 0..config.fiber_count {
        let across = 2.0 * (fiber + 1) as f64 / (config.fiber_count + 1) as f64 - 1.0;
        let spec = fiber_spec(body, fiber, across, 0.0, 0, config);
        append_fiber(scene, ribbon, segments, &spec, config);
    }
    for (side_index, side) in [-1.0, 1.0].into_iter().enumerate() {
        for fiber in 0..config.fringe_count {
            let rank = (fiber + 1) as f64 / config.fringe_count as f64;
            let across = side * (1.035 + config.fringe_spread * rank);
            let ordinal = config.fiber_count + side_index * config.fringe_count + fiber;
            let packet = 1 + side_index * config.fringe_bundles + fiber % config.fringe_bundles;
            let spec = fiber_spec(body, ordinal, across, rank, packet, config);
            append_fiber(scene, ribbon, segments, &spec, config);
        }
    }
}

fn append_fiber(
    scene: &mut Scene,
    ribbon: &Ribbon,
    segments: usize,
    spec: &FiberSpec,
    config: &CalligraphyConfig,
) {
    let mut points = Vec::with_capacity(segments + 1);
    let mut radii = Vec::with_capacity(segments + 1);
    for index in 0..=segments {
        let u = index as f64 / segments as f64;
        let coordinate = spec.tail_offset + (1.0 - spec.tail_offset) * u;
        let section = interpolate_section(ribbon, coordinate);
        let width_ratio = (section.width / ribbon.nominal_width).clamp(0.0, 1.6);
        let own_taper = smoothstep(u / 0.14) * smoothstep((1.0 - u) / 0.065);
        let radius = config.fiber_radius
            * width_ratio
            * (1.0 - spec.fringe_rank * 0.35)
            * spec.radius_scale
            * own_taper;
        let age_arch = 4.0 * section.age_coordinate * (1.0 - section.age_coordinate);
        let phase = std::f64::consts::TAU * section.arc / config.fringe_pitch + spec.group_phase;
        let breath = 1.0 - config.fringe_breathing * (0.5 + 0.5 * phase.sin());
        let across = if spec.fringe_rank > 0.0 {
            spec.across.signum() * (1.035 + config.fringe_spread * spec.fringe_rank * breath)
        } else {
            let wave = (std::f64::consts::TAU * section.arc / config.fiber_fan_pitch
                + spec.group_phase)
                .sin();
            let fan = spec.across
                + config.fiber_fan_strength
                    * wave
                    * spec.across
                    * (1.0 - spec.across * spec.across);
            let spacing = 2.0 / (config.fiber_count + 1) as f64;
            let allowance =
                (1.0 - 2.0 * config.fiber_fan_strength) * spacing * 0.35 * config.fiber_wander;
            fan + allowance
                * (std::f64::consts::TAU * section.arc / config.fiber_fan_pitch * 1.31 + spec.phase)
                    .sin()
        };
        let lift = radius * 1.4
            + section.width
                * config.fringe_lift
                * spec.fringe_rank
                * age_arch
                * (0.35 + 0.65 * (0.5 + 0.5 * (phase + 1.1).sin()));
        points.push(surface_point(section, across, config) + section.normal * lift);
        radii.push(radius);
    }
    if radii.iter().any(|&radius| radius > 0.0) {
        scene.strands.push(Strand { points, radii, material: spec.material });
    }
}

fn append_glint(scene: &mut Scene, sample: BodySample, radius: f64, material: usize) {
    if radius <= 0.0 {
        return;
    }
    for axis in [sample.tangent, sample.normal, sample.binormal] {
        scene.strands.push(Strand {
            points: vec![
                sample.position - axis * radius * 2.0,
                sample.position,
                sample.position + axis * radius * 2.0,
            ],
            radii: vec![0.0, radius, 0.0],
            material,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silk::OrbitData;

    fn source() -> OrbitSeries {
        let samples = (0..=512)
            .map(|index| {
                let t = f64::from(index) / 512.0;
                std::array::from_fn(|body| {
                    let phase = t * 5.0 + body as f64 * 1.7;
                    V3::new(phase.cos(), phase.sin(), (phase * 0.43).sin() * 0.35)
                })
            })
            .collect();
        OrbitSeries::new(&OrbitData {
            seed: "geometry-test".to_owned(),
            dt: 0.01,
            masses: [1.0; 3],
            samples,
            provenance: serde_json::json!({"synthetic_unit_fixture":true}),
        })
        .unwrap()
    }

    fn small_config() -> CalligraphyConfig {
        CalligraphyConfig {
            segments: 40,
            across_segments: 8,
            fiber_segments: 24,
            fiber_count: 6,
            fringe_count: 2,
            ..CalligraphyConfig::default()
        }
    }

    #[test]
    fn source_curve_is_the_exact_surface_spine_and_tips_are_tapered() {
        let source = source();
        let config = small_config();
        let end = 0.65;
        let start = end - config.history_fraction;
        let ribbon = make_ribbon(&source, 1, start, end, 1.0, &config).unwrap().unwrap();
        assert_eq!(ribbon.sections.first().unwrap().width, 0.0);
        assert_eq!(ribbon.sections.last().unwrap().width, 0.0);
        for (index, &section) in ribbon.sections.iter().enumerate() {
            let t = index as f64 / (ribbon.sections.len() - 1) as f64;
            let fraction =
                if index + 1 == ribbon.sections.len() { end } else { start + (end - start) * t };
            let actual = source.sample_body(1, fraction).unwrap().position;
            assert!((section.position - actual).length() < 1e-14);
            assert_eq!(surface_point(section, 0.0, &config), section.position);
        }
    }

    #[test]
    fn early_frames_do_not_borrow_future_history_or_form_wide_patches() {
        let source = source();
        let config = small_config();
        let beginning = scene(&source, 0.0, &config).unwrap();
        assert!(beginning.triangles.is_empty());
        assert_eq!(beginning.strands.len(), 9);
        for body in 0..3 {
            let actual = source.sample_body(body, 0.0).unwrap().position;
            for glint in &beginning.strands[body * 3..body * 3 + 3] {
                assert_eq!(glint.points[1], actual);
            }
        }
        let end = 0.0002;
        let ribbon = make_ribbon(&source, 0, 0.0, end, 1.0, &config).unwrap().unwrap();
        let length = ribbon.sections.last().unwrap().arc - ribbon.sections[0].arc;
        assert!(
            ribbon.sections.iter().all(|section| section.width <= length * config.arc_width_limit)
        );
        assert_eq!(ribbon.sections[0].position, source.sample_body(0, 0.0).unwrap().position);
        assert_eq!(
            ribbon.sections.last().unwrap().position,
            source.sample_body(0, end).unwrap().position
        );
    }

    #[test]
    fn geometry_is_finite_repeatable_and_material_indices_are_valid() {
        let source = source();
        let config = CalligraphyConfig { echoes: true, ..small_config() };
        let a = scene(&source, 0.81, &config).unwrap();
        let b = scene(&source, 0.81, &config).unwrap();
        assert_eq!(a.vertices.len(), b.vertices.len());
        assert_eq!(a.triangles.len(), b.triangles.len());
        for (first, second) in a.vertices.iter().zip(&b.vertices) {
            assert_eq!(first.position, second.position);
            assert_eq!(first.normal, second.normal);
            assert!(
                first.position.is_finite() && first.normal.is_finite() && first.tangent.is_finite()
            );
            assert!((first.normal.length() - 1.0).abs() < 1e-8);
            assert!((first.tangent.length() - 1.0).abs() < 1e-8);
        }
        for triangle in &a.triangles {
            assert!(triangle.material < a.materials.len());
            assert!(triangle.indices.iter().all(|&index| (index as usize) < a.vertices.len()));
        }
        for strand in &a.strands {
            assert_eq!(strand.points.len(), strand.radii.len());
            assert!(strand.points.iter().all(|point| point.is_finite()));
            assert!(strand.radii.iter().all(|radius| radius.is_finite() && *radius >= 0.0));
            assert!(strand.material < a.materials.len());
        }
    }

    #[test]
    fn presets_change_the_form_and_invalid_time_is_rejected() {
        let source = source();
        let standard = small_config();
        let airy = CalligraphyConfig {
            segments: standard.segments,
            across_segments: standard.across_segments,
            fiber_segments: standard.fiber_segments,
            ..CalligraphyConfig::gossamer()
        };
        let a = scene(&source, 0.6, &standard).unwrap();
        let b = scene(&source, 0.6, &airy).unwrap();
        assert_ne!(a.vertices[0].position, b.vertices[0].position);
        assert!(scene(&source, -0.01, &standard).is_err());
        assert!(scene(&source, 1.01, &standard).is_err());
        assert!(scene(&source, f64::NAN, &standard).is_err());
    }

    #[test]
    fn smooth_width_caps_are_conservative_without_a_derivative_corner() {
        for value in [1e-6, 0.01, 0.1, 1.0, 10.0] {
            for limit in [1e-6, 0.01, 0.1, 1.0, 10.0] {
                let width = smooth_cap(value, limit);
                assert!(width > 0.0 && width <= value && width <= limit);
            }
        }
        let h = 1e-6;
        let at = smooth_cap(1.0, 1.0);
        let left = (at - smooth_cap(1.0 - h, 1.0)) / h;
        let right = (smooth_cap(1.0 + h, 1.0) - at) / h;
        assert!((left - right).abs() < 1e-5);
        assert_eq!(rounded_opening(0.0, 0.7, 0.1), 0.0);
        assert!(rounded_opening(1e-7, 0.7, 0.1) / 1e-7 < 1e-5);
    }

    #[test]
    fn arc_budget_preserves_the_current_head_and_continuous_history() {
        let source = source();
        let config =
            CalligraphyConfig { history_fraction: 0.5, history_arc_length: 1.0, ..small_config() };
        let end = 0.7;
        for body in 0..3 {
            let start = history_start(&source, body, end, &config).unwrap();
            assert!(start >= end - config.history_fraction && start < end);
            let head = source.sample_body(body, end).unwrap();
            let tail = source.sample_body(body, start).unwrap();
            assert!((head.arc_length - tail.arc_length - 1.0).abs() < 1e-9);
            let ribbon = make_ribbon(&source, body, start, end, 1.0, &config).unwrap().unwrap();
            assert_eq!(ribbon.sections.last().unwrap().position, head.position);
            let next = history_start(&source, body, end + 1e-5, &config).unwrap();
            assert!(next > start && (next - start) < 1e-3);
        }
    }

    #[test]
    fn fiber_optics_are_independent_of_the_veil_and_have_stable_variation() {
        let source = source();
        let config =
            CalligraphyConfig { surface_opacity: 0.0, fiber_optical_depth: 0.8, ..small_config() };
        let a = scene(&source, 0.6, &config).unwrap();
        let b = scene(&source, 0.6, &CalligraphyConfig { surface_opacity: 0.9, ..config.clone() })
            .unwrap();
        assert_eq!(a.materials[0].optical_depth, V3::ZERO);
        assert_ne!(b.materials[0].optical_depth, V3::ZERO);
        assert_eq!(a.materials[3].optical_depth, b.materials[3].optical_depth);
        let depth = a.materials[3].optical_depth;
        assert!(((depth.x + depth.y + depth.z) / 3.0 - 0.8).abs() < 1e-12);
        assert_ne!(a.materials[3].front_color, a.materials[3 + FIBER_TINTS - 1].front_color);
        let repeat = scene(&source, 0.6, &config).unwrap();
        for (first, second) in a.strands.iter().zip(&repeat.strands) {
            assert_eq!(first.points, second.points);
            assert_eq!(first.radii, second.radii);
            assert_eq!(first.material, second.material);
        }
        let varied =
            scene(&source, 0.6, &CalligraphyConfig { detail_seed: 1234, ..config }).unwrap();
        assert!(
            a.vertices
                .iter()
                .zip(&varied.vertices)
                .all(|(first, second)| first.position == second.position)
        );
        assert_ne!(a.strands[0].points, varied.strands[0].points);
    }
}
