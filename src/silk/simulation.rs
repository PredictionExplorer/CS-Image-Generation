//! A fixed-pattern cloth driven by the unmodified three-body trajectory.
//!
//! Stretch, shear and dihedral bending use compliant position constraints. Contact
//! uses swept surface primitives rather than vertex particles. Unsafe steps are
//! retried with smaller intervals; an unresolved configuration is an error.

mod contact;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::{ClothBake, FrameStats, Mesh, OrbitData, SilkResult, V3};

/// How a body's position constrains a small region of material.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AttachmentMode {
    /// Preserve each patch's original orientation as well as its position.
    Rigid,
    /// Constrain the area-weighted center, leaving local rotation unconstrained.
    #[default]
    Centroid,
}

/// Whether hidden pre-roll advances the source trajectory before visible playback.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrerollMode {
    /// Preserve the original timing: hidden pre-roll consumes the source prefix.
    #[default]
    AdvanceSource,
    /// Hold the first source positions until the first visible frame.
    HoldFirst,
}

/// Physical and sampling settings for a reusable cloth bake.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SimulationConfig {
    /// Rigid pads preserve legacy studies; centroid pads can turn naturally.
    pub attachment_mode: AttachmentMode,
    /// Approximate number of radial mesh edges across the circular sheet.
    pub subdivisions: usize,
    /// Number of visible frames, including the first settled frame.
    pub frames: u32,
    /// Visible frames per second.
    pub fps: u32,
    /// Fixed baseline simulation intervals per output frame.
    pub substeps: u32,
    /// Ordered constraint sweeps per simulation interval.
    pub iterations: u32,
    /// Bounded groups of four contact-only polishing sweeps after material solving.
    pub contact_iterations: u32,
    /// Enforce the fixed material-distance upper bound to each attachment center.
    pub long_range_attachments: bool,
    /// Allowed distance ratio in long-range attachment bounds.
    pub attachment_stretch_limit: f64,
    /// Material distance relative to the maximum required support separation.
    pub slack: f64,
    /// Maximum body-pair separation after one fixed world-space scaling.
    pub world_size: f64,
    /// Compliance of warp and weft edge constraints.
    pub stretch_compliance: f64,
    /// Compliance of diagonal, shear-resisting edge constraints.
    pub shear_compliance: f64,
    /// Compliance of signed dihedral bending constraints.
    pub bend_compliance: f64,
    /// Velocity damping coefficient, in inverse seconds.
    pub damping: f64,
    /// Additional face-normal air resistance, in inverse seconds.
    pub air_drag: f64,
    /// Constant downward acceleration in world units per second squared.
    pub gravity: f64,
    /// Minimum separation between contacting cloth mid-surfaces.
    pub thickness: f64,
    /// Coulomb coefficient used by contact displacement friction.
    pub friction: f64,
    /// Enable vertex–triangle and edge–edge self-contact.
    pub self_collision: bool,
    /// Hidden time to gather the flat rest pattern onto the initial supports.
    pub settle_seconds: f64,
    /// Hidden cloth preparation time after gathering onto the initial supports.
    pub preroll_seconds: f64,
    /// Advance the source during pre-roll, or preserve the complete visible interval.
    pub preroll_mode: PrerollMode,
    /// Small, static initial drape as a fraction of world size; breaks flat symmetry.
    pub initial_drape: f64,
    /// Beginning of the source interval, as a fraction of its sample range.
    pub orbit_start: f64,
    /// End of the source interval, as a fraction of its sample range.
    pub orbit_end: f64,
    /// Maximum recursive timestep bisections before a bake fails explicitly.
    pub max_step_halvings: u32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            attachment_mode: AttachmentMode::Centroid,
            subdivisions: 64,
            frames: 240,
            fps: 30,
            substeps: 8,
            iterations: 32,
            contact_iterations: 12,
            long_range_attachments: true,
            attachment_stretch_limit: 1.03,
            slack: 1.12,
            world_size: 3.5,
            stretch_compliance: 1e-8,
            shear_compliance: 2e-8,
            bend_compliance: 1.0,
            damping: 0.35,
            air_drag: 0.5,
            gravity: 0.25,
            thickness: 0.008,
            friction: 0.06,
            self_collision: true,
            settle_seconds: 1.5,
            preroll_seconds: 1.0,
            preroll_mode: PrerollMode::AdvanceSource,
            initial_drape: 0.04,
            orbit_start: 0.0,
            orbit_end: 1.0,
            max_step_halvings: 5,
        }
    }
}

#[derive(Clone)]
struct Stretch {
    vertices: [usize; 2],
    rest: f64,
    compliance: f64,
}

struct Bend {
    vertices: [usize; 4],
    rest_angle: f64,
}

struct AttachmentLimit {
    vertex: usize,
    body: usize,
    maximum: f64,
}

#[derive(Clone)]
struct Pin {
    vertex: usize,
    body: usize,
    offset: V3,
    weight: f64,
}

struct Pattern {
    attachment_mode: AttachmentMode,
    mesh: Mesh,
    stretch: Vec<Stretch>,
    bend: Vec<Bend>,
    attachment_limits: Vec<AttachmentLimit>,
    edges: Vec<[usize; 2]>,
    pins: Vec<Pin>,
    inverse_mass: Vec<f64>,
    rest_supports: [V3; 3],
}

struct Driver<'a> {
    orbit: &'a OrbitData,
    config: &'a SimulationConfig,
    center: V3,
    scale: f64,
    duration: f64,
    rest_supports: [V3; 3],
}

impl Driver<'_> {
    fn source_fraction(&self, motion_time: f64) -> f64 {
        if self.config.preroll_mode == PrerollMode::HoldFirst {
            let visible_time = motion_time - self.config.preroll_seconds;
            if motion_time <= self.config.preroll_seconds {
                return self.config.orbit_start;
            }
            // Use the same endpoint expression as the frame loop, avoiding a
            // subtraction-rounding loss of the final source sample.
            if motion_time >= self.config.preroll_seconds + self.duration {
                return self.config.orbit_end;
            }
            return self.config.orbit_start
                + (self.config.orbit_end - self.config.orbit_start)
                    * (visible_time / self.duration).clamp(0.0, 1.0);
        }
        self.config.orbit_start
            + (self.config.orbit_end - self.config.orbit_start)
                * (motion_time / self.duration).clamp(0.0, 1.0)
    }

    fn supports(&self, time: f64) -> [V3; 3] {
        let fraction = self.source_fraction(time.max(0.0));
        let source = sample_orbit(self.orbit, fraction);
        let target = source.map(|position| (position - self.center) * self.scale);
        if time < 0.0 && self.config.settle_seconds > 0.0 {
            let t = (1.0 + time / self.config.settle_seconds).clamp(0.0, 1.0);
            let ease = t * t * t * (10.0 + t * (-15.0 + 6.0 * t));
            std::array::from_fn(|body| self.rest_supports[body].lerp(target[body], ease))
        } else {
            target
        }
    }
}

struct State {
    positions: Vec<V3>,
    velocities: Vec<V3>,
}

/// Simulate a continuous cloth and save its visible geometry for later rendering.
///
/// All supports share one constant space/time transform. The hidden settling
/// phase changes the supports, never the rest pattern. In `hold_first` mode,
/// pre-roll keeps the first support positions fixed and the visible frames span
/// the complete requested source interval, including both endpoint samples.
pub fn bake(orbit: &OrbitData, config: &SimulationConfig) -> SilkResult<ClothBake> {
    validate(orbit, config)?;
    let center = orbit.samples[0]
        .iter()
        .zip(orbit.masses)
        .fold(V3::ZERO, |sum, (&position, mass)| sum + position * mass)
        / orbit.masses.iter().sum::<f64>();
    let mut maximum_pair_distance: f64 = 0.0;
    let first = ((orbit.samples.len() - 1) as f64 * config.orbit_start).floor() as usize;
    let last = ((orbit.samples.len() - 1) as f64 * config.orbit_end).ceil() as usize;
    for sample in &orbit.samples[first..=last] {
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            maximum_pair_distance = maximum_pair_distance.max((sample[a] - sample[b]).length());
        }
    }
    if maximum_pair_distance < 1e-9 {
        return Err("silk requires distinct moving supports".into());
    }
    let scale = config.world_size / maximum_pair_distance;
    let initial = sample_orbit(orbit, config.orbit_start).map(|p| (p - center) * scale);
    let pattern = make_pattern(initial, config)?;
    let visible_duration = f64::from(config.frames - 1) / f64::from(config.fps);
    let duration = match config.preroll_mode {
        PrerollMode::AdvanceSource => config.preroll_seconds + visible_duration,
        PrerollMode::HoldFirst => visible_duration,
    };
    let driver =
        Driver { orbit, config, center, scale, duration, rest_supports: pattern.rest_supports };
    let mut state = State {
        positions: initial_drape(&pattern, config),
        velocities: vec![V3::ZERO; pattern.mesh.positions.len()],
    };
    set_pins(
        &mut state.positions,
        &pattern.pins,
        driver.supports(-config.settle_seconds),
        config.attachment_mode,
    );
    let timestep = 1.0 / (f64::from(config.fps) * f64::from(config.substeps));
    let mut hidden_stats = FrameStats::default();
    tracing::info!(
        vertices = pattern.mesh.positions.len(),
        triangles = pattern.mesh.triangles.len(),
        settle_seconds = config.settle_seconds,
        preroll_seconds = config.preroll_seconds,
        "Silk hidden preparation begins"
    );
    simulate_range(
        &mut state,
        &pattern,
        &driver,
        -config.settle_seconds,
        config.preroll_seconds,
        timestep,
        &mut hidden_stats,
    )?;
    let settled = measure(
        &state.positions,
        &pattern,
        driver.supports(config.preroll_seconds),
        hidden_stats.contacts,
    );
    tracing::info!(
        max_stretch = settled.max_stretch,
        contacts = settled.contacts,
        "Silk settled; visible playback begins"
    );

    let mut frames = Vec::with_capacity(config.frames as usize);
    let mut stats = Vec::with_capacity(config.frames as usize);
    frames.push(state.positions.clone());
    stats.push(measure(&state.positions, &pattern, driver.supports(config.preroll_seconds), 0));
    for frame in 1..config.frames {
        let start = config.preroll_seconds + f64::from(frame - 1) / f64::from(config.fps);
        let end = config.preroll_seconds + f64::from(frame) / f64::from(config.fps);
        let mut frame_stats = FrameStats::default();
        simulate_range(&mut state, &pattern, &driver, start, end, timestep, &mut frame_stats)?;
        frames.push(state.positions.clone());
        stats.push(measure(&state.positions, &pattern, driver.supports(end), frame_stats.contacts));
        if frame.is_multiple_of((config.frames / 10).max(1)) || frame + 1 == config.frames {
            let latest = stats.last().expect("visible frame statistics were just appended");
            tracing::info!(
                frame = frame + 1,
                frames = config.frames,
                playback_time = end,
                max_stretch = latest.max_stretch,
                contacts = latest.contacts,
                "Silk baking"
            );
        }
    }
    let mut stretch_ratios: Vec<f64> = frames
        .iter()
        .flat_map(|positions| {
            pattern.stretch.iter().map(|edge| {
                (positions[edge.vertices[0]] - positions[edge.vertices[1]]).length() / edge.rest
            })
        })
        .collect();
    stretch_ratios.sort_unstable_by(f64::total_cmp);
    let strain_summary = serde_json::json!({
        "maximum":stretch_ratios.last(),
        "percentile_95":stretch_ratios[(stretch_ratios.len()-1)*95/100],
        "fraction_above_1_10":stretch_ratios.iter().filter(|&&ratio|ratio>1.10).count() as f64/stretch_ratios.len() as f64,
        "fraction_above_1_20":stretch_ratios.iter().filter(|&&ratio|ratio>1.20).count() as f64/stretch_ratios.len() as f64,
    });
    let recipe = serde_json::json!({
        "version": "tidal-silk-xpbd-v2",
        "seed": orbit.seed,
        "configuration": config,
        "source_provenance": orbit.provenance,
        "source_dt": orbit.dt,
        "source_samples": orbit.samples.len(),
        "world_center": center,
        "world_scale": scale,
        "source_start_fraction": config.orbit_start,
        "first_visible_source_fraction": driver.source_fraction(config.preroll_seconds),
        "source_end_fraction": config.orbit_end,
        "last_visible_source_fraction":driver.source_fraction(config.preroll_seconds+visible_duration),
        "preroll_mode":config.preroll_mode,
        "visible_motion_duration_seconds":visible_duration,
        "encoded_duration_seconds":f64::from(config.frames)/f64::from(config.fps),
        "source_time_per_playback_second": (config.orbit_end-config.orbit_start)
            * (orbit.samples.len()-1) as f64 * orbit.dt / duration,
        "settling": "quintic support gathering; fixed material metric; hidden",
        "contact": "swept BVH, vertex-triangle and edge-edge proximity, conservative advancement, bounded step bisection",
        "vertices": pattern.mesh.positions.len(),
        "triangles": pattern.mesh.triangles.len(),
        "attachment_vertices": pattern.pins.len(),
        "strain_summary":strain_summary,
    });
    Ok(ClothBake { mesh: pattern.mesh, frames, fps: config.fps, stats, recipe })
}

fn validate(orbit: &OrbitData, config: &SimulationConfig) -> SilkResult<()> {
    if orbit.samples.len() < 2 || !orbit.dt.is_finite() || orbit.dt <= 0.0 {
        return Err("silk needs at least two uniformly timed source samples".into());
    }
    if orbit.samples.iter().flatten().any(|position| !position.is_finite())
        || orbit.masses.iter().any(|mass| !mass.is_finite() || *mass <= 0.0)
    {
        return Err("silk source positions and positive masses must be finite".into());
    }
    if !(6..=256).contains(&config.subdivisions)
        || config.frames < 2
        || config.fps == 0
        || config.substeps == 0
        || config.iterations == 0
        || config.contact_iterations == 0
        || config.max_step_halvings > 12
    {
        return Err("invalid silk mesh, frame, or solver settings".into());
    }
    let nonnegative = [
        config.stretch_compliance,
        config.shear_compliance,
        config.bend_compliance,
        config.damping,
        config.air_drag,
        config.gravity,
        config.friction,
        config.settle_seconds,
        config.preroll_seconds,
        config.initial_drape,
    ];
    if nonnegative.iter().any(|value| !value.is_finite() || *value < 0.0)
        || !config.slack.is_finite()
        || config.slack < 1.01
        || !config.attachment_stretch_limit.is_finite()
        || config.attachment_stretch_limit < 1.0
        || !config.world_size.is_finite()
        || config.world_size <= 0.0
        || !config.thickness.is_finite()
        || config.thickness <= 0.0
        || !config.orbit_start.is_finite()
        || !config.orbit_end.is_finite()
        || config.orbit_start < 0.0
        || config.orbit_end > 1.0
        || config.orbit_start >= config.orbit_end
    {
        return Err("invalid silk material or source interval".into());
    }
    Ok(())
}

fn sample_orbit(orbit: &OrbitData, fraction: f64) -> [V3; 3] {
    let index = fraction.clamp(0.0, 1.0) * (orbit.samples.len() - 1) as f64;
    let first = index.floor() as usize;
    let second = (first + 1).min(orbit.samples.len() - 1);
    std::array::from_fn(|body| {
        orbit.samples[first][body].lerp(orbit.samples[second][body], index - first as f64)
    })
}

fn initial_drape(pattern: &Pattern, config: &SimulationConfig) -> Vec<V3> {
    let [a, b, c] = pattern.rest_supports;
    let normal = (b - a).cross(c - a).normalized();
    let pad_scale = config.world_size * 0.18;
    let mut positions: Vec<_> = pattern
        .mesh
        .positions
        .iter()
        .zip(&pattern.mesh.uv)
        .map(|(&position, &[u, v])| {
            let x = 2.0 * u - 1.0;
            let y = 2.0 * v - 1.0;
            let envelope = (1.0 - x * x - y * y).max(0.0);
            let distance = pattern
                .rest_supports
                .iter()
                .map(|&support| (position - support).length())
                .fold(f64::INFINITY, f64::min);
            let t = (distance / pad_scale).clamp(0.0, 1.0);
            let pad_falloff = t * t * (3.0 - 2.0 * t);
            position
                + normal
                    * (config.world_size
                        * config.initial_drape
                        * envelope
                        * pad_falloff
                        * (1.0 + 0.2 * x))
        })
        .collect();
    set_pins(&mut positions, &pattern.pins, pattern.rest_supports, config.attachment_mode);
    positions
}

fn make_pattern(supports: [V3; 3], config: &SimulationConfig) -> SilkResult<Pattern> {
    let n = config.subdivisions;
    let mut x_axis = (supports[2] - supports[1]).normalized();
    if x_axis.length_squared() < 0.5 {
        x_axis = V3::new(1.0, 0.0, 0.0);
    }
    let up = supports[0] - (supports[1] + supports[2]) * 0.5;
    let mut y_axis = (up - x_axis * up.dot(x_axis)).normalized();
    if y_axis.length_squared() < 0.5 {
        let reference =
            if x_axis.y.abs() < 0.8 { V3::new(0.0, 1.0, 0.0) } else { V3::new(0.0, 0.0, 1.0) };
        y_axis = (reference - x_axis * reference.dot(x_axis)).normalized();
    }
    let center = (supports[0] + supports[1] + supports[2]) / 3.0;
    let radius = config.world_size * config.slack / (3.0_f64.sqrt() * 0.70);
    let rings = n / 2;
    let mut positions = vec![center];
    let mut uv = vec![[0.5, 0.5]];
    let mut starts = vec![0];
    for ring in 1..=rings {
        starts.push(positions.len());
        for vertex in 0..6 * ring {
            let angle = std::f64::consts::TAU * vertex as f64 / (6 * ring) as f64;
            let (sin, cos) = angle.sin_cos();
            let u = ring as f64 / rings as f64 * cos;
            let v = ring as f64 / rings as f64 * sin;
            positions.push(center + x_axis * (u * radius) + y_axis * (v * radius));
            uv.push([(u + 1.0) * 0.5, (v + 1.0) * 0.5]);
        }
    }
    let mut triangles = Vec::new();
    for vertex in 0..6 {
        triangles.push([0, u32::try_from(1 + vertex)?, u32::try_from(1 + (vertex + 1) % 6)?]);
    }
    // Merge adjacent rings in angular order; integer comparisons keep the
    // topology independent of trigonometric rounding at aligned radial edges.
    for ring in 1..rings {
        let inner_count = 6 * ring;
        let outer_count = 6 * (ring + 1);
        let (mut inner, mut outer) = (0, 0);
        while inner < inner_count || outer < outer_count {
            let a = starts[ring] + inner % inner_count;
            let b = starts[ring + 1] + outer % outer_count;
            let c = if inner < inner_count
                && (outer == outer_count || (inner + 1) * outer_count <= (outer + 1) * inner_count)
            {
                inner += 1;
                starts[ring] + inner % inner_count
            } else {
                outer += 1;
                starts[ring + 1] + outer % outer_count
            };
            triangles.push([u32::try_from(a)?, u32::try_from(b)?, u32::try_from(c)?]);
        }
    }
    // Three interior pads leave a generous continuous hem outside the supports.
    let ideal = [
        center + y_axis * (radius * 0.70),
        center + x_axis * (-radius * 0.70 * 3.0_f64.sqrt() * 0.5) - y_axis * (radius * 0.35),
        center + x_axis * (radius * 0.70 * 3.0_f64.sqrt() * 0.5) - y_axis * (radius * 0.35),
    ];
    let anchors: [usize; 3] = ideal.map(|target| {
        positions
            .iter()
            .enumerate()
            .min_by(|(ia, a), (ib, b)| {
                (**a - target)
                    .length_squared()
                    .total_cmp(&(**b - target).length_squared())
                    .then(ia.cmp(ib))
            })
            .map_or(0, |(index, _)| index)
    });
    let minimum_spacing = [(0, 1), (1, 2), (2, 0)]
        .into_iter()
        .map(|(a, b)| (positions[anchors[a]] - positions[anchors[b]]).length())
        .fold(f64::INFINITY, f64::min);
    let expansion = config.world_size * config.slack / minimum_spacing;
    for position in &mut positions {
        *position = center + (*position - center) * expansion;
    }
    let mut rest_supports = anchors.map(|index| positions[index]);
    let pad_radius = (radius * 0.065).max(2.0 * radius / n as f64 * 1.15) * expansion;
    let mut pins = Vec::new();
    for (vertex, &position) in positions.iter().enumerate() {
        if let Some(body) =
            (0..3).find(|&body| (position - rest_supports[body]).length() <= pad_radius)
        {
            pins.push(Pin { vertex, body, offset: position - rest_supports[body], weight: 0.0 });
        }
    }
    let mut masses = vec![0.0; positions.len()];
    let mut adjacency: BTreeMap<[usize; 2], Vec<usize>> = BTreeMap::new();
    for triangle in &triangles {
        let [a, b, c] = triangle.map(|index| index as usize);
        let area = (positions[b] - positions[a]).cross(positions[c] - positions[a]).length() * 0.5;
        for vertex in [a, b, c] {
            masses[vertex] += area / 3.0;
        }
        for (edge, opposite) in [([a, b], c), ([b, c], a), ([c, a], b)] {
            let key = [edge[0].min(edge[1]), edge[0].max(edge[1])];
            adjacency.entry(key).or_default().push(opposite);
        }
    }
    if masses.iter().any(|mass| *mass <= 1e-12) {
        return Err("silk pattern contains an unconnected or degenerate vertex".into());
    }
    let mut inverse_mass: Vec<f64> =
        masses.iter().map(|mass| if *mass > 1e-12 { 1.0 / mass } else { 0.0 }).collect();
    let mut patch_mass = [0.0; 3];
    for pin in &pins {
        patch_mass[pin.body] += masses[pin.vertex];
    }
    for pin in &mut pins {
        pin.weight = masses[pin.vertex] / patch_mass[pin.body];
    }
    if config.attachment_mode == AttachmentMode::Rigid {
        for pin in &pins {
            inverse_mass[pin.vertex] = 0.0;
        }
    } else {
        rest_supports = std::array::from_fn(|body| {
            pins.iter()
                .filter(|pin| pin.body == body)
                .fold(V3::ZERO, |sum, pin| sum + positions[pin.vertex] * pin.weight)
        });
        for pin in &mut pins {
            pin.offset = positions[pin.vertex] - rest_supports[pin.body];
        }
    }
    let mut stretch = Vec::with_capacity(adjacency.len());
    let mut bend = Vec::new();
    let mut edges = Vec::with_capacity(adjacency.len());
    for (vertices, opposites) in adjacency {
        let du = (uv[vertices[0]][0] - uv[vertices[1]][0]).abs();
        let dv = (uv[vertices[0]][1] - uv[vertices[1]][1]).abs();
        let squared = du * du + dv * dv;
        let diagonal_weight = 4.0 * du * du * dv * dv / (squared * squared);
        stretch.push(Stretch {
            vertices,
            rest: (positions[vertices[0]] - positions[vertices[1]]).length(),
            compliance: config.stretch_compliance * (1.0 - diagonal_weight)
                + config.shear_compliance * diagonal_weight,
        });
        edges.push(vertices);
        if opposites.len() == 2 {
            let indices = [vertices[0], vertices[1], opposites[0], opposites[1]];
            if let Some((angle, _)) = bending_geometry(indices.map(|index| positions[index])) {
                bend.push(Bend { vertices: indices, rest_angle: angle });
            }
        }
    }
    let mut attachment_limits = Vec::new();
    if config.long_range_attachments {
        for (vertex, &mass) in inverse_mass.iter().enumerate() {
            if mass > 0.0 && !pins.iter().any(|pin| pin.vertex == vertex) {
                for (body, &support) in rest_supports.iter().enumerate() {
                    attachment_limits.push(AttachmentLimit {
                        vertex,
                        body,
                        maximum: (positions[vertex] - support).length()
                            * config.attachment_stretch_limit,
                    });
                }
            }
        }
    }
    Ok(Pattern {
        attachment_mode: config.attachment_mode,
        mesh: Mesh { positions, triangles, uv },
        stretch,
        bend,
        attachment_limits,
        edges,
        pins,
        inverse_mass,
        rest_supports,
    })
}

fn simulate_range(
    state: &mut State,
    pattern: &Pattern,
    driver: &Driver<'_>,
    start: f64,
    end: f64,
    timestep: f64,
    stats: &mut FrameStats,
) -> SilkResult<()> {
    if end <= start {
        return Ok(());
    }
    let count = ((end - start) / timestep).ceil().max(1.0) as u32;
    for interval in 0..count {
        let a = start + (end - start) * f64::from(interval) / f64::from(count);
        let b = start + (end - start) * f64::from(interval + 1) / f64::from(count);
        advance(state, pattern, driver, a, b, 0, stats)?;
    }
    Ok(())
}

fn advance(
    state: &mut State,
    pattern: &Pattern,
    driver: &Driver<'_>,
    start: f64,
    end: f64,
    depth: u32,
    stats: &mut FrameStats,
) -> SilkResult<()> {
    match try_step(state, pattern, driver, end, end - start) {
        Ok(contacts) => {
            stats.contacts += contacts;
            Ok(())
        }
        Err(reason) => {
            if depth >= driver.config.max_step_halvings {
                return Err(format!(
                    "silk could not resolve step at {end:.6}s after {depth} bisections: {reason}"
                )
                .into());
            }
            let middle = (start + end) * 0.5;
            advance(state, pattern, driver, start, middle, depth + 1, stats)?;
            advance(state, pattern, driver, middle, end, depth + 1, stats)
        }
    }
}

fn try_step(
    state: &mut State,
    pattern: &Pattern,
    driver: &Driver<'_>,
    end: f64,
    h: f64,
) -> Result<usize, &'static str> {
    let config = driver.config;
    let mut normals = vec![V3::ZERO; state.positions.len()];
    for triangle in &pattern.mesh.triangles {
        let [a, b, c] = triangle.map(|index| index as usize);
        let normal = (state.positions[b] - state.positions[a])
            .cross(state.positions[c] - state.positions[a]);
        for index in [a, b, c] {
            normals[index] += normal;
        }
    }
    let mut next = state.positions.clone();
    let attenuation = 1.0 / (1.0 + config.damping * h);
    for index in 0..next.len() {
        if pattern.inverse_mass[index] > 0.0 {
            let normal = normals[index].normalized();
            let velocity = state.velocities[index] * attenuation;
            let normal_velocity = normal * velocity.dot(normal);
            let drag = normal_velocity * (config.air_drag * h / (1.0 + config.air_drag * h));
            next[index] += (velocity - drag) * h + V3::new(0.0, -config.gravity, 0.0) * (h * h);
        }
    }
    let targets = driver.supports(end);
    set_pins(&mut next, &pattern.pins, targets, config.attachment_mode);
    let mut contacts = Vec::new();
    let mut stretch_lambda = vec![0.0; pattern.stretch.len()];
    let mut bend_lambda = vec![0.0; pattern.bend.len()];
    for _ in 0..config.iterations {
        for (constraint, lambda) in pattern.stretch.iter().zip(&mut stretch_lambda) {
            solve_stretch(&mut next, &pattern.inverse_mass, constraint, lambda, h);
        }
        for (constraint, lambda) in pattern.bend.iter().zip(&mut bend_lambda) {
            solve_bend(
                &mut next,
                &pattern.inverse_mass,
                constraint,
                lambda,
                config.bend_compliance,
                h,
            );
        }
        for constraint in &pattern.attachment_limits {
            solve_attachment_limit(&mut next, constraint, targets);
        }
        // Material constraints themselves can fold one region through another.
        // Rebuild from the corrected surface, not only the initial predictor.
        if config.self_collision {
            contacts = contact::generate(
                &state.positions,
                &next,
                &pattern.mesh.triangles,
                &pattern.edges,
                &pattern.inverse_mass,
                config.thickness,
            )?;
        }
        for constraint in &mut contacts {
            constraint.solve(
                &mut next,
                &state.positions,
                &pattern.inverse_mass,
                config.thickness,
                config.friction,
            );
        }
        set_pins(&mut next, &pattern.pins, targets, config.attachment_mode);
    }
    // Dense stacks need their own convergence budget. Ending immediately after
    // one contact sweep allows a later pair to push an earlier one through its
    // neighbor, even though every individual projection was valid.
    if config.self_collision {
        for _ in 0..config.contact_iterations {
            if contact::has_clearance(
                &next,
                &pattern.mesh.triangles,
                &pattern.edges,
                config.thickness * 0.55,
            )? {
                break;
            }
            contacts = contact::generate(
                &state.positions,
                &next,
                &pattern.mesh.triangles,
                &pattern.edges,
                &pattern.inverse_mass,
                config.thickness,
            )?;
            for _ in 0..4 {
                for constraint in &mut contacts {
                    constraint.solve(
                        &mut next,
                        &state.positions,
                        &pattern.inverse_mass,
                        config.thickness,
                        config.friction,
                    );
                }
                set_pins(&mut next, &pattern.pins, targets, config.attachment_mode);
            }
        }
    }
    if next.iter().any(|point| !point.is_finite()) {
        return Err("non-finite positions");
    }
    let maximum_stretch = pattern
        .stretch
        .iter()
        .map(|edge| (next[edge.vertices[0]] - next[edge.vertices[1]]).length() / edge.rest)
        .fold(1.0, f64::max);
    if maximum_stretch > 1.65 {
        return Err("structural stretch exceeds safety threshold");
    }
    if config.self_collision
        && !contact::safe_step(
            &state.positions,
            &next,
            &pattern.mesh.triangles,
            &pattern.edges,
            &pattern.inverse_mass,
            config.thickness,
        )?
    {
        return Err("unresolved surface crossing");
    }
    for (velocity, (&new, &old)) in
        state.velocities.iter_mut().zip(next.iter().zip(&state.positions))
    {
        *velocity = (new - old) / h;
    }
    state.positions = next;
    Ok(contacts.len())
}

fn set_pins(positions: &mut [V3], pins: &[Pin], supports: [V3; 3], mode: AttachmentMode) {
    if mode == AttachmentMode::Rigid {
        for pin in pins {
            positions[pin.vertex] = supports[pin.body] + pin.offset;
        }
    } else {
        for (body, &target) in supports.iter().enumerate() {
            let center = pins
                .iter()
                .filter(|pin| pin.body == body)
                .fold(V3::ZERO, |sum, pin| sum + positions[pin.vertex] * pin.weight);
            let translation = target - center;
            for pin in pins.iter().filter(|pin| pin.body == body) {
                positions[pin.vertex] += translation;
            }
        }
    }
}

fn solve_attachment_limit(positions: &mut [V3], constraint: &AttachmentLimit, supports: [V3; 3]) {
    let difference = positions[constraint.vertex] - supports[constraint.body];
    let length = difference.length();
    if length <= constraint.maximum {
        return;
    }
    positions[constraint.vertex] -= difference * ((length - constraint.maximum) / length);
}

fn solve_stretch(
    positions: &mut [V3],
    inverse_mass: &[f64],
    constraint: &Stretch,
    lambda: &mut f64,
    h: f64,
) {
    let [a, b] = constraint.vertices;
    let difference = positions[a] - positions[b];
    let length = difference.length();
    if length < 1e-12 {
        return;
    }
    let alpha = constraint.compliance / (h * h);
    let denominator = inverse_mass[a] + inverse_mass[b] + alpha;
    if denominator < 1e-12 {
        return;
    }
    let change = (-(length - constraint.rest) - alpha * (*lambda)) / denominator;
    *lambda += change;
    let direction = difference / length;
    positions[a] += direction * (inverse_mass[a] * change);
    positions[b] -= direction * (inverse_mass[b] * change);
}

fn bending_geometry(points: [V3; 4]) -> Option<(f64, [V3; 4])> {
    let [a, b, c, d] = points;
    let edge = b - a;
    let length_squared = edge.length_squared();
    let n1 = edge.cross(c - a);
    let n2 = (d - a).cross(edge);
    let n1_squared = n1.length_squared();
    let n2_squared = n2.length_squared();
    if length_squared < 1e-16 || n1_squared < 1e-18 || n2_squared < 1e-18 {
        return None;
    }
    let length = length_squared.sqrt();
    let first = n1 / n1_squared.sqrt();
    let second = n2 / n2_squared.sqrt();
    let angle = second.cross(first).dot(edge / length).atan2(first.dot(second));
    let qc = n1 * (length / n1_squared);
    let qd = n2 * (length / n2_squared);
    let qa = qc * ((c - b).dot(edge) / length_squared) + qd * ((d - b).dot(edge) / length_squared);
    let qb = -qc * ((c - a).dot(edge) / length_squared) - qd * ((d - a).dot(edge) / length_squared);
    Some((angle, [qa, qb, qc, qd]))
}

fn solve_bend(
    positions: &mut [V3],
    inverse_mass: &[f64],
    constraint: &Bend,
    lambda: &mut f64,
    compliance: f64,
    h: f64,
) {
    let Some((angle, gradient)) =
        bending_geometry(constraint.vertices.map(|index| positions[index]))
    else {
        return;
    };
    let mut error = angle - constraint.rest_angle;
    if error > std::f64::consts::PI {
        error -= std::f64::consts::TAU;
    }
    if error < -std::f64::consts::PI {
        error += std::f64::consts::TAU;
    }
    let alpha = compliance / (h * h);
    let denominator = constraint
        .vertices
        .iter()
        .zip(gradient)
        .fold(alpha, |sum, (&index, g)| sum + inverse_mass[index] * g.length_squared());
    if denominator < 1e-12 {
        return;
    }
    let change = (-error - alpha * (*lambda)) / denominator;
    *lambda += change;
    for (&index, g) in constraint.vertices.iter().zip(gradient) {
        positions[index] += g * (inverse_mass[index] * change);
    }
}

fn measure(positions: &[V3], pattern: &Pattern, supports: [V3; 3], contacts: usize) -> FrameStats {
    let max_stretch = pattern
        .stretch
        .iter()
        .map(|edge| {
            (positions[edge.vertices[0]] - positions[edge.vertices[1]]).length() / edge.rest
        })
        .fold(0.0, f64::max);
    let pin_error = if pattern.attachment_mode == AttachmentMode::Rigid {
        pattern
            .pins
            .iter()
            .map(|pin| (positions[pin.vertex] - supports[pin.body] - pin.offset).length())
            .fold(0.0, f64::max)
    } else {
        supports
            .iter()
            .enumerate()
            .map(|(body, &support)| {
                let center = pattern
                    .pins
                    .iter()
                    .filter(|pin| pin.body == body)
                    .fold(V3::ZERO, |sum, pin| sum + positions[pin.vertex] * pin.weight);
                (center - support).length()
            })
            .fold(0.0, f64::max)
    };
    FrameStats { max_stretch, contacts, pin_error }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> OrbitData {
        OrbitData {
            seed: "test-orbit".to_string(),
            dt: 0.01,
            masses: [1.0; 3],
            samples: (0..30)
                .map(|i| {
                    let t = f64::from(i) / 29.0;
                    [
                        V3::new(0.0, 0.8, 0.1 * t),
                        V3::new(-0.7, -0.4, 0.03 * t),
                        V3::new(0.7, -0.4, -0.05 * t),
                    ]
                })
                .collect(),
            provenance: serde_json::json!({"fixture":true}),
        }
    }

    #[test]
    fn bake_is_repeatable_and_keeps_attachment_patches_exact() {
        let mut config = SimulationConfig {
            subdivisions: 8,
            frames: 3,
            fps: 24,
            substeps: 3,
            iterations: 8,
            settle_seconds: 0.3,
            preroll_seconds: 0.1,
            self_collision: false,
            ..SimulationConfig::default()
        };
        for mode in [AttachmentMode::Rigid, AttachmentMode::Centroid] {
            config.attachment_mode = mode;
            let first = bake(&fixture(), &config).unwrap();
            let second = bake(&fixture(), &config).unwrap();
            assert_eq!(first.frames, second.frames);
            assert!(first.frames.iter().flatten().all(|point| point.is_finite()));
            assert!(first.stats.iter().all(|stats| stats.pin_error < 1e-12));
            assert_eq!(first.mesh.positions, second.mesh.positions);
        }
    }

    #[test]
    fn dihedral_gradient_matches_finite_difference() {
        let points = [
            V3::new(0.0, 0.0, 0.0),
            V3::new(1.0, 0.0, 0.0),
            V3::new(0.2, 1.0, 0.3),
            V3::new(0.3, -1.0, 0.2),
        ];
        let (angle, gradients) = bending_geometry(points).unwrap();
        for vertex in 0..4 {
            for axis in 0..3 {
                let mut offset = points;
                let delta = match axis {
                    0 => V3::new(1e-6, 0.0, 0.0),
                    1 => V3::new(0.0, 1e-6, 0.0),
                    _ => V3::new(0.0, 0.0, 1e-6),
                };
                offset[vertex] += delta;
                let numerical = (bending_geometry(offset).unwrap().0 - angle) / 1e-6;
                assert!(
                    (numerical - gradients[vertex].axis(axis)).abs() < 2e-5,
                    "vertex {vertex}, axis {axis}: {numerical}"
                );
            }
        }
    }

    #[test]
    fn material_metric_is_not_changed_by_support_motion() {
        let config = SimulationConfig { subdivisions: 12, ..SimulationConfig::default() };
        let pattern = make_pattern(fixture().samples[0], &config).unwrap();
        let lengths: Vec<_> = pattern.stretch.iter().map(|edge| edge.rest).collect();
        let mut moved = pattern.mesh.positions.clone();
        set_pins(&mut moved, &pattern.pins, fixture().samples[29], config.attachment_mode);
        assert_eq!(lengths, pattern.stretch.iter().map(|edge| edge.rest).collect::<Vec<_>>());
        assert!(pattern.pins.len() > 3);
    }

    #[test]
    fn circular_pattern_has_consistent_faces_and_a_smooth_connected_boundary() {
        let config = SimulationConfig { subdivisions: 16, ..SimulationConfig::default() };
        let supports = fixture().samples[0];
        let pattern = make_pattern(supports, &config).unwrap();
        let mut counts = BTreeMap::new();
        let mut used = vec![false; pattern.mesh.positions.len()];
        let reference = (pattern.rest_supports[1] - pattern.rest_supports[0])
            .cross(pattern.rest_supports[2] - pattern.rest_supports[0]);
        for triangle in &pattern.mesh.triangles {
            let [a, b, c] = triangle.map(|index| index as usize);
            assert!(
                (pattern.mesh.positions[b] - pattern.mesh.positions[a])
                    .cross(pattern.mesh.positions[c] - pattern.mesh.positions[a])
                    .dot(reference)
                    > 0.0
            );
            for vertex in [a, b, c] {
                used[vertex] = true;
            }
            for [first, second] in [[a, b], [b, c], [c, a]] {
                *counts.entry([first.min(second), first.max(second)]).or_insert(0) += 1;
            }
        }
        assert!(used.into_iter().all(|connected| connected));
        let boundary: Vec<_> = counts.iter().filter(|(_, count)| **count == 1).collect();
        assert_eq!(boundary.len(), 6 * (config.subdivisions / 2));
        for (edge, _) in boundary {
            for &index in edge {
                let [u, v] = pattern.mesh.uv[index];
                assert!(
                    ((2.0 * u - 1.0) * (2.0 * u - 1.0) + (2.0 * v - 1.0) * (2.0 * v - 1.0) - 1.0)
                        .abs()
                        < 1e-12
                );
            }
        }
    }

    #[test]
    fn centroid_attachment_preserves_rotation_and_reaches_its_target() {
        let config = SimulationConfig {
            subdivisions: 16,
            attachment_mode: AttachmentMode::Centroid,
            ..SimulationConfig::default()
        };
        let pattern = make_pattern(fixture().samples[0], &config).unwrap();
        let mut positions = pattern.mesh.positions.clone();
        let center = pattern.rest_supports[0];
        for pin in pattern.pins.iter().filter(|pin| pin.body == 0) {
            let local = positions[pin.vertex] - center;
            positions[pin.vertex] = center + V3::new(-local.y, local.x, local.z);
        }
        let rotated = positions.clone();
        set_pins(&mut positions, &pattern.pins, pattern.rest_supports, AttachmentMode::Centroid);
        assert!(positions.iter().zip(&rotated).all(|(&a, &b)| (a - b).length() < 1e-12));
        let mut targets = pattern.rest_supports;
        targets[0] += V3::new(0.5, -0.2, 0.7);
        set_pins(&mut positions, &pattern.pins, targets, AttachmentMode::Centroid);
        let stats = measure(&positions, &pattern, targets, 0);
        assert!(stats.pin_error < 1e-12);
        for pin in pattern.pins.iter().filter(|pin| pin.body == 0) {
            assert!(
                (positions[pin.vertex] - targets[0] - (rotated[pin.vertex] - center)).length()
                    < 1e-12
            );
            assert!(pattern.inverse_mass[pin.vertex] > 0.0);
        }
    }

    #[test]
    fn hold_first_covers_visible_endpoints_at_multiple_frame_rates() {
        let orbit = fixture();
        for fps in [24, 30, 60] {
            for frames in [2, fps * 30] {
                let config = SimulationConfig {
                    frames,
                    fps,
                    preroll_seconds: 1.25,
                    preroll_mode: PrerollMode::HoldFirst,
                    orbit_start: 0.2,
                    orbit_end: 0.8,
                    ..SimulationConfig::default()
                };
                let duration = f64::from(frames - 1) / f64::from(fps);
                let driver = Driver {
                    orbit: &orbit,
                    config: &config,
                    center: V3::ZERO,
                    scale: 1.0,
                    duration,
                    rest_supports: sample_orbit(&orbit, config.orbit_start),
                };
                for time in [0.0, config.preroll_seconds * 0.5, config.preroll_seconds] {
                    assert_eq!(driver.source_fraction(time), config.orbit_start);
                    assert_eq!(driver.supports(time), sample_orbit(&orbit, config.orbit_start));
                }
                let middle = config.preroll_seconds + duration * 0.5;
                assert!((driver.source_fraction(middle) - 0.5).abs() < 1e-12);
                let end = config.preroll_seconds + duration;
                assert_eq!(driver.source_fraction(end), config.orbit_end);
                assert_eq!(driver.supports(end), sample_orbit(&orbit, config.orbit_end));
                assert_eq!(driver.source_fraction(end + 1.0), config.orbit_end);
            }
        }
    }

    #[test]
    fn omitted_preroll_mode_keeps_legacy_source_timing() {
        let config: SimulationConfig =
            serde_json::from_str(r#"{"frames":900,"fps":30,"preroll_seconds":1.0}"#).unwrap();
        assert_eq!(config.preroll_mode, PrerollMode::AdvanceSource);
        let orbit = fixture();
        let duration =
            config.preroll_seconds + f64::from(config.frames - 1) / f64::from(config.fps);
        let driver = Driver {
            orbit: &orbit,
            config: &config,
            center: V3::ZERO,
            scale: 1.0,
            duration,
            rest_supports: orbit.samples[0],
        };
        assert_eq!(driver.source_fraction(config.preroll_seconds), 1.0 / duration);
        assert!(driver.source_fraction(config.preroll_seconds) > 0.0);
        assert_eq!(driver.source_fraction(duration), 1.0);
    }

    #[test]
    fn full_visible_bake_reaches_actual_first_and_last_source_positions() {
        let orbit = fixture();
        let config = SimulationConfig {
            subdivisions: 8,
            frames: 3,
            fps: 24,
            substeps: 3,
            iterations: 8,
            settle_seconds: 0.3,
            preroll_seconds: 0.2,
            preroll_mode: PrerollMode::HoldFirst,
            self_collision: false,
            ..SimulationConfig::default()
        };
        let result = bake(&orbit, &config).unwrap();
        let center: V3 = serde_json::from_value(result.recipe["world_center"].clone()).unwrap();
        let scale = result.recipe["world_scale"].as_f64().unwrap();
        let first = orbit.samples[0].map(|point| (point - center) * scale);
        let last = orbit.samples.last().unwrap().map(|point| (point - center) * scale);
        let pattern = make_pattern(first, &config).unwrap();
        assert!(measure(&result.frames[0], &pattern, first, 0).pin_error < 1e-12);
        assert!(measure(result.frames.last().unwrap(), &pattern, last, 0).pin_error < 1e-12);
        assert_eq!(result.recipe["first_visible_source_fraction"], 0.0);
        assert_eq!(result.recipe["last_visible_source_fraction"], 1.0);
        assert_eq!(result.recipe["preroll_mode"], "hold_first");
    }
}
