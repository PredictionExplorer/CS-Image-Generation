//! Aurora Veils: broad luminous graphs shaped by the three recorded histories.
//!
//! Fixed source axes drive large lateral and vertical sweeps. Each sheet has a
//! strictly increasing unsheared chronology coordinate and a positive vertical
//! span. An invertible world-height shear curves its rays without folding the
//! surface onto itself. Broad gatherings follow actual source arc length; their finest
//! radius and length variation belongs to fixed ray identities, avoiding a
//! fast source phase that could flicker between film frames.

use super::{
    EmissionProfile, EmissionStop, Material, OrbitSeries, Scene, SilkResult, Strand, Triangle,
    UvFeather, V3, Vertex,
};
use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, TAU};

/// Source, geometry and light controls for three open atmospheric curtains.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AuroraConfig {
    /// Actual recent history; a formed opening requires verified prehistory.
    pub history_fraction: f64,
    /// Fixed orthonormal source basis; never fitted again during a film.
    pub source_axes: [V3; 3],
    /// Shared per-axis world scales inside the bounded tanh source mapping.
    pub source_scales: [f64; 3],
    /// Maximum lateral translation and vertical lower-edge displacement.
    pub source_motion: [f64; 2],
    /// Maximum extra lateral bend from a source-driven, invertible height shear.
    pub lean_amplitude: f64,
    /// Fixed world height at which the lateral height shear changes sign.
    pub lean_height_center: f64,
    /// Positive world-height scale of the broad, smooth lateral curve.
    pub lean_height_scale: f64,
    /// Permanent chronological widths in original body order.
    pub widths: [f64; 3],
    /// Enable original bodies A, B and C independently for composition proofs.
    pub curtains: [bool; 3],
    /// Fixed depth centers; all enabled layers must remain separated.
    pub depth_lanes: [f64; 3],
    /// Maximum depth displacement from the third fixed source component.
    pub source_depth: f64,
    /// Maximum broad fold depth, with its phase fixed in source arc length.
    pub fold_amplitude: f64,
    /// World source arc length for one broad depth-fold cycle.
    pub fold_pitch: f64,
    /// Positive minimum sheet height, including at chronological ends.
    pub height_floor: f64,
    /// Broad positive height swell in the interior of the history window.
    pub height_bulge: f64,
    /// Additional height in close source passages.
    pub height_proximity: f64,
    /// Additional height in fast source passages.
    pub height_speed: f64,
    /// Intervals along each broad history sheet.
    pub sheet_segments: usize,
    /// Intervals from the lower edge to the upper haze.
    pub height_segments: usize,
    /// Fine curved rays per body, following the sheared sheet.
    pub ray_count: usize,
    /// Intervals along each fine ray.
    pub ray_segments: usize,
    /// Nominal fine-ray radius in world units before continuous tapering.
    pub ray_radius: f64,
    /// Stable fractional fine-ray radius variation, bounded below one.
    pub ray_radius_variation: f64,
    /// Maximum removal of the upper ray length for irregular dissolving ends.
    pub ray_length_variation: f64,
    /// World source arc length for one broad luminous gathering.
    pub ray_gather_pitch: f64,
    /// Characteristic ray-index scale of smooth aperiodic radius/length detail.
    /// The field name is retained for recipe compatibility; the texture has no period.
    pub ray_detail_period: f64,
    /// Minimum ray radiance relative to its bright gatherings.
    pub ray_dark_floor: f64,
    /// Chronological fade widths at the old and current ends.
    pub side_feather: [f64; 2],
    /// Relative-height fade widths at the lower edge and upper haze.
    pub height_feather: [f64; 2],
    /// Mean material optical depth of the broad sheets.
    pub sheet_optical_depth: f64,
    /// Mean material optical depth of the fine rays.
    pub ray_optical_depth: f64,
    /// Relative emission of the quiet web between bright rays.
    pub sheet_emission: f64,
    /// Relative emission of individual fine rays before source modulation.
    pub ray_emission: f64,
    /// Overall emitted-light hierarchy, preserving original body identities.
    pub body_emission: [f64; 3],
    /// Shared world-height light profile for both sheets and rays.
    ///
    /// Geometry supplies its own smooth sheet UV feather. Rays share the world
    /// profile but use geometric radii to dissolve at their irregular ends.
    pub emission_profile: EmissionProfile,
    /// Stable detail seed; never changes source paths or their clock.
    pub detail_seed: u64,
    /// Base dielectric materials in original body order.
    pub materials: [Material; 3],
}

impl Default for AuroraConfig {
    fn default() -> Self {
        Self {
            history_fraction: 0.26,
            source_axes: [V3::new(1.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0), V3::new(0.0, 0.0, 1.0)],
            source_scales: [1.5; 3],
            source_motion: [1.8, 1.8],
            lean_amplitude: 0.7,
            lean_height_center: 0.8,
            lean_height_scale: 1.3,
            widths: [5.2, 4.8, 5.6],
            curtains: [true; 3],
            depth_lanes: [-1.2, 0.0, 1.2],
            source_depth: 0.18,
            fold_amplitude: 0.28,
            fold_pitch: 2.8,
            height_floor: 0.25,
            height_bulge: 1.7,
            height_proximity: 0.6,
            height_speed: 0.2,
            sheet_segments: 1024,
            height_segments: 192,
            ray_count: 3072,
            ray_segments: 160,
            ray_radius: 0.00046,
            ray_radius_variation: 0.42,
            ray_length_variation: 0.24,
            ray_gather_pitch: 3.8,
            ray_detail_period: 17.0,
            ray_dark_floor: 0.045,
            side_feather: [0.18, 0.14],
            height_feather: [0.025, 0.55],
            sheet_optical_depth: 0.055,
            ray_optical_depth: 0.22,
            sheet_emission: 0.12,
            ray_emission: 1.05,
            body_emission: [1.0, 0.72, 0.88],
            emission_profile: default_profile(),
            detail_seed: 4_731_029,
            materials: std::array::from_fn(|_| Material {
                front_color: V3::new(0.045, 0.13, 0.12),
                back_color: V3::new(0.025, 0.08, 0.11),
                optical_depth: V3::new(0.055, 0.055, 0.055),
                roughness: 0.55,
                anisotropy: 0.55,
                sheen: 0.04,
                emission: V3::ZERO,
                fiber_strength: 0.0,
                metallic: 0.0,
                ..Material::default()
            }),
        }
    }
}

fn default_profile() -> EmissionProfile {
    EmissionProfile {
        origin: V3::new(0.0, -2.0, 0.0),
        axis: V3::new(0.0, 1.0, 0.0),
        extent: 6.6,
        stops: vec![
            EmissionStop { position: 0.0, emission: V3::new(0.12, 0.48, 0.26), density: 0.0 },
            EmissionStop { position: 0.15, emission: V3::new(0.25, 1.55, 0.58), density: 0.78 },
            EmissionStop { position: 0.38, emission: V3::new(0.12, 1.28, 0.67), density: 0.95 },
            EmissionStop { position: 0.60, emission: V3::new(0.13, 0.62, 0.92), density: 0.68 },
            EmissionStop { position: 0.82, emission: V3::new(0.25, 0.10, 0.61), density: 0.35 },
            EmissionStop { position: 1.0, emission: V3::new(0.17, 0.045, 0.28), density: 0.0 },
        ],
        uv_feather: None,
    }
}

#[derive(Clone, Copy, Debug)]
struct Row {
    x: f64,
    base: f64,
    height: f64,
    depth: f64,
    fold: f64,
    arc: f64,
    proximity: f64,
    lean: f64,
    lean_center: f64,
    lean_scale: f64,
}

impl Row {
    fn position(self, v: f64) -> V3 {
        let y = self.base + self.height * v;
        V3::new(self.x + self.shear(y).0, y, self.depth + self.fold * (PI * v).sin())
    }

    fn vertical(self, v: f64) -> V3 {
        let y = self.base + self.height * v;
        V3::new(self.shear(y).1 * self.height, self.height, self.fold * PI * (PI * v).cos())
    }

    fn longitudinal(self, derivative: RowDerivative, v: f64) -> V3 {
        let dy = derivative.base + derivative.height * v;
        let y = self.base + self.height * v;
        V3::new(
            derivative.x + self.shear(y).1 * dy,
            dy,
            derivative.depth + derivative.fold * (PI * v).sin(),
        )
    }

    fn shear(self, y: f64) -> (f64, f64) {
        let curve = ((y - self.lean_center) / self.lean_scale).tanh();
        (self.lean * curve, self.lean / self.lean_scale * (1.0 - curve * curve))
    }
}

#[derive(Clone, Copy, Debug)]
struct RowDerivative {
    x: f64,
    base: f64,
    height: f64,
    depth: f64,
    fold: f64,
}

struct Field<'a> {
    source: &'a OrbitSeries,
    config: &'a AuroraConfig,
    time: f64,
    head_x: [f64; 3],
    lean: [f64; 3],
}

impl<'a> Field<'a> {
    fn new(source: &'a OrbitSeries, time: f64, config: &'a AuroraConfig) -> SilkResult<Self> {
        let head = source.sample(time).ok_or("aurora head is outside the recorded source")?;
        let head_x = head
            .bodies
            .map(|body| bounded_component(body.position, 0, config) * config.source_motion[0]);
        let lean = head.bodies.map(|body| {
            config.lean_amplitude
                * (0.65 * bounded_component(body.position, 0, config)
                    + 0.35 * bounded_component(body.position, 1, config))
        });
        Ok(Self { source, config, time, head_x, lean })
    }

    fn row(&self, body: usize, u: f64) -> SilkResult<Row> {
        // Preserve the exact cached visible head when u reaches one. No wrapped
        // histories, clamped source requests or future samples are introduced.
        let fraction =
            if u == 1.0 { self.time } else { self.time - self.config.history_fraction * (1.0 - u) };
        let sample =
            self.source.sample_body(body, fraction).ok_or("aurora history is unavailable")?;
        let config = self.config;
        let phase = TAU * sample.arc_length / config.fold_pitch + body as f64 * 2.1;
        let wave = 0.72 * phase.sin() + 0.28 * (phase * 0.47 + 0.7).sin();
        Ok(Row {
            x: self.head_x[body] + config.widths[body] * (u - 0.5),
            base: bounded_component(sample.position, 1, config) * config.source_motion[1],
            height: config.height_floor
                + (config.height_bulge
                    + config.height_proximity * sample.proximity
                    + config.height_speed * sample.speed)
                    * (PI * u).sin().powi(2),
            depth: config.depth_lanes[body]
                + config.source_depth * bounded_component(sample.position, 2, config),
            fold: config.fold_amplitude * (0.65 + 0.35 * sample.proximity) * wave,
            arc: sample.arc_length,
            proximity: sample.proximity,
            lean: self.lean[body],
            lean_center: config.lean_height_center,
            lean_scale: config.lean_height_scale,
        })
    }

    fn vertex_row(&self, body: usize, u: f64) -> SilkResult<(Row, RowDerivative)> {
        // The original unsheared x derivative is exact. The invertible shear
        // carries this derivative and the positive vertical span into world
        // coordinates without changing their XY area. A small source interval
        // supplies the derivative of the recorded, Hermite-interpolated history.
        let du = 1e-5;
        let low = (u - du).max(0.0);
        let high = (u + du).min(1.0);
        let left = self.row(body, low)?;
        let right = self.row(body, high)?;
        let inverse = 1.0 / (high - low);
        let derivative = RowDerivative {
            x: self.config.widths[body],
            base: (right.base - left.base) * inverse,
            height: (right.height - left.height) * inverse,
            depth: (right.depth - left.depth) * inverse,
            fold: (right.fold - left.fold) * inverse,
        };
        Ok((self.row(body, u)?, derivative))
    }
}

fn bounded_component(position: V3, axis: usize, config: &AuroraConfig) -> f64 {
    (position.dot(config.source_axes[axis]) / config.source_scales[axis]).tanh()
}

/// Build one full-detail frame of source-driven luminous veils.
///
/// Real history must cover the complete requested window. Source time remains
/// in `[0,1]`; negative history is accepted only through verified prehistory.
pub fn scene(source: &OrbitSeries, time: f64, config: &AuroraConfig) -> SilkResult<Scene> {
    validate(source, time, config)?;
    let field = Field::new(source, time, config)?;
    let enabled = config.curtains.iter().filter(|enabled| **enabled).count();
    let mut scene = Scene {
        vertices: Vec::with_capacity(
            enabled * (config.sheet_segments + 1) * (config.height_segments + 1),
        ),
        triangles: Vec::with_capacity(2 * enabled * config.sheet_segments * config.height_segments),
        strands: Vec::with_capacity(enabled * config.ray_count),
        materials: Vec::with_capacity(enabled * (config.ray_count + 1)),
    };
    for body in 0..3 {
        if !config.curtains[body] {
            continue;
        }
        add_sheet(&mut scene, &field, body)?;
        add_rays(&mut scene, &field, body)?;
    }
    if scene.vertices.iter().any(|vertex| {
        !vertex.position.is_finite() || !vertex.normal.is_finite() || !vertex.tangent.is_finite()
    }) || scene.strands.iter().any(|strand| {
        !strand.points.iter().all(|point| point.is_finite())
            || !strand.radii.iter().all(|radius| radius.is_finite() && *radius >= 0.0)
    }) {
        return Err("aurora geometry exceeds finite coordinates".into());
    }
    Ok(scene)
}

fn add_sheet(scene: &mut Scene, field: &Field<'_>, body: usize) -> SilkResult<()> {
    let config = field.config;
    let material = scene.materials.len();
    scene.materials.push(make_material(config, body, config.sheet_emission, true));
    let first = u32::try_from(scene.vertices.len())?;
    let across = config.height_segments + 1;
    for index in 0..=config.sheet_segments {
        let u = index as f64 / config.sheet_segments as f64;
        let (row, derivative) = field.vertex_row(body, u)?;
        for height in 0..=config.height_segments {
            let v = height as f64 / config.height_segments as f64;
            let du = row.longitudinal(derivative, v);
            let dv = row.vertical(v);
            scene.vertices.push(Vertex {
                position: row.position(v),
                normal: du.cross(dv).normalized(),
                tangent: dv.normalized(),
                uv: [u, v],
            });
        }
    }
    for index in 0..config.sheet_segments {
        for height in 0..config.height_segments {
            let a = first + u32::try_from(index * across + height)?;
            let b = a + u32::try_from(across)?;
            scene.triangles.push(Triangle { indices: [a, b, a + 1], material });
            scene.triangles.push(Triangle { indices: [a + 1, b, b + 1], material });
        }
    }
    Ok(())
}

fn add_rays(scene: &mut Scene, field: &Field<'_>, body: usize) -> SilkResult<()> {
    let config = field.config;
    let seed_phase = TAU * hash_unit(config.detail_seed ^ body as u64);
    for index in 0..config.ray_count {
        let u = (index as f64 + 0.5) / config.ray_count as f64;
        let row = field.row(body, u)?;
        let gathering = TAU * row.arc / config.ray_gather_pitch + seed_phase;
        let broad = (0.5 + 0.5 * gathering.sin()).powi(2);
        let density = config.ray_dark_floor + (1.0 - config.ray_dark_floor) * broad;
        // The smallest detail belongs to material-ray identity, never to an
        // advancing source phase. Its several smooth scales remain stationary
        // on the curtain even during a very fast physical close encounter.
        let (fine, length_wave) = ray_variation(index, body, config);
        let radius = config.ray_radius * (1.0 + config.ray_radius_variation * fine);
        let length = 1.0 - config.ray_length_variation * (0.5 + 0.5 * length_wave);
        let end_feather = side_envelope(u, config);
        let brightness =
            config.ray_emission * density * (0.78 + 0.22 * row.proximity) * (0.86 + 0.14 * fine);
        let material = scene.materials.len();
        scene.materials.push(make_material(config, body, brightness, false));
        let mut strand = Strand {
            points: Vec::with_capacity(config.ray_segments + 1),
            radii: Vec::with_capacity(config.ray_segments + 1),
            material,
        };
        for height in 0..=config.ray_segments {
            let relative = height as f64 / config.ray_segments as f64;
            let v = relative * length;
            strand.points.push(row.position(v));
            strand.radii.push(radius * end_feather * height_envelope(relative, config));
        }
        scene.strands.push(strand);
    }
    Ok(())
}

fn make_material(config: &AuroraConfig, body: usize, brightness: f64, sheet: bool) -> Material {
    let mut material = config.materials[body].clone();
    let depth = if sheet { config.sheet_optical_depth } else { config.ray_optical_depth };
    let mean =
        (material.optical_depth.x + material.optical_depth.y + material.optical_depth.z) / 3.0;
    material.optical_depth = if mean > 1e-12 {
        material.optical_depth * (depth / mean)
    } else {
        V3::new(depth, depth, depth)
    };
    let multiplier = brightness * config.body_emission[body];
    material.emission *= multiplier;
    let mut profile = config.emission_profile.clone();
    for stop in &mut profile.stops {
        stop.emission *= multiplier;
    }
    profile.uv_feather =
        sheet.then_some(UvFeather { u: config.side_feather, v: config.height_feather });
    material.emission_profile = Some(profile);
    material
}

fn smoothstep(value: f64) -> f64 {
    let u = value.clamp(0.0, 1.0);
    u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
}

fn side_envelope(u: f64, config: &AuroraConfig) -> f64 {
    smoothstep(u / config.side_feather[0]) * smoothstep((1.0 - u) / config.side_feather[1])
}

fn height_envelope(v: f64, config: &AuroraConfig) -> f64 {
    smoothstep(v / config.height_feather[0]) * smoothstep((1.0 - v) / config.height_feather[1])
}

fn hash_unit(mut value: u64) -> f64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((value ^ (value >> 31)) >> 11) as f64 / 9_007_199_254_740_992.0
}

fn value_noise(coordinate: f64, seed: u64) -> f64 {
    let cell = coordinate.floor();
    let index = cell as i64 as u64;
    let value = |at: u64| 2.0 * hash_unit(seed ^ at.wrapping_mul(0x9e37_79b9_7f4a_7c15)) - 1.0;
    let left = value(index);
    let right = value(index.wrapping_add(1));
    // Quintic interpolation has zero first and second derivatives at every
    // lattice knot. Each knot is independently hashed, with no tiled pattern.
    let blend = smoothstep(coordinate - cell);
    (left * (1.0 - blend) + right * blend).clamp(-1.0, 1.0)
}

fn ray_variation(index: usize, body: usize, config: &AuroraConfig) -> (f64, f64) {
    let coordinate = (index as f64 + 0.5) / config.ray_detail_period;
    let seed = config.detail_seed ^ (body as u64).wrapping_mul(0xd1b5_4a32_d192_ed03);
    let fine = 0.25 * value_noise(coordinate * 2.0, seed ^ 0x68e3_1da4)
        + 0.45 * value_noise(coordinate, seed ^ 0xb529_7a4d)
        + 0.30 * value_noise(coordinate / 1.9, seed ^ 0x1b56_c4e9);
    let length = 0.40 * value_noise(coordinate * 2.0, seed ^ 0x7a14_3521)
        + 0.35 * value_noise(coordinate, seed ^ 0x93c4_6761)
        + 0.25 * value_noise(coordinate / 1.9, seed ^ 0x54a3_a821);
    (fine.clamp(-1.0, 1.0), length.clamp(-1.0, 1.0))
}

fn validate(source: &OrbitSeries, time: f64, config: &AuroraConfig) -> SilkResult<()> {
    if !time.is_finite() || !(0.0..=1.0).contains(&time) {
        return Err("aurora time must be a finite source fraction in [0,1]".into());
    }
    let positive = [
        config.history_fraction,
        config.height_floor,
        config.lean_height_scale,
        config.fold_pitch,
        config.ray_gather_pitch,
        config.ray_detail_period,
        config.ray_radius,
    ];
    if positive
        .iter()
        .chain(config.widths.iter())
        .chain(config.source_scales.iter())
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("aurora dimensions, scales and pitches must be positive and finite".into());
    }
    if time - config.history_fraction < source.history_start_fraction() {
        return Err("aurora needs genuine prehistory covering the requested history window".into());
    }
    let nonnegative = [
        config.source_depth,
        config.lean_amplitude,
        config.fold_amplitude,
        config.height_bulge,
        config.height_proximity,
        config.height_speed,
        config.sheet_optical_depth,
        config.ray_optical_depth,
        config.sheet_emission,
        config.ray_emission,
    ];
    if nonnegative
        .iter()
        .chain(config.source_motion.iter())
        .chain(config.body_emission.iter())
        .any(|value| !value.is_finite() || *value < 0.0)
        || !config.depth_lanes.iter().all(|value| value.is_finite())
        || !config.lean_height_center.is_finite()
    {
        return Err(
            "aurora motion, depth and light controls must be finite with valid signs".into()
        );
    }
    for axis in &config.source_axes {
        if !axis.is_finite() || (axis.length_squared() - 1.0).abs() > 1e-8 {
            return Err("aurora source axes must be a fixed orthonormal basis".into());
        }
    }
    for first in 0..3 {
        for second in first + 1..3 {
            if config.source_axes[first].dot(config.source_axes[second]).abs() > 1e-8 {
                return Err("aurora source axes must be mutually perpendicular".into());
            }
            if config.curtains[first]
                && config.curtains[second]
                && (config.depth_lanes[first] - config.depth_lanes[second]).abs()
                    <= 2.0 * (config.source_depth + config.fold_amplitude)
            {
                return Err(
                    "aurora depth lanes must remain separated over the entire source".into()
                );
            }
        }
    }
    if [config.ray_radius_variation, config.ray_length_variation, config.ray_dark_floor]
        .iter()
        .any(|value| !value.is_finite() || !(0.0..1.0).contains(value))
        || config
            .side_feather
            .iter()
            .chain(config.height_feather.iter())
            .any(|value| !value.is_finite() || *value <= 0.0 || *value > 1.0)
    {
        return Err("aurora detail variation and smooth feather widths must be bounded".into());
    }
    if config.sheet_segments < 2
        || config.height_segments < 2
        || config.ray_segments < 2
        || config.ray_count == 0
    {
        return Err("aurora sheets and rays require at least two intervals and one ray".into());
    }
    let vertices = config
        .sheet_segments
        .checked_add(1)
        .and_then(|n| config.height_segments.checked_add(1).and_then(|m| n.checked_mul(m)))
        .and_then(|n| n.checked_mul(3))
        .ok_or("aurora sheet size overflows")?;
    let ray_points = config
        .ray_segments
        .checked_add(1)
        .and_then(|n| n.checked_mul(config.ray_count))
        .and_then(|n| n.checked_mul(3))
        .ok_or("aurora ray size overflows")?;
    if vertices > 30_000_000 || ray_points > 60_000_000 {
        return Err("aurora detail exceeds supported per-frame geometry limits".into());
    }
    if config.materials.iter().any(|material| material.metallic != 0.0) {
        return Err("aurora emissive curtains require dielectric base materials".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silk::OrbitData;

    fn source() -> OrbitSeries {
        let samples = (0..257)
            .map(|index| {
                let t = f64::from(index) / 256.0;
                std::array::from_fn(|body| {
                    let angle = TAU * (t + body as f64 / 3.0);
                    V3::new(angle.cos(), angle.sin(), 0.27 * (2.0 * angle + t).sin())
                })
            })
            .collect();
        OrbitSeries::new(&OrbitData {
            seed: "aurora-test".to_owned(),
            dt: 0.01,
            masses: [1.0; 3],
            samples,
            provenance: serde_json::json!({"synthetic_unit_fixture":true}),
        })
        .unwrap()
    }

    fn small_config() -> AuroraConfig {
        AuroraConfig {
            sheet_segments: 16,
            height_segments: 12,
            ray_count: 24,
            ray_segments: 16,
            ..AuroraConfig::default()
        }
    }

    #[test]
    fn source_history_endpoints_and_large_motion_are_preserved() {
        let source = source();
        let config = small_config();
        for time in [0.26, 0.50, 0.75, 1.0] {
            let field = Field::new(&source, time, &config).unwrap();
            for body in 0..3 {
                for u in [0.0, 1.0] {
                    let row = field.row(body, u).unwrap();
                    let source_time = time - config.history_fraction * (1.0 - u);
                    let actual = source.sample_body(body, source_time).unwrap();
                    assert_eq!(row.arc, actual.arc_length);
                    assert_eq!(
                        row.base,
                        config.source_motion[1] * bounded_component(actual.position, 1, &config)
                    );
                }
            }
        }
        let early = Field::new(&source, 0.30, &config).unwrap();
        let late = Field::new(&source, 0.80, &config).unwrap();
        let move_a =
            early.row(0, 0.5).unwrap().position(0.0) - late.row(0, 0.5).unwrap().position(0.0);
        assert!(move_a.x.abs() > 0.5);
        assert!(move_a.y.abs() > 1.0);
    }

    #[test]
    fn sheets_are_regular_bounded_and_depth_separated() {
        let source = source();
        let config = small_config();
        let field = Field::new(&source, 0.65, &config).unwrap();
        let height_bound = config.height_floor
            + config.height_bulge
            + config.height_proximity
            + config.height_speed;
        for body in 0..3 {
            for index in 0..=64 {
                let u = f64::from(index) / 64.0;
                let (row, derivative) = field.vertex_row(body, u).unwrap();
                assert!(row.height >= config.height_floor);
                for v in [0.0, 0.2, 0.5, 0.8, 1.0] {
                    let point = row.position(v);
                    let du = row.longitudinal(derivative, v);
                    let dv = row.vertical(v);
                    let shear_slope = row.shear(point.y).1;
                    assert!((du.x - shear_slope * du.y - config.widths[body]).abs() < 1e-12);
                    assert!((dv.x - shear_slope * dv.y).abs() < 1e-12);
                    assert_eq!(dv.y, row.height);
                    assert!(du.cross(dv).length() >= config.widths[body] * config.height_floor);
                    assert!((du.cross(dv).z - config.widths[body] * row.height).abs() < 1e-10);
                    assert!(
                        point.x.abs()
                            <= config.source_motion[0]
                                + config.widths[body] * 0.5
                                + config.lean_amplitude
                    );
                    assert!(point.y >= -config.source_motion[1]);
                    assert!(point.y <= config.source_motion[1] + height_bound);
                    assert!(
                        (point.z - config.depth_lanes[body]).abs()
                            <= config.source_depth + config.fold_amplitude
                    );
                }
            }
        }
    }

    #[test]
    fn rays_follow_the_same_surface_and_dissolve_at_both_ends() {
        let source = source();
        let config = small_config();
        let field = Field::new(&source, 0.6, &config).unwrap();
        let scene = scene(&source, 0.6, &config).unwrap();
        assert_eq!(scene.strands.len(), 3 * config.ray_count);
        assert_eq!(scene.triangles.len(), 6 * config.sheet_segments * config.height_segments);
        for (index, ray) in scene.strands.iter().enumerate() {
            let body = index / config.ray_count;
            let u = ((index % config.ray_count) as f64 + 0.5) / config.ray_count as f64;
            let row = field.row(body, u).unwrap();
            assert_eq!(ray.radii[0], 0.0);
            assert_eq!(*ray.radii.last().unwrap(), 0.0);
            assert!(ray.radii[ray.radii.len() / 2] > 0.0);
            for point in &ray.points {
                let v = (point.y - row.base) / row.height;
                assert!((*point - row.position(v)).length() < 1e-12);
            }
            assert!(
                scene.materials[ray.material]
                    .emission_profile
                    .as_ref()
                    .unwrap()
                    .uv_feather
                    .is_none()
            );
        }
    }

    #[test]
    fn source_driven_height_shear_curves_rays_and_has_an_exact_inverse() {
        let source = source();
        let config = small_config();
        let field = Field::new(&source, 0.55, &config).unwrap();
        let later = Field::new(&source, 0.80, &config).unwrap();
        assert_ne!(field.lean, later.lean);
        for body in 0..3 {
            let mut previous_unsheared = f64::NEG_INFINITY;
            for index in 0..=64 {
                let u = f64::from(index) / 64.0;
                let (row, derivative) = field.vertex_row(body, u).unwrap();
                assert!(row.lean.abs() <= config.lean_amplitude);
                for v in [0.0, 0.25, 0.5, 0.75, 1.0] {
                    let point = row.position(v);
                    let inverse_x = point.x - row.shear(point.y).0;
                    assert!((inverse_x - row.x).abs() < 1e-12);
                    let h = 1e-6;
                    let numerical = (row.position(v + h) - row.position(v - h)) / (2.0 * h);
                    assert!((numerical - row.vertical(v)).length() < 1e-8);
                    let cross = row.longitudinal(derivative, v).cross(row.vertical(v));
                    assert!(cross.z > 0.0);
                }
                let inverse_x = row.position(0.6).x - row.shear(row.position(0.6).y).0;
                assert!(inverse_x > previous_unsheared);
                previous_unsheared = inverse_x;
            }
        }
        let row = field.row(0, 0.5).unwrap();
        assert!((row.position(0.85).x - row.position(0.15).x).abs() > 0.05);
        assert!((row.vertical(0.2).x - row.vertical(0.8).x).abs() > 0.02);
    }

    #[test]
    fn fine_ray_radii_and_lengths_belong_to_fixed_material_identity() {
        let source = source();
        let config = small_config();
        let first = scene(&source, 0.40, &config).unwrap();
        let later = scene(&source, 0.85, &config).unwrap();
        let first_field = Field::new(&source, 0.40, &config).unwrap();
        let later_field = Field::new(&source, 0.85, &config).unwrap();
        for (index, (a, b)) in first.strands.iter().zip(&later.strands).enumerate() {
            assert_eq!(a.radii, b.radii);
            let body = index / config.ray_count;
            let u = ((index % config.ray_count) as f64 + 0.5) / config.ray_count as f64;
            let row_a = first_field.row(body, u).unwrap();
            let row_b = later_field.row(body, u).unwrap();
            let length_a = (a.points.last().unwrap().y - row_a.base) / row_a.height;
            let length_b = (b.points.last().unwrap().y - row_b.base) / row_b.height;
            assert!((length_a - length_b).abs() < 1e-12);
        }
        assert_ne!(first.strands[0].points, later.strands[0].points);
    }

    #[test]
    fn value_noise_is_smooth_across_lattice_boundaries() {
        for seed in [0, 42, u64::MAX] {
            for cell in -8..=8 {
                let x = f64::from(cell);
                let center = value_noise(x, seed);
                assert!((-1.0..=1.0).contains(&center));
                for h in [0.001_f64, 0.0001] {
                    let left = value_noise(x - h, seed);
                    let right = value_noise(x + h, seed);
                    // At a C2 lattice join the endpoint departure is cubic,
                    // rather than a jump or a sudden change in first slope.
                    let bound = 21.0 * h.powi(3) + 1e-13;
                    assert!((left - center).abs() <= bound);
                    assert!((right - center).abs() <= bound);
                }
            }
        }
    }

    #[test]
    fn aperiodic_ray_detail_is_bounded_seeded_and_gently_varying() {
        let config = AuroraConfig { ray_detail_period: 13.0, ..small_config() };
        let alternate = AuroraConfig { detail_seed: config.detail_seed + 1, ..config.clone() };
        let mut minimum = 1.0_f64;
        let mut maximum = -1.0_f64;
        for body in 0..3 {
            for index in 0..2048 {
                let a = ray_variation(index, body, &config);
                let next = ray_variation(index + 1, body, &config);
                assert!((-1.0..=1.0).contains(&a.0));
                assert!((-1.0..=1.0).contains(&a.1));
                assert!((a.0 - next.0).abs() < 0.36);
                assert!((a.1 - next.1).abs() < 0.40);
                assert_eq!(a, ray_variation(index, body, &config));
                assert_ne!(a, ray_variation(index, body, &alternate));
                assert_ne!(a, ray_variation(index + 13, body, &config));
                minimum = minimum.min(a.0);
                maximum = maximum.max(a.0);
            }
        }
        assert!(maximum - minimum > 0.75);
    }

    #[test]
    fn missing_history_invalid_axes_and_unknown_controls_are_rejected() {
        let source = source();
        let mut config = small_config();
        assert!(scene(&source, 0.0, &config).is_err());
        assert!(scene(&source, f64::NAN, &config).is_err());
        config.source_axes[1] = config.source_axes[0];
        assert!(scene(&source, 0.6, &config).is_err());
        assert!(serde_json::from_str::<AuroraConfig>(r#"{"height_bluge":1.0}"#).is_err());
    }

    #[test]
    fn geometry_and_emission_are_exactly_repeatable() {
        let source = source();
        let config = small_config();
        let first = scene(&source, 0.73, &config).unwrap();
        let again = scene(&source, 0.73, &config).unwrap();
        assert_eq!(first.vertices.len(), again.vertices.len());
        for (a, b) in first.vertices.iter().zip(&again.vertices) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.normal, b.normal);
            assert_eq!(a.tangent, b.tangent);
        }
        for (a, b) in first.strands.iter().zip(&again.strands) {
            assert_eq!(a.points, b.points);
            assert_eq!(a.radii, b.radii);
        }
        assert_eq!(
            serde_json::to_value(&first.materials).unwrap(),
            serde_json::to_value(&again.materials).unwrap()
        );
    }
}
