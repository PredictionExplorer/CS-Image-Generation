//! Gravity Loom: an open woven conch made from three pairwise motion histories.
//!
//! Chronology has a fixed X coordinate. Recorded positions modulate separated
//! rail neighborhoods, and actual three-dimensional pair distances shape the
//! arches between them. Permanent material-space windows cut real alternating
//! warp/weft fibers. This is a designed mapping, not a free cloth simulation.

use super::{Material, OrbitSeries, Scene, SilkResult, Strand, V3};
use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, TAU};

const TINTS: usize = 16;
const TIME_AXIS: V3 = V3::new(1.0, 0.0, 0.0);
const REFERENCE_RADIUS: f64 = 1.1;

/// Permanent outer silhouette, independent of animation time.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoomProfile {
    /// Slender old tail, broad belly, and a larger open mouth.
    #[default]
    Conch,
    /// Symmetric belly with two narrower open ends.
    Spindle,
}

/// Construction of each arch between two source-modulated rails.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoomCrossSection {
    /// Original straight chord with an outward bow.
    Bowed,
    /// Rounded polar shell with smooth radial joins at the shared rails.
    #[default]
    Polar,
}

/// A permanent rotated ellipse removed from one panel's material coordinates.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoomAperture {
    /// Whether this panel has an open window.
    pub enabled: bool,
    /// Center in chronological/across-panel coordinates.
    pub center: [f64; 2],
    /// Positive ellipse half-axes in material coordinates.
    pub half_axes: [f64; 2],
    /// Fixed rotation in the material plane, in degrees.
    pub tilt_degrees: f64,
}

impl Default for LoomAperture {
    fn default() -> Self {
        Self { enabled: true, center: [0.43, 0.48], half_axes: [0.39, 0.37], tilt_degrees: -6.0 }
    }
}

/// Geometry and dye controls for the three relationship panels AB, BC, and CA.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoomConfig {
    /// Fixed outer silhouette.
    pub profile: LoomProfile,
    /// Rounded shell or original bowed-panel construction.
    pub cross_section: LoomCrossSection,
    /// Genuine recent history represented from left to right.
    pub history_fraction: f64,
    /// World-space length of the strictly monotone time axis.
    pub axis_length: f64,
    /// Permanent full-wave centerline displacement along Y.
    pub centerline_y: f64,
    /// Permanent half-wave centerline displacement along Z.
    pub centerline_z: f64,
    /// Conch radius at the oldest end.
    pub radius_tail: f64,
    /// Conch radius at the newest end.
    pub radius_mouth: f64,
    /// Extra asymmetric belly radius of the conch.
    pub radius_bulge: f64,
    /// Minimum radius of the spindle alternative.
    pub radius_min: f64,
    /// Maximum radius of the spindle alternative.
    pub radius_max: f64,
    /// Permanent rotation of all rail neighborhoods around the time axis.
    pub rail_rotation_degrees: f64,
    /// Fixed total twist from oldest to newest history; never a time-driven spin.
    pub twist_degrees: f64,
    /// Maximum displacement as a fraction of local radius; at most 0.40.
    pub source_influence: f64,
    /// Soft radial scale of the shared, bounded projected-position mapping.
    pub source_radius: f64,
    /// Fixed orthonormal source axes mapped to the shell's Y and Z displacements.
    ///
    /// The default uses original X/Y. A seed-specific fitted plane must be
    /// computed once from the complete source and stored explicitly in the recipe.
    pub source_axes: [V3; 2],
    /// Shared actual 3D pair distance giving half the closeness response.
    pub pair_distance_scale: f64,
    /// Outward arch height as a fraction of local radius at large separation.
    pub bow_base: f64,
    /// Additional arch height as a fraction of radius at a close encounter.
    pub bow_response: f64,
    /// Maximum smooth central compression of across-panel material coordinates.
    pub compression: f64,
    /// Enable AB, BC, and CA independently for material and composition proofs.
    pub panels: [bool; 3],
    /// Permanent staggered windows, in AB/BC/CA order.
    pub apertures: [LoomAperture; 3],
    /// Longitudinal yarn groups across each panel.
    pub warp_groups: usize,
    /// Crosswise yarn rows along the history axis.
    pub weft_groups: usize,
    /// Fine fibers in each longitudinal yarn group.
    pub warp_fibers: usize,
    /// Fine fibers in each crosswise yarn group.
    pub weft_fibers: usize,
    /// Longitudinal sampling intervals before clipping at windows.
    pub warp_segments: usize,
    /// Across-panel sampling intervals before clipping at windows.
    pub weft_segments: usize,
    /// Sampling intervals around each complete aperture braid.
    pub rim_segments: usize,
    /// Width of each yarn group relative to its material-space lattice pitch.
    pub bundle_width: f64,
    /// Individual fiber radius at a local rail radius of 1.1 world units.
    pub fiber_radius: f64,
    /// Stable fractional radius variation between fine fibers.
    pub radius_variation: f64,
    /// Alternating crossing height at a rail radius of 1.1 world units.
    pub crossing_lift: f64,
    /// End taper length in units of the opposing yarn group's cell pitch.
    pub edge_taper_cells: f64,
    /// Fraction of the oldest end over which all fibers and braids feather away.
    pub terminal_feather_old: f64,
    /// Fraction of the newest end over which all fibers and braids feather away.
    pub terminal_feather_new: f64,
    /// Number of fine fibers forming each quiet boundary cable.
    pub rim_fibers: usize,
    /// Radius of the boundary cable's braided centerlines at reference scale.
    pub rim_braid_radius: f64,
    /// Individual boundary fiber radius at reference scale.
    pub rim_fiber_radius: f64,
    /// Braid revolutions along a complete boundary cable.
    pub rim_turns: f64,
    /// Relative reflected brightness of boundary fibers.
    pub rim_brightness: f64,
    /// Mean optical depth shared by individual fine fibers.
    pub fiber_optical_depth: f64,
    /// Relative reflected brightness of the fine fibers.
    pub fiber_brightness: f64,
    /// Small stable dye variation within each body's yarn families.
    pub tint_variation: f64,
    /// Fraction of yarn groups receiving the restrained blue accent.
    pub accent_fraction: f64,
    /// Fraction of other yarn groups receiving a thin metallic gilding.
    pub gilded_fraction: f64,
    /// Conductor contribution of the selectively gilded fibers.
    pub gilded_metallic: f64,
    /// Fixed detail seed; never changes the recorded source motion.
    pub detail_seed: u64,
    /// Champagne, bronze, and pearl materials in original body order.
    pub materials: [Material; 3],
    /// Secondary dye used sparingly in fine yarn groups.
    pub accent_material: Material,
}

impl Default for LoomConfig {
    fn default() -> Self {
        Self {
            profile: LoomProfile::Conch,
            cross_section: LoomCrossSection::Polar,
            history_fraction: 0.22,
            axis_length: 7.2,
            centerline_y: 0.42,
            centerline_z: 0.32,
            radius_tail: 0.06,
            radius_mouth: 0.15,
            radius_bulge: 1.25,
            radius_min: 0.06,
            radius_max: 1.35,
            rail_rotation_degrees: 0.0,
            twist_degrees: 180.0,
            source_influence: 0.22,
            source_radius: 2.0,
            source_axes: [V3::new(1.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0)],
            pair_distance_scale: 1.4,
            bow_base: 0.055,
            bow_response: 0.16,
            compression: 0.40,
            panels: [true; 3],
            apertures: [
                LoomAperture::default(),
                LoomAperture {
                    center: [0.59, 0.53],
                    half_axes: [0.35, 0.40],
                    tilt_degrees: 8.0,
                    ..LoomAperture::default()
                },
                LoomAperture {
                    center: [0.50, 0.48],
                    half_axes: [0.42, 0.36],
                    tilt_degrees: -5.0,
                    ..LoomAperture::default()
                },
            ],
            warp_groups: 128,
            weft_groups: 256,
            warp_fibers: 2,
            weft_fibers: 2,
            warp_segments: 3072,
            weft_segments: 512,
            rim_segments: 1536,
            bundle_width: 0.34,
            fiber_radius: 0.00035,
            radius_variation: 0.16,
            crossing_lift: 0.0015,
            edge_taper_cells: 0.38,
            terminal_feather_old: 0.05,
            terminal_feather_new: 0.08,
            rim_fibers: 5,
            rim_braid_radius: 0.0012,
            rim_fiber_radius: 0.00046,
            rim_turns: 72.0,
            rim_brightness: 0.72,
            fiber_optical_depth: 0.86,
            fiber_brightness: 1.10,
            tint_variation: 0.12,
            accent_fraction: 0.035,
            gilded_fraction: 0.06,
            gilded_metallic: 0.82,
            detail_seed: 834_927,
            materials: default_materials(),
            accent_material: Material {
                front_color: V3::new(0.21, 0.37, 0.48),
                back_color: V3::new(0.12, 0.22, 0.30),
                roughness: 0.34,
                anisotropy: 0.80,
                sheen: 0.43,
                metallic: 0.08,
                fiber_strength: 0.0,
                ..Material::default()
            },
        }
    }
}

fn default_materials() -> [Material; 3] {
    [
        (V3::new(0.82, 0.68, 0.45), 0.68),
        (V3::new(0.50, 0.28, 0.13), 0.75),
        (V3::new(0.78, 0.73, 0.65), 0.15),
    ]
    .map(|(color, metallic)| Material {
        front_color: color,
        back_color: color * 0.60,
        optical_depth: V3::new(0.72, 0.79, 0.90),
        roughness: 0.32,
        anisotropy: 0.80,
        sheen: 0.45,
        fiber_strength: 0.0,
        metallic,
        ..Material::default()
    })
}

#[derive(Clone, Copy)]
struct Row {
    center: V3,
    center_du: V3,
    rails: [V3; 3],
    derivatives: [V3; 3],
    angles: [f64; 3],
    angles_du: [f64; 3],
    polar_radii: [f64; 3],
    polar_radii_du: [f64; 3],
    closeness: [f64; 3],
    closeness_du: [f64; 3],
    radius: f64,
    radius_du: f64,
}

#[derive(Clone, Copy)]
struct SurfacePoint {
    position: V3,
    du: V3,
    dv: V3,
    normal: V3,
}

struct Field<'a> {
    source: &'a OrbitSeries,
    config: &'a LoomConfig,
    time: f64,
    rows: Vec<Row>,
}

impl<'a> Field<'a> {
    fn new(source: &'a OrbitSeries, time: f64, config: &'a LoomConfig) -> SilkResult<Self> {
        let rows = (0..=config.warp_segments)
            .map(|index| {
                measure_row(source, time, index as f64 / config.warp_segments as f64, config)
            })
            .collect::<SilkResult<Vec<_>>>()?;
        Ok(Self { source, config, time, rows })
    }

    fn row(&self, u: f64) -> SilkResult<Row> {
        let index = u * self.config.warp_segments as f64;
        let nearest = index.round();
        if (nearest - index).abs() < 1e-10 {
            Ok(self.rows[nearest as usize])
        } else {
            measure_row(self.source, self.time, u, self.config)
        }
    }
}

/// Construct one complete woven history object without changing source motion.
///
/// A fully formed opening requires genuine prehistory at least as long as the
/// requested window. Missing source history is an error, never a wrapped or
/// fabricated trajectory. The configuration can expose only AB for a close-up.
pub fn scene(source: &OrbitSeries, time: f64, config: &LoomConfig) -> SilkResult<Scene> {
    validate(source, time, config)?;
    let field = Field::new(source, time, config)?;
    let mut scene = Scene { materials: prepare_materials(config), ..Scene::default() };
    for panel in 0..3 {
        if !config.panels[panel] {
            continue;
        }
        add_panel_threads(&mut scene, &field, panel, true)?;
        add_panel_threads(&mut scene, &field, panel, false)?;
        if config.apertures[panel].enabled {
            add_aperture_braid(&mut scene, &field, panel)?;
        }
        add_end_braid(&mut scene, &field, panel, 0.0)?;
        add_end_braid(&mut scene, &field, panel, 1.0)?;
    }
    for body in 0..3 {
        if config.panels[body] || config.panels[(body + 2) % 3] {
            add_rail_braid(&mut scene, &field, body);
        }
    }
    if scene.strands.iter().any(|strand| {
        !strand.points.iter().all(|point| point.is_finite())
            || !strand.radii.iter().all(|radius| radius.is_finite() && *radius >= 0.0)
    }) {
        return Err("loom geometry exceeded finite position or radius range".into());
    }
    Ok(scene)
}

fn validate(source: &OrbitSeries, time: f64, config: &LoomConfig) -> SilkResult<()> {
    if !time.is_finite() || !(0.0..=1.0).contains(&time) {
        return Err("loom time must be a finite fraction in [0,1]".into());
    }
    let positive = [
        config.history_fraction,
        config.axis_length,
        config.radius_tail,
        config.radius_mouth,
        config.radius_min,
        config.radius_max,
        config.source_radius,
        config.pair_distance_scale,
        config.fiber_radius,
        config.crossing_lift,
        config.edge_taper_cells,
        config.rim_fiber_radius,
        config.fiber_optical_depth,
    ];
    let nonnegative = [
        config.radius_bulge,
        config.bow_base,
        config.bow_response,
        config.rim_braid_radius,
        config.rim_turns,
        config.rim_brightness,
        config.fiber_brightness,
    ];
    if positive.iter().any(|value| !value.is_finite() || *value <= 0.0)
        || nonnegative.iter().any(|value| !value.is_finite() || *value < 0.0)
        || ![
            config.centerline_y,
            config.centerline_z,
            config.rail_rotation_degrees,
            config.twist_degrees,
        ]
        .iter()
        .all(|value| value.is_finite())
    {
        return Err("loom dimensions and optical controls must be finite with valid signs".into());
    }
    if time - config.history_fraction < source.history_start_fraction() {
        return Err(
            "loom needs verified source prehistory covering its full history_fraction".into()
        );
    }
    if !config.source_axes.iter().all(|axis| axis.is_finite())
        || config.source_axes.iter().any(|axis| (axis.length_squared() - 1.0).abs() > 1e-6)
        || config.source_axes[0].dot(config.source_axes[1]).abs() > 1e-6
    {
        return Err("loom source_axes must be fixed orthonormal directions (tolerance 1e-6)".into());
    }
    if !(0.0..=0.40).contains(&config.source_influence)
        || !(0.0..=0.70).contains(&config.compression)
        || !(0.0..=0.65).contains(&config.bundle_width)
        || !(0.0..=0.80).contains(&config.radius_variation)
        || !(0.0..=0.50).contains(&config.tint_variation)
        || !(0.0..=1.0).contains(&config.accent_fraction)
        || !(0.0..=1.0).contains(&config.gilded_fraction)
        || !(0.0..=1.0).contains(&config.gilded_metallic)
        || !(0.0..=0.35).contains(&config.terminal_feather_old)
        || !(0.0..=0.35).contains(&config.terminal_feather_new)
        || config.radius_max < config.radius_min
        || config.radius_mouth < config.radius_tail
        || !config.panels.iter().any(|&enabled| enabled)
    {
        return Err("loom shape, variation, or panel controls exceed their supported range".into());
    }
    if config.warp_groups < 2
        || config.weft_groups < 2
        || config.warp_fibers == 0
        || config.weft_fibers == 0
        || config.rim_fibers == 0
        || config.warp_segments < config.weft_groups.saturating_mul(4)
        || config.weft_segments < config.warp_groups.saturating_mul(3)
        || config.rim_segments < 24
        || config.rim_segments as f64 <= config.rim_turns * 6.0
    {
        return Err(
            "loom fibers need positive counts and enough samples to resolve crossing/braid curves"
                .into(),
        );
    }
    if [
        config.warp_groups,
        config.weft_groups,
        config.warp_fibers,
        config.weft_fibers,
        config.warp_segments,
        config.weft_segments,
        config.rim_segments,
        config.rim_fibers,
    ]
    .iter()
    .any(|&count| count > 1_000_000)
    {
        return Err("loom geometry counts exceed the supported allocation range".into());
    }
    let segments = (config.warp_groups as u128
        * config.warp_fibers as u128
        * (config.warp_segments as u128 + 1)
        + config.weft_groups as u128
            * config.weft_fibers as u128
            * (config.weft_segments as u128 + 1))
        * 3
        + config.rim_fibers as u128
            * (config.rim_segments as u128 + config.warp_segments as u128 + 2)
            * 12;
    if segments > 40_000_000 {
        return Err("loom configuration exceeds forty million pre-clipping control points".into());
    }
    if config.crossing_lift
        <= config.fiber_radius * (1.0 + config.radius_variation)
            / (PI * config.bundle_width * 0.5).cos()
    {
        return Err("loom crossing lift must clear the full opposing fiber radii".into());
    }
    for aperture in config.apertures {
        validate_aperture(aperture)?;
    }
    for material in config.materials.iter().chain(std::iter::once(&config.accent_material)) {
        if !material.front_color.is_finite()
            || !material.back_color.is_finite()
            || !material.optical_depth.is_finite()
            || !material.emission.is_finite()
            || ![
                material.roughness,
                material.anisotropy,
                material.sheen,
                material.fiber_frequency,
                material.fiber_strength,
                material.metallic,
            ]
            .iter()
            .all(|value| value.is_finite())
        {
            return Err("loom materials must contain finite colors and optical values".into());
        }
    }
    Ok(())
}

fn validate_aperture(aperture: LoomAperture) -> SilkResult<()> {
    if !aperture.center.iter().chain(aperture.half_axes.iter()).all(|value| value.is_finite())
        || !aperture.tilt_degrees.is_finite()
        || aperture.half_axes.iter().any(|&value| value <= 0.0)
    {
        return Err("loom apertures require finite centers, positive axes, and finite tilt".into());
    }
    let (sine, cosine) = aperture.tilt_degrees.to_radians().sin_cos();
    let [a, b] = aperture.half_axes;
    let extent = [(a * cosine).hypot(b * sine), (a * sine).hypot(b * cosine)];
    if aperture
        .center
        .iter()
        .zip(extent)
        .any(|(&center, half)| center - half < 0.035 || center + half > 0.965)
    {
        return Err("loom apertures must retain a 0.035 material-space border".into());
    }
    Ok(())
}

fn radius(u: f64, config: &LoomConfig) -> (f64, f64) {
    match config.profile {
        LoomProfile::Conch => {
            let blend = u * u * u * (10.0 + u * (-15.0 + 6.0 * u));
            let blend_du = 30.0 * u * u * (1.0 - u).powi(2);
            let belly = u.powi(3) * (1.0 - u).powi(2) / 0.03456;
            let belly_du = u * u * (1.0 - u) * (3.0 - 5.0 * u) / 0.03456;
            (
                config.radius_tail
                    + (config.radius_mouth - config.radius_tail) * blend
                    + config.radius_bulge * belly,
                (config.radius_mouth - config.radius_tail) * blend_du
                    + config.radius_bulge * belly_du,
            )
        }
        LoomProfile::Spindle => (
            config.radius_min + (config.radius_max - config.radius_min) * (PI * u).sin().powi(2),
            PI * (config.radius_max - config.radius_min) * (TAU * u).sin(),
        ),
    }
}

fn mapped_measurement(
    source: &OrbitSeries,
    fraction: f64,
    config: &LoomConfig,
) -> SilkResult<([V3; 3], [f64; 3])> {
    let frame = source.sample(fraction).ok_or("loom source sample is outside verified history")?;
    let positions = frame.bodies.map(|body| body.position);
    let q = positions.map(|point| {
        let y = point.dot(config.source_axes[0]);
        let z = point.dot(config.source_axes[1]);
        let denominator = config.source_radius.hypot(y.hypot(z));
        V3::new(0.0, y / denominator, z / denominator)
    });
    let closeness = std::array::from_fn(|pair| {
        let difference = positions[(pair + 1) % 3] - positions[pair];
        let distance = difference.x.hypot(difference.y).hypot(difference.z);
        let inverse = 1.0 / (distance / config.pair_distance_scale).hypot(1.0);
        inverse * inverse
    });
    Ok((q, closeness))
}

fn measure_row(source: &OrbitSeries, time: f64, u: f64, config: &LoomConfig) -> SilkResult<Row> {
    let fraction = if u == 1.0 { time } else { time - config.history_fraction * (1.0 - u) };
    let (q, closeness) = mapped_measurement(source, fraction, config)?;
    let left = (fraction - 1e-6).max(source.history_start_fraction());
    let right = (fraction + 1e-6).min(1.0);
    let (q_left, c_left) = mapped_measurement(source, left, config)?;
    let (q_right, c_right) = mapped_measurement(source, right, config)?;
    let derivative_scale = config.history_fraction / (right - left);
    let (radius, radius_du) = radius(u, config);
    let center = V3::new(
        config.axis_length * (u - 0.5),
        config.centerline_y * (TAU * u).sin(),
        config.centerline_z * (PI * u).sin().powi(2),
    );
    let center_du = V3::new(
        config.axis_length,
        TAU * config.centerline_y * (TAU * u).cos(),
        PI * config.centerline_z * (TAU * u).sin(),
    );
    let twist_du = config.twist_degrees.to_radians();
    let angles: [f64; 3] = std::array::from_fn(|body| {
        PI * 0.5
            + TAU * body as f64 / 3.0
            + config.rail_rotation_degrees.to_radians()
            + twist_du * (u - 0.5)
    });
    let rest: [V3; 3] = angles.map(|angle| {
        let (sine, cosine) = angle.sin_cos();
        V3::new(0.0, cosine, sine)
    });
    let directions: [V3; 3] =
        std::array::from_fn(|body| rest[body] + q[body] * config.source_influence);
    let directions_du: [V3; 3] = std::array::from_fn(|body| {
        V3::new(0.0, -rest[body].z, rest[body].y) * twist_du
            + (q_right[body] - q_left[body]) * (config.source_influence * derivative_scale)
    });
    Ok(Row {
        center,
        center_du,
        rails: std::array::from_fn(|body| center + directions[body] * radius),
        derivatives: std::array::from_fn(|body| {
            center_du + directions[body] * radius_du + directions_du[body] * radius
        }),
        angles: std::array::from_fn(|body| {
            angles[body]
                + (rest[body].y * directions[body].z - rest[body].z * directions[body].y)
                    .atan2(rest[body].dot(directions[body]))
        }),
        angles_du: std::array::from_fn(|body| {
            (directions[body].y * directions_du[body].z
                - directions[body].z * directions_du[body].y)
                / directions[body].length_squared()
        }),
        polar_radii: directions.map(|direction| direction.length() * radius),
        polar_radii_du: std::array::from_fn(|body| {
            radius_du * directions[body].length()
                + radius * directions[body].dot(directions_du[body]) / directions[body].length()
        }),
        closeness,
        closeness_du: std::array::from_fn(|pair| (c_right[pair] - c_left[pair]) * derivative_scale),
        radius,
        radius_du,
    })
}

fn surface(row: &Row, panel: usize, v: f64, config: &LoomConfig) -> SurfacePoint {
    match config.cross_section {
        LoomCrossSection::Bowed => bowed_surface(row, panel, v, config),
        LoomCrossSection::Polar => polar_surface(row, panel, v, config),
    }
}

fn polar_surface(row: &Row, panel: usize, v: f64, config: &LoomConfig) -> SurfacePoint {
    let next = (panel + 1) % 3;
    let span = row.angles[next] - row.angles[panel] + if next == 0 { TAU } else { 0.0 };
    let span_du = row.angles_du[next] - row.angles_du[panel];
    let k = config.compression * row.closeness[panel];
    let f = v + k * (TAU * v).sin() / TAU;
    let f_v = 1.0 + k * (TAU * v).cos();
    let f_u = config.compression * row.closeness_du[panel] * (TAU * v).sin() / TAU;
    let angle = row.angles[panel] + span * f;
    let angle_u = row.angles_du[panel] + span_du * f + span * f_u;
    let angle_v = span * f_v;
    let (sine, cosine) = angle.sin_cos();
    let radial = V3::new(0.0, cosine, sine);
    let angular = V3::new(0.0, -sine, cosine);
    // Zero endpoint slope makes adjacent panels share a smooth radial tangent.
    let blend = smoothstep(f);
    let blend_f = 30.0 * f * f * (1.0 - f).powi(2);
    let delta_radius = row.polar_radii[next] - row.polar_radii[panel];
    let delta_radius_du = row.polar_radii_du[next] - row.polar_radii_du[panel];
    let bow = row.radius * (config.bow_base + config.bow_response * row.closeness[panel]);
    let bow_du = row.radius_du * (config.bow_base + config.bow_response * row.closeness[panel])
        + row.radius * config.bow_response * row.closeness_du[panel];
    let g = (PI * f).sin().powi(2);
    let g_f = PI * (TAU * f).sin();
    let r = row.polar_radii[panel] + delta_radius * blend + bow * g;
    let r_u = row.polar_radii_du[panel]
        + delta_radius_du * blend
        + delta_radius * blend_f * f_u
        + bow_du * g
        + bow * g_f * f_u;
    let r_v = (delta_radius * blend_f + bow * g_f) * f_v;
    let du = row.center_du + radial * r_u + angular * (r * angle_u);
    let dv = radial * r_v + angular * (r * angle_v);
    SurfacePoint { position: row.center + radial * r, du, dv, normal: dv.cross(du).normalized() }
}

fn bowed_surface(row: &Row, panel: usize, v: f64, config: &LoomConfig) -> SurfacePoint {
    let next = (panel + 1) % 3;
    let difference = row.rails[next] - row.rails[panel];
    let difference_du = row.derivatives[next] - row.derivatives[panel];
    let direction = difference.normalized();
    let normal = direction.cross(TIME_AXIS);
    let direction_du =
        (difference_du - direction * direction.dot(difference_du)) / difference.length();
    let normal_du = direction_du.cross(TIME_AXIS);
    let k = config.compression * row.closeness[panel];
    let f = v + k * (TAU * v).sin() / TAU;
    let f_v = 1.0 + k * (TAU * v).cos();
    let f_u = config.compression * row.closeness_du[panel] * (TAU * v).sin() / TAU;
    let bow = row.radius * (config.bow_base + config.bow_response * row.closeness[panel]);
    let bow_du = row.radius_du * (config.bow_base + config.bow_response * row.closeness[panel])
        + row.radius * config.bow_response * row.closeness_du[panel];
    let g = (PI * f).sin().powi(2);
    let g_f = PI * (TAU * f).sin();
    let du = row.derivatives[panel]
        + difference * f_u
        + difference_du * f
        + normal * (bow_du * g + bow * g_f * f_u)
        + normal_du * (bow * g);
    let dv = (difference + normal * (bow * g_f)) * f_v;
    SurfacePoint {
        position: row.rails[panel] + difference * f + normal * (bow * g),
        du,
        dv,
        normal: dv.cross(du).normalized(),
    }
}

fn surviving_intervals(aperture: LoomAperture, warp: bool, fixed: f64) -> Vec<[f64; 2]> {
    if !aperture.enabled {
        return vec![[0.0, 1.0]];
    }
    let (sine, cosine) = aperture.tilt_degrees.to_radians().sin_cos();
    let origin = if warp { [0.0, fixed] } else { [fixed, 0.0] };
    let x = origin[0] - aperture.center[0];
    let y = origin[1] - aperture.center[1];
    let [a, b] = aperture.half_axes;
    let offset = [(x * cosine + y * sine) / a, (-x * sine + y * cosine) / b];
    let direction = if warp { [cosine / a, -sine / b] } else { [sine / a, cosine / b] };
    let qa = direction[0].powi(2) + direction[1].powi(2);
    let qb = 2.0 * (direction[0] * offset[0] + direction[1] * offset[1]);
    let qc = offset[0].powi(2) + offset[1].powi(2) - 1.0;
    let discriminant = qb * qb - 4.0 * qa * qc;
    if discriminant <= 0.0 {
        return vec![[0.0, 1.0]];
    }
    let root = discriminant.sqrt();
    let low = ((-qb - root) / (2.0 * qa)).max(0.0);
    let high = ((-qb + root) / (2.0 * qa)).min(1.0);
    if low >= high {
        return vec![[0.0, 1.0]];
    }
    let mut intervals = Vec::with_capacity(2);
    if low > 1e-12 {
        intervals.push([0.0, low]);
    }
    if high < 1.0 - 1e-12 {
        intervals.push([high, 1.0]);
    }
    intervals
}

fn curve_parameters(interval: [f64; 2], segments: usize) -> Vec<f64> {
    let [start, end] = interval;
    let first = (start * segments as f64).floor() as usize + 1;
    let last = (end * segments as f64).ceil() as usize;
    let mut values = Vec::with_capacity(last.saturating_sub(first) + 2);
    values.push(start);
    for index in first..last {
        let value = index as f64 / segments as f64;
        if value > start + 1e-13 && value < end - 1e-13 {
            values.push(value);
        }
    }
    values.push(end);
    values
}

fn smoothstep(value: f64) -> f64 {
    let u = value.clamp(0.0, 1.0);
    u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
}

fn terminal_feather(u: f64, config: &LoomConfig) -> f64 {
    let old = if config.terminal_feather_old > 0.0 {
        smoothstep(u / config.terminal_feather_old)
    } else {
        1.0
    };
    let new = if config.terminal_feather_new > 0.0 {
        smoothstep((1.0 - u) / config.terminal_feather_new)
    } else {
        1.0
    };
    old * new
}

fn crossing_height(
    warp: bool,
    group: usize,
    parameter: f64,
    radius: f64,
    config: &LoomConfig,
) -> f64 {
    let count = if warp { config.weft_groups } else { config.warp_groups };
    let sign = if warp { 1.0 } else { -1.0 };
    sign * config.crossing_lift * radius / REFERENCE_RADIUS
        * (PI * (count as f64 * parameter - 0.5 + group as f64)).cos()
}

fn add_panel_threads(
    scene: &mut Scene,
    field: &Field<'_>,
    panel: usize,
    warp: bool,
) -> SilkResult<()> {
    let config = field.config;
    let (groups, fibers, segments, opposing) = if warp {
        (config.warp_groups, config.warp_fibers, config.warp_segments, config.weft_groups)
    } else {
        (config.weft_groups, config.weft_fibers, config.weft_segments, config.warp_groups)
    };
    for group in 0..groups {
        for fiber in 0..fibers {
            let offset = if fibers > 1 { fiber as f64 / (fibers - 1) as f64 - 0.5 } else { 0.0 };
            let fixed = (group as f64 + 0.5 + offset * config.bundle_width) / groups as f64;
            let group_id = (panel as u64 * 131_071 + u64::from(warp) * 65_537 + group as u64 * 257)
                ^ config.detail_seed;
            let identity = group_id.wrapping_add(fiber as u64);
            let radius_factor = 1.0 + config.radius_variation * (2.0 * noise(identity) - 1.0);
            let dye = if warp { panel } else { (panel + 1) % 3 };
            let material = material_index(dye, group_id, identity, config);
            let fixed_row = if warp { None } else { Some(field.row(fixed)?) };
            for interval in surviving_intervals(config.apertures[panel], warp, fixed) {
                let parameters = curve_parameters(interval, segments);
                let mut strand = Strand {
                    points: Vec::with_capacity(parameters.len()),
                    radii: Vec::with_capacity(parameters.len()),
                    material,
                };
                for parameter in parameters {
                    let row = if let Some(row) = fixed_row { row } else { field.row(parameter)? };
                    let v = if warp { fixed } else { parameter };
                    let point = surface(&row, panel, v, config);
                    let edge_distance = (parameter - interval[0]).min(interval[1] - parameter);
                    let u = if warp { parameter } else { fixed };
                    let taper =
                        smoothstep(edge_distance * opposing as f64 / config.edge_taper_cells)
                            * terminal_feather(u, config);
                    let lift = crossing_height(warp, group, parameter, row.radius, config) * taper;
                    strand.points.push(point.position + point.normal * lift);
                    strand.radii.push(
                        config.fiber_radius * radius_factor * row.radius / REFERENCE_RADIUS * taper,
                    );
                }
                if warp && strand.points.windows(2).any(|pair| pair[1].x <= pair[0].x) {
                    return Err("loom crossing lift folds a warp fiber along the time axis; reduce lift or increase axis_length".into());
                }
                scene.strands.push(strand);
            }
        }
    }
    Ok(())
}

fn add_aperture_braid(scene: &mut Scene, field: &Field<'_>, panel: usize) -> SilkResult<()> {
    let config = field.config;
    let aperture = config.apertures[panel];
    let (sine, cosine) = aperture.tilt_degrees.to_radians().sin_cos();
    let mut cable = Vec::with_capacity(config.rim_segments + 1);
    for index in 0..=config.rim_segments {
        let parameter = index as f64 / config.rim_segments as f64;
        let angle = TAU * parameter;
        let (s, c) = angle.sin_cos();
        let [a, b] = aperture.half_axes;
        let u = aperture.center[0] + a * c * cosine - b * s * sine;
        let v = aperture.center[1] + a * c * sine + b * s * cosine;
        let row = field.row(u)?;
        let point = surface(&row, panel, v, config);
        let du = -a * s * cosine - b * c * sine;
        let dv = -a * s * sine + b * c * cosine;
        let tangent = (point.du * du + point.dv * dv).normalized();
        cable.push((point.position, point.normal, tangent.cross(point.normal), row.radius, u));
    }
    add_braid(scene, &cable, panel, config.rim_turns.round(), config);
    Ok(())
}

fn add_end_braid(scene: &mut Scene, field: &Field<'_>, panel: usize, u: f64) -> SilkResult<()> {
    let config = field.config;
    let row = field.row(u)?;
    let mut cable = Vec::with_capacity(config.weft_segments + 1);
    for index in 0..=config.weft_segments {
        let v = index as f64 / config.weft_segments as f64;
        let point = surface(&row, panel, v, config);
        cable.push((
            point.position,
            point.normal,
            point.dv.normalized().cross(point.normal),
            row.radius,
            u,
        ));
    }
    add_braid(scene, &cable, panel, config.rim_turns * 0.20, config);
    Ok(())
}

fn add_rail_braid(scene: &mut Scene, field: &Field<'_>, body: usize) {
    let mut cable = Vec::with_capacity(field.rows.len());
    for (index, row) in field.rows.iter().enumerate() {
        let tangent = row.derivatives[body].normalized();
        let normal = (V3::new(0.0, 1.0, 0.0) - tangent * tangent.y).normalized();
        let u = index as f64 / (field.rows.len() - 1) as f64;
        cable.push((row.rails[body], normal, tangent.cross(normal), row.radius, u));
    }
    add_braid(scene, &cable, body, field.config.rim_turns, field.config);
}

fn add_braid(
    scene: &mut Scene,
    cable: &[(V3, V3, V3, f64, f64)],
    dye: usize,
    turns: f64,
    config: &LoomConfig,
) {
    for fiber in 0..config.rim_fibers {
        let mut strand = Strand {
            points: Vec::with_capacity(cable.len()),
            radii: Vec::with_capacity(cable.len()),
            material: 7 * TINTS + dye,
        };
        for (index, &(position, normal, binormal, radius, u)) in cable.iter().enumerate() {
            let parameter = index as f64 / (cable.len() - 1) as f64;
            let phase = TAU * (turns * parameter + fiber as f64 / config.rim_fibers as f64);
            let (sine, cosine) = phase.sin_cos();
            let scale = radius / REFERENCE_RADIUS * terminal_feather(u, config);
            strand.points.push(
                position + (normal * cosine + binormal * sine) * (config.rim_braid_radius * scale),
            );
            strand.radii.push(config.rim_fiber_radius * scale);
        }
        scene.strands.push(strand);
    }
}

fn prepare_materials(config: &LoomConfig) -> Vec<Material> {
    let mut materials = Vec::with_capacity(7 * TINTS + 3);
    for source in config.materials.iter().chain(std::iter::once(&config.accent_material)) {
        for tint in 0..TINTS {
            let mut material = source.clone();
            let variation = config.tint_variation * (2.0 * tint as f64 / (TINTS - 1) as f64 - 1.0);
            let dye =
                V3::new(1.0 + variation * 0.35, 1.0 + variation * 0.10, 1.0 - variation * 0.35);
            material.front_color = material.front_color.hadamard(dye) * config.fiber_brightness;
            material.back_color = material.back_color.hadamard(dye) * config.fiber_brightness;
            let mean =
                (material.optical_depth.x + material.optical_depth.y + material.optical_depth.z)
                    / 3.0;
            material.optical_depth = if mean > 1e-12 {
                material.optical_depth * (config.fiber_optical_depth / mean)
            } else {
                V3::new(1.0, 1.0, 1.0) * config.fiber_optical_depth
            };
            materials.push(material);
        }
    }
    for dye in 0..3 {
        for tint in 0..TINTS {
            let mut material = materials[dye * TINTS + tint].clone();
            material.metallic = config.gilded_metallic;
            material.roughness = material.roughness.max(0.28);
            materials.push(material);
        }
    }
    for source in &config.materials {
        let mut material = source.clone();
        material.front_color *= config.rim_brightness;
        material.back_color *= config.rim_brightness;
        material.sheen *= 0.8;
        materials.push(material);
    }
    materials
}

fn material_index(dye: usize, group: u64, identity: u64, config: &LoomConfig) -> usize {
    let bank = if noise(group ^ 0x95a4_61b3_9c70_582d) < config.accent_fraction {
        3
    } else if noise(group ^ 0xa251_97bd_05fc_368e) < config.gilded_fraction {
        4 + dye
    } else {
        dye
    };
    bank * TINTS + (noise(identity ^ 0x194b_37d5_e7a2_604f) * TINTS as f64) as usize
}

fn noise(mut value: u64) -> f64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value >> 11) as f64 / (1_u64 << 53) as f64
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
                    V3::new(angle.cos(), angle.sin(), 0.35 * (2.0 * angle + t).sin())
                })
            })
            .collect();
        OrbitSeries::new(&OrbitData {
            seed: "loom-test".to_owned(),
            dt: 0.01,
            masses: [1.0; 3],
            samples,
            provenance: serde_json::json!({"synthetic_unit_fixture":true}),
        })
        .unwrap()
    }

    fn small_config() -> LoomConfig {
        LoomConfig {
            warp_groups: 4,
            weft_groups: 6,
            warp_fibers: 3,
            weft_fibers: 3,
            warp_segments: 32,
            weft_segments: 24,
            rim_segments: 64,
            rim_fibers: 3,
            rim_turns: 6.0,
            ..LoomConfig::default()
        }
    }

    #[test]
    fn alternating_crossings_have_real_opposite_heights_and_fiber_clearance() {
        let config = LoomConfig::default();
        for m in 0..config.warp_groups {
            for n in 0..config.weft_groups {
                let u = (n as f64 + 0.5) / config.weft_groups as f64;
                let v = (m as f64 + 0.5) / config.warp_groups as f64;
                let warp = crossing_height(true, m, u, REFERENCE_RADIUS, &config);
                let weft = crossing_height(false, n, v, REFERENCE_RADIUS, &config);
                assert!((warp + weft).abs() < 1e-14);
                assert!((warp.abs() - config.crossing_lift).abs() < 1e-14);
                assert!(
                    (warp - weft).abs()
                        > 2.0 * config.fiber_radius * (1.0 + config.radius_variation)
                );
                assert_eq!(warp.is_sign_positive(), (n + m) % 2 == 0);
            }
        }
    }

    #[test]
    fn rotated_apertures_cut_separate_components_at_the_exact_boundary() {
        let aperture = LoomConfig::default().apertures[1];
        for warp in [true, false] {
            let fixed = aperture.center[usize::from(warp)];
            let pieces = surviving_intervals(aperture, warp, fixed);
            assert_eq!(pieces.len(), 2);
            assert!(pieces[0][1] < pieces[1][0]);
            for boundary in [pieces[0][1], pieces[1][0]] {
                let [u, v] = if warp { [boundary, fixed] } else { [fixed, boundary] };
                let (sine, cosine) = aperture.tilt_degrees.to_radians().sin_cos();
                let x = u - aperture.center[0];
                let y = v - aperture.center[1];
                let ellipse = ((x * cosine + y * sine) / aperture.half_axes[0]).powi(2)
                    + ((-x * sine + y * cosine) / aperture.half_axes[1]).powi(2);
                assert!((ellipse - 1.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn source_rails_keep_fixed_chronology_and_positive_panel_area() {
        let source = source();
        for cross_section in [LoomCrossSection::Bowed, LoomCrossSection::Polar] {
            let config = LoomConfig { cross_section, ..small_config() };
            for index in 0..=32 {
                let u = f64::from(index) / 32.0;
                let row = measure_row(&source, 0.65, u, &config).unwrap();
                for panel in 0..3 {
                    assert_eq!(row.rails[panel].x, config.axis_length * (u - 0.5));
                    let separation = (row.rails[(panel + 1) % 3] - row.rails[panel]).length();
                    assert!(
                        separation >= row.radius * (3.0_f64.sqrt() - 2.0 * config.source_influence)
                    );
                    for v in [0.0, 0.25, 0.50, 0.75, 1.0] {
                        let point = surface(&row, panel, v, &config);
                        assert_eq!(point.position.x, config.axis_length * (u - 0.5));
                        assert_eq!(point.du.x, config.axis_length);
                        assert_eq!(point.dv.x, 0.0);
                        let transverse_bound = match cross_section {
                            LoomCrossSection::Bowed => {
                                3.0_f64.sqrt() - 2.0 * config.source_influence
                            }
                            LoomCrossSection::Polar => {
                                (1.0 - config.source_influence)
                                    * (TAU / 3.0 - 2.0 * config.source_influence.asin())
                            }
                        };
                        let lower = config.axis_length
                            * (1.0 - config.compression)
                            * row.radius
                            * transverse_bound;
                        assert!(point.du.cross(point.dv).length() >= lower);
                        assert!((point.normal.length() - 1.0).abs() < 1e-12);
                    }
                }
            }
        }
    }

    #[test]
    fn polar_shell_is_round_and_shared_rails_have_smooth_surface_normals() {
        let source = source();
        let config = LoomConfig {
            source_influence: 0.0,
            bow_base: 0.0,
            bow_response: 0.0,
            compression: 0.0,
            twist_degrees: 540.0,
            ..small_config()
        };
        for index in 0..=20 {
            let u = f64::from(index) / 20.0;
            let row = measure_row(&source, 0.7, u, &config).unwrap();
            for panel in 0..3 {
                for v in [0.0, 0.17, 0.50, 0.83, 1.0] {
                    let point = surface(&row, panel, v, &config);
                    assert!(((point.position - row.center).length() - row.radius).abs() < 1e-12);
                }
                let finish = surface(&row, panel, 1.0, &config);
                let start = surface(&row, (panel + 1) % 3, 0.0, &config);
                assert!((finish.position - start.position).length() < 1e-12);
                assert!(finish.normal.dot(start.normal) > 1.0 - 1e-12);
            }
        }
    }

    #[test]
    fn polar_surface_derivatives_follow_the_actual_geometry() {
        let source = source();
        let config = small_config();
        let h = 1e-6;
        for u in [0.08, 0.3, 0.57, 0.91] {
            let row = measure_row(&source, 0.8, u, &config).unwrap();
            for panel in 0..3 {
                for v in [0.1, 0.43, 0.8] {
                    let point = surface(&row, panel, v, &config);
                    let before = measure_row(&source, 0.8, u - h, &config).unwrap();
                    let after = measure_row(&source, 0.8, u + h, &config).unwrap();
                    let du = (surface(&after, panel, v, &config).position
                        - surface(&before, panel, v, &config).position)
                        / (2.0 * h);
                    let dv = (surface(&row, panel, v + h, &config).position
                        - surface(&row, panel, v - h, &config).position)
                        / (2.0 * h);
                    assert!((point.du - du).length() < 1e-6);
                    assert!((point.dv - dv).length() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn geometry_is_repeatable_finite_and_continuous_between_frames() {
        let source = source();
        let config = small_config();
        let a = scene(&source, 0.6, &config).unwrap();
        let b = scene(&source, 0.6, &config).unwrap();
        let next = scene(&source, 0.600001, &config).unwrap();
        assert!(!a.strands.is_empty());
        assert_eq!(a.strands.len(), next.strands.len());
        for ((first, repeat), later) in a.strands.iter().zip(&b.strands).zip(&next.strands) {
            assert_eq!(first.points, repeat.points);
            assert_eq!(first.radii, repeat.radii);
            assert_eq!(first.material, repeat.material);
            assert_eq!(first.points.len(), first.radii.len());
            assert_eq!(first.points.len(), later.points.len());
            assert!(first.material < a.materials.len());
            for (&point, &following) in first.points.iter().zip(&later.points) {
                assert!(point.is_finite());
                assert!((point - following).length() < 1e-3);
            }
            assert!(first.radii.iter().all(|radius| radius.is_finite() && *radius >= 0.0));
        }
    }

    #[test]
    fn profile_derivatives_and_bounded_source_mapping_are_consistent() {
        let source = source();
        for profile in [LoomProfile::Conch, LoomProfile::Spindle] {
            let config = LoomConfig { profile, ..small_config() };
            for index in 1..100 {
                let u = f64::from(index) / 100.0;
                let (r, derivative) = radius(u, &config);
                let numerical = (radius(u + 1e-6, &config).0 - radius(u - 1e-6, &config).0) / 2e-6;
                assert!(r >= config.radius_tail.min(config.radius_min));
                assert!((derivative - numerical).abs() < 1e-7);
                let (q, closeness) = mapped_measurement(&source, u, &config).unwrap();
                assert!(q.iter().all(|point| point.x == 0.0 && point.length() < 1.0));
                assert!(closeness.iter().all(|c| (0.0..=1.0).contains(c)));
            }
        }
    }

    #[test]
    fn panel_proofs_and_missing_history_are_explicit() {
        let source = source();
        let config = small_config();
        let complete = scene(&source, 0.7, &config).unwrap();
        let single =
            scene(&source, 0.7, &LoomConfig { panels: [true, false, false], ..config.clone() })
                .unwrap();
        assert!(single.strands.len() < complete.strands.len());
        assert!(single.strands.len() > complete.strands.len() / 3);
        assert!(scene(&source, 0.0, &config).is_err());
        assert!(scene(&source, f64::NAN, &config).is_err());
        assert!(scene(&source, 0.7, &LoomConfig { source_influence: 0.5, ..config }).is_err());
        assert!(serde_json::from_str::<LoomConfig>("{\"warp_gropus\":56}").is_err());
    }

    #[test]
    fn terminal_feather_removes_end_cuffs_and_preserves_interior_yarns() {
        let source = source();
        let config = small_config();
        assert_eq!(terminal_feather(0.0, &config), 0.0);
        assert_eq!(terminal_feather(1.0, &config), 0.0);
        assert_eq!(terminal_feather(0.5, &config), 1.0);
        assert!(terminal_feather(1e-6, &config) < 1e-12);
        let h = 1e-6;
        let join = config.terminal_feather_old;
        assert!((terminal_feather(join, &config) - terminal_feather(join - h, &config)) / h < 1e-6);

        let field = Field::new(&source, 0.7, &config).unwrap();
        let mut boundaries = Scene::default();
        add_end_braid(&mut boundaries, &field, 0, 0.0).unwrap();
        add_end_braid(&mut boundaries, &field, 0, 1.0).unwrap();
        assert!(
            boundaries.strands.iter().flat_map(|strand| &strand.radii).all(|&radius| radius == 0.0)
        );
        let mut rails = Scene::default();
        add_rail_braid(&mut rails, &field, 0);
        for strand in rails.strands {
            assert_eq!(strand.radii[0], 0.0);
            assert_eq!(*strand.radii.last().unwrap(), 0.0);
            assert!(strand.radii[strand.radii.len() / 2] > 0.0);
        }

        let unfeathered =
            LoomConfig { terminal_feather_old: 0.0, terminal_feather_new: 0.0, ..config };
        assert_eq!(terminal_feather(0.0, &unfeathered), 1.0);
        assert_eq!(terminal_feather(1.0, &unfeathered), 1.0);
        let field = Field::new(&source, 0.7, &unfeathered).unwrap();
        let mut boundaries = Scene::default();
        add_end_braid(&mut boundaries, &field, 0, 1.0).unwrap();
        assert!(
            boundaries.strands.iter().flat_map(|strand| &strand.radii).all(|&radius| radius > 0.0)
        );
    }

    #[test]
    fn fixed_source_plane_captures_x_motion_without_changing_physical_pair_measurements() {
        let source = source();
        let config = small_config();
        let yz = LoomConfig {
            source_axes: [V3::new(0.0, 1.0, 0.0), V3::new(0.0, 0.0, 1.0)],
            ..config.clone()
        };
        let (xy_points, xy_pairs) = mapped_measurement(&source, 0.7, &config).unwrap();
        let (yz_points, yz_pairs) = mapped_measurement(&source, 0.7, &yz).unwrap();
        assert_eq!(xy_pairs, yz_pairs);
        assert_ne!(xy_points, yz_points);
        let frame = source.sample(0.7).unwrap();
        for (body, mapped) in frame.bodies.iter().zip(xy_points) {
            let point = body.position;
            let denominator = config.source_radius.hypot(point.x.hypot(point.y));
            assert_eq!(mapped, V3::new(0.0, point.x / denominator, point.y / denominator));
        }
        let invalid = LoomConfig { source_axes: [V3::new(1.0, 0.0, 0.0); 2], ..config };
        assert!(scene(&source, 0.7, &invalid).is_err());
    }
}
