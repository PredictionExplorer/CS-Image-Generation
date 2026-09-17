//! A permanent, open growth surface driven by the recorded triangle's shape.
//!
//! This is an explicit shape encoding, not a displaced copy of the trajectories.
//! Signals are prepared over the complete recording once. A growing object reveals
//! fixed canonical rings and one terminal ring; previous surface positions never
//! depend on the current endpoint or film frame rate. The authored starter lip
//! exists at source time zero. A fixed C2 source clock eases its join and is
//! inverted to place the requested source endpoint exactly up to rounding.
//! Componentwise PCHIP interpolation preserves the frozen signals' local ranges.
//! Wall thickness, rounded long edges and end caps
//! make the open sheet a closed material volume for offline subsurface rendering.

use crate::atelier::{OrbitSeries, SilkResult, V3};
use crate::remaining::mesh::Mesh;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, TAU};

/// Frozen source mapping and authored growth grammar.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Recipe {
    /// Visible endpoint of the original recording, from zero through one.
    pub source_fraction: f64,
    /// Canonical longitudinal divisions over the completed object.
    pub longitudinal_segments: usize,
    /// Samples across the open arc, independent of growth time.
    pub transverse_segments: usize,
    /// Subdivisions around each rounded side lip.
    pub lip_segments: usize,
    /// Canonical source-feature intervals, fixed for all frames.
    pub feature_samples: usize,
    /// Gaussian feature smoothing standard deviation in source-time units.
    pub smoothing: f64,
    /// Fraction of the construction parameter occupied by the authored starting lip.
    pub seed_fraction: f64,
    /// Sweep of the growth axis, in complete turns.
    pub turns: f64,
    /// Added horizontal growth beyond the axis's 0.1-unit initial radius.
    pub axis_radius: f64,
    /// Additional vertical growth above the expanding rim.
    pub rise: f64,
    /// Radius of the broad final rim before source modulation.
    pub rim_radius: f64,
    /// Open arc span in degrees; the missing sector remains open throughout.
    pub aperture_degrees: f64,
    /// Final authored axis azimuth in degrees.
    pub end_azimuth_degrees: f64,
    /// Rotation of the open arc around its growth tangent, in degrees.
    pub profile_rotation_degrees: f64,
    /// Additional twist accumulated over the construction, in degrees.
    pub twist_degrees: f64,
    /// Bounded source size influence on rim expansion.
    pub size_response: f64,
    /// Close-encounter constriction strength.
    pub encounter_response: f64,
    /// Ordered distance-ratio influence on the three broad lobes.
    pub lobe_response: f64,
    /// Threefold rim folding, strengthened by non-collinear triangle shapes.
    pub triad_relief: f64,
    /// Bounded accumulated triangle turning of the growing profile, in radians.
    pub turn_response: f64,
    /// Unequal extension of the two free edges along the growth direction.
    pub lip_asymmetry: f64,
    /// Nominal material thickness in construction units.
    pub thickness: f64,
    /// Number of chronological ribs; zero is useful for silhouette studies.
    pub ribs: f64,
    /// Absolute relief of fine ribs in construction units.
    pub rib_relief: f64,
}

impl Default for Recipe {
    fn default() -> Self {
        Self {
            source_fraction: 1.0,
            longitudinal_segments: 400,
            transverse_segments: 128,
            lip_segments: 6,
            feature_samples: 1024,
            smoothing: 0.035,
            seed_fraction: 0.035,
            turns: 0.42,
            axis_radius: 1.8,
            rise: 0.35,
            rim_radius: 1.45,
            aperture_degrees: 215.0,
            end_azimuth_degrees: -60.0,
            profile_rotation_degrees: 180.0,
            twist_degrees: 32.0,
            size_response: 0.2,
            encounter_response: 0.22,
            lobe_response: 0.32,
            triad_relief: 0.18,
            turn_response: 0.55,
            lip_asymmetry: 0.28,
            thickness: 0.024,
            ribs: 0.0,
            rib_relief: 0.0,
        }
    }
}

impl Recipe {
    /// Reject nonfinite, degenerate or unbounded configurations before allocation.
    pub fn validate(&self) -> SilkResult<()> {
        for (name, value, low, high) in [
            ("source_fraction", self.source_fraction, 0.0, 1.0),
            ("smoothing", self.smoothing, 0.002, 0.15),
            ("seed_fraction", self.seed_fraction, 0.01, 0.15),
            ("turns", self.turns, 0.05, 1.5),
            ("axis_radius", self.axis_radius, 0.1, 4.0),
            ("rise", self.rise, 0.0, 3.0),
            ("rim_radius", self.rim_radius, 0.3, 2.5),
            ("aperture_degrees", self.aperture_degrees, 100.0, 300.0),
            ("end_azimuth_degrees", self.end_azimuth_degrees, -360.0, 360.0),
            ("profile_rotation_degrees", self.profile_rotation_degrees, -360.0, 360.0),
            ("twist_degrees", self.twist_degrees, -180.0, 180.0),
            ("size_response", self.size_response, 0.0, 0.4),
            ("encounter_response", self.encounter_response, 0.0, 0.4),
            ("lobe_response", self.lobe_response, 0.0, 0.7),
            ("triad_relief", self.triad_relief, 0.0, 0.7),
            ("turn_response", self.turn_response, 0.0, 1.5),
            ("lip_asymmetry", self.lip_asymmetry, -0.6, 0.6),
            ("thickness", self.thickness, 0.006, 0.08),
            ("ribs", self.ribs, 0.0, 240.0),
            ("rib_relief", self.rib_relief, 0.0, 0.015),
        ] {
            if !value.is_finite() || !(low..=high).contains(&value) {
                return Err(format!("{name} must be finite and within {low}..{high}").into());
            }
        }
        if !(64..=4096).contains(&self.longitudinal_segments)
            || !(32..=512).contains(&self.transverse_segments)
            || !(3..=16).contains(&self.lip_segments)
            || !(128..=8192).contains(&self.feature_samples)
        {
            return Err("shell sampling is outside the documented bounded ranges".into());
        }
        // Each smoothed shape component stays in [-1,1]. This conservative
        // bound keeps the radial profile positive even between source knots.
        if self.lobe_response * (0.65 * 2.0_f64.sqrt() + self.triad_relief) > 0.9 {
            return Err("combined rim lobing must preserve a positive radial profile".into());
        }
        if self.rib_relief > 0.0 && (self.longitudinal_segments as f64) < 10.0 * self.ribs {
            return Err(
                "resolved ribs require at least ten longitudinal samples per nominal rib".into()
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Signal {
    size: f64,
    close: f64,
    a: f64,
    b: f64,
    turn: f64,
    travel: f64,
}

impl Signal {
    fn components(self) -> [f64; 6] {
        [self.size, self.close, self.a, self.b, self.turn, self.travel]
    }

    fn from_components([size, close, a, b, turn, travel]: [f64; 6]) -> Self {
        Self { size, close, a, b, turn, travel }
    }

    fn interpolate(self, other: Self, left_slope: Self, right_slope: Self, t: f64) -> Self {
        if t <= 0.0 {
            return self;
        }
        if t >= 1.0 {
            return other;
        }
        let t2 = t * t;
        let t3 = t2 * t;
        let [h00, h10, h01, h11] =
            [2.0 * t3 - 3.0 * t2 + 1.0, t3 - 2.0 * t2 + t, -2.0 * t3 + 3.0 * t2, t3 - t2];
        let a = self.components();
        let b = other.components();
        let da = left_slope.components();
        let db = right_slope.components();
        Self::from_components(std::array::from_fn(|i| {
            h00 * a[i] + h10 * da[i] + h01 * b[i] + h11 * db[i]
        }))
    }
}

/// Uniform-knot PCHIP tangents, in value per knot interval. Harmonic interior
/// slopes and constrained one-sided endpoints preserve each component's local
/// monotonicity and prevent range overshoot between the frozen source samples.
fn signal_slopes(signals: &[Signal]) -> Vec<Signal> {
    let count = signals.len();
    let mut slopes = vec![[0.0; 6]; count];
    for (component, _) in signals[0].components().iter().enumerate() {
        let differences: Vec<f64> = signals
            .windows(2)
            .map(|pair| pair[1].components()[component] - pair[0].components()[component])
            .collect();
        slopes[0][component] = endpoint_slope(differences[0], differences[1]);
        slopes[count - 1][component] =
            endpoint_slope(differences[count - 2], differences[count - 3]);
        for i in 1..count - 1 {
            let a = differences[i - 1];
            let b = differences[i];
            if a != 0.0 && b != 0.0 && a.signum() == b.signum() {
                let small = a.abs().min(b.abs());
                let large = a.abs().max(b.abs());
                slopes[i][component] = a.signum() * (2.0 * small / (1.0 + small / large));
            }
        }
    }
    slopes.into_iter().map(Signal::from_components).collect()
}

fn endpoint_slope(first: f64, second: f64) -> f64 {
    let slope = 0.5 * (3.0 * first - second);
    if first == 0.0 || slope.signum() != first.signum() {
        0.0
    } else if first.signum() != second.signum() && slope.abs() > 3.0 * first.abs() {
        3.0 * first
    } else {
        slope
    }
}

/// Fixed C2 source clock: the authored seed has no source-time velocity or
/// acceleration, while the end joins the identity clock with unit derivative.
fn source_time(u: f64, seed: f64) -> f64 {
    let s = ((u - seed) / (1.0 - seed)).clamp(0.0, 1.0);
    (s * s * s * (6.0 + s * (-8.0 + 3.0 * s))).clamp(0.0, 1.0)
}

fn construction_end(time: f64, seed: f64) -> f64 {
    if time == 0.0 {
        return seed;
    }
    if time == 1.0 {
        return 1.0;
    }
    let (mut lower, mut upper) = (seed, 1.0);
    for _ in 0..64 {
        let middle = lower + (upper - lower) * 0.5;
        if source_time(middle, seed) < time {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    lower + (upper - lower) * 0.5
}

struct Surface<'a> {
    recipe: &'a Recipe,
    signals: Vec<Signal>,
    slopes: Vec<Signal>,
}

impl<'a> Surface<'a> {
    fn new(source: &OrbitSeries, recipe: &'a Recipe) -> SilkResult<Self> {
        let count = recipe.feature_samples;
        let mut edges = Vec::with_capacity(count + 1);
        let mut travel = Vec::with_capacity(count + 1);
        let mut perimeters = Vec::with_capacity(count + 1);
        let mut area_axis = V3::ZERO;
        let mut largest_area = 0.0;
        for i in 0..=count {
            let frame = source.sample(i as f64 / count as f64).ok_or("source frame unavailable")?;
            let p = frame.bodies.map(|body| body.position);
            let e = [p[1] - p[0], p[2] - p[1], p[0] - p[2]];
            let area = e[0].cross(-e[2]);
            if area.length_squared() > largest_area {
                largest_area = area.length_squared();
                area_axis = area.normalized();
            }
            perimeters.push(e.iter().map(|edge| edge.length()).sum::<f64>());
            travel.push(frame.bodies.iter().map(|body| body.arc_length).sum::<f64>());
            edges.push(e);
        }
        let mut sorted = perimeters.clone();
        sorted.sort_by(f64::total_cmp);
        let reference = sorted[count / 2].max(1e-8);
        let close_distance = reference / 12.0;
        let total_travel = (travel[count] - travel[0]).max(1e-12);
        let mut raw = Vec::with_capacity(count + 1);
        let mut spins = Vec::with_capacity(count + 1);
        for i in 0..=count {
            let squares = edges[i].map(V3::length_squared);
            let sum = squares.iter().sum::<f64>().max(1e-16);
            let before = i.saturating_sub(1);
            let after = (i + 1).min(count);
            let dt = (after - before) as f64 / count as f64;
            let omega = (0..3)
                .map(|j| edges[i][j].cross((edges[after][j] - edges[before][j]) / dt))
                .fold(V3::ZERO, |a, b| a + b)
                / sum;
            let spin = omega.dot(area_axis);
            spins.push(spin.abs());
            let clear = squares
                .iter()
                .map(|d2| 1.0 - 1.0 / (1.0 + (d2 / close_distance.powi(2)).powi(2)))
                .product::<f64>();
            raw.push(Signal {
                size: (perimeters[i] / reference).ln().tanh(),
                close: 1.0 - clear,
                a: (2.0 * squares[0] - squares[1] - squares[2]) / sum,
                b: 3.0_f64.sqrt() * (squares[1] - squares[2]) / sum,
                turn: spin,
                travel: (travel[i] - travel[0]) / total_travel,
            });
        }
        spins.sort_by(f64::total_cmp);
        let spin_scale = spins[count * 9 / 10].max(1e-8);
        for item in &mut raw {
            item.turn = (item.turn / spin_scale).tanh();
        }
        let sigma = recipe.smoothing * count as f64;
        let radius = (3.0 * sigma).ceil() as usize;
        let mut signals = Vec::with_capacity(count + 1);
        for i in 0..=count {
            let mut item = Signal::default();
            let mut total = 0.0;
            for (j, sample) in raw
                .iter()
                .enumerate()
                .take((i + radius).min(count) + 1)
                .skip(i.saturating_sub(radius))
            {
                let weight = (-0.5 * ((j as f64 - i as f64) / sigma).powi(2)).exp();
                item.size += sample.size * weight;
                item.close += sample.close * weight;
                item.a += sample.a * weight;
                item.b += sample.b * weight;
                item.turn += sample.turn * weight;
                total += weight;
            }
            item.size /= total;
            item.close /= total;
            item.a /= total;
            item.b /= total;
            item.turn /= total;
            item.travel = raw[i].travel;
            signals.push(item);
        }
        let mut integral = 0.0;
        let mut previous = signals[0].turn;
        signals[0].turn = 0.0;
        for item in &mut signals[1..] {
            let current = item.turn;
            integral += (previous + current) * 0.5 / count as f64;
            item.turn = integral;
            previous = current;
        }
        let slopes = signal_slopes(&signals);
        let surface = Self { recipe, signals, slopes };
        if recipe.ribs > 0.0 && recipe.rib_relief > 0.0 {
            let maximum = surface.maximum_rib_phase_advance();
            if maximum > 0.1 + 1e-12 {
                return Err(format!(
                    "rib phase advances {maximum:.6} cycles in a canonical geometry interval; require at most 0.1 by increasing longitudinal_segments or reducing ribs"
                ).into());
            }
        }
        Ok(surface)
    }

    fn signal(&self, u: f64) -> Signal {
        let t = source_time(u, self.recipe.seed_fraction);
        let x = t * (self.signals.len() - 1) as f64;
        let i = (x.floor() as usize).min(self.signals.len() - 2);
        self.signals[i].interpolate(
            self.signals[i + 1],
            self.slopes[i],
            self.slopes[i + 1],
            x - i as f64,
        )
    }

    fn rib_phase_cycles(&self, u: f64) -> f64 {
        self.recipe.ribs * (0.45 * u + 0.55 * self.signal(u).travel)
    }

    fn rib_amplitude(&self, u: f64) -> f64 {
        let t = source_time(u, self.recipe.seed_fraction);
        // Identically zero throughout the authored seed, with zero first and
        // second derivatives at its end under the fixed C2 source clock.
        self.recipe.rib_relief * -(-8.0 * t).exp_m1()
    }

    fn rib_height(&self, u: f64) -> f64 {
        let phase = TAU * self.rib_phase_cycles(u);
        self.rib_amplitude(u) * (0.5 + 0.5 * phase.cos()).powi(3)
    }

    fn maximum_rib_phase_advance(&self) -> f64 {
        let mut previous = self.rib_phase_cycles(0.0);
        let mut maximum = 0.0_f64;
        for index in 1..=self.recipe.longitudinal_segments {
            let current =
                self.rib_phase_cycles(index as f64 / self.recipe.longitudinal_segments as f64);
            maximum = maximum.max(current - previous);
            previous = current;
        }
        maximum
    }

    fn width(&self, u: f64) -> f64 {
        let growth = 0.25 * u + 0.75 * u * u;
        0.045 + (self.recipe.rim_radius - 0.045) * growth
    }

    fn center(&self, u: f64) -> V3 {
        let c = self.recipe;
        let angle = c.end_azimuth_degrees.to_radians() + TAU * c.turns * (u - 1.0);
        let radius = 0.1 + c.axis_radius * (0.65 * u + 0.35 * u * u);
        V3::new(radius * angle.cos(), radius * angle.sin(), 1.55 * self.width(u) + c.rise * u)
    }

    fn point(&self, u: f64, v: f64) -> V3 {
        let c = self.recipe;
        let s = self.signal(u);
        let h = 1e-5;
        let ua = (u - h).max(0.0);
        let ub = (u + h).min(1.0);
        let tangent = (self.center(ub) - self.center(ua)).normalized();
        let center = self.center(u);
        let radial = V3::new(center.x, center.y, 0.0).normalized();
        let n = (radial - tangent * radial.dot(tangent)).normalized();
        let b = n.cross(tangent).normalized();
        let phi = c.profile_rotation_degrees.to_radians()
            + c.twist_degrees.to_radians() * u
            + c.turn_response * s.turn
            + c.aperture_degrees.to_radians() * 0.5 * v;
        let triangle_fullness = (1.0 - s.a * s.a - s.b * s.b).clamp(0.0, 1.0);
        let triad = c.triad_relief * (0.5 + 0.5 * triangle_fullness);
        let shape = 1.0
            + c.lobe_response
                * (0.65 * s.a * (2.0 * phi).cos()
                    + 0.65 * s.b * (2.0 * phi).sin()
                    + triad * (3.0 * phi + 0.5).cos());
        let width =
            self.width(u) * (c.size_response * s.size - c.encounter_response * s.close).exp();
        let radius = width * shape;
        center
            + (n * phi.cos() + b * phi.sin()) * radius
            + tangent * (c.lip_asymmetry * width * (v.powi(3) + 0.3 * v) * u)
    }

    fn frame(&self, u: f64, v: f64) -> (V3, V3, V3) {
        let h = 1e-5;
        let du = self.point((u + h).min(1.0), v) - self.point((u - h).max(0.0), v);
        let dv = self.point(u, (v + h).min(1.0)) - self.point(u, (v - h).max(-1.0));
        (self.point(u, v), du.cross(dv).normalized(), dv.normalized())
    }

    fn half_thickness(&self, u: f64) -> f64 {
        self.recipe.thickness * 0.5 * (0.55 + 0.45 * u * u * (3.0 - 2.0 * u))
    }

    fn wall(&self, u: f64, q: f64) -> V3 {
        let q = q.rem_euclid(4.0);
        let h = self.half_thickness(u);
        if q <= 1.0 {
            let v = 2.0 * q - 1.0;
            let (p, n, _) = self.frame(u, v);
            let relief = self.rib_height(u) * (1.0 - v * v).powi(2);
            // Relief is deposited along the smooth base normal. The base
            // frame never follows the high-curvature rib corrugation itself.
            p + n * (h + relief)
        } else if q < 2.0 {
            let (p, n, t) = self.frame(u, 1.0);
            let a = PI * (q - 1.0);
            p + (n * a.cos() + t * a.sin()) * h
        } else if q <= 3.0 {
            let v = 5.0 - 2.0 * q;
            let (p, n, _) = self.frame(u, v);
            let relief = self.rib_height(u) * (1.0 - v * v).powi(2);
            p - n * (h + 0.5 * relief)
        } else {
            let (p, n, t) = self.frame(u, -1.0);
            let a = PI * (q - 3.0);
            p + (-n * a.cos() - t * a.sin()) * h
        }
    }

    fn normal(&self, u: f64, q: f64) -> V3 {
        let h = 1e-5;
        let du = self.wall((u + h).min(1.0), q) - self.wall((u - h).max(0.0), q);
        let dq = self.wall(u, q + h) - self.wall(u, q - h);
        du.cross(dq).normalized()
    }
}

/// Measurements of the actual exported growth prefix.
#[derive(Clone, Debug, Serialize)]
pub struct Diagnostics {
    /// Number of retained canonical rows including the moving front.
    pub rows: usize,
    /// Number of samples around each closed porcelain-wall section.
    pub ring_vertices: usize,
    /// Construction coordinate of the moving front.
    pub construction_end: f64,
    /// Explicit initial authored seed, rather than invented physical prehistory.
    pub seed_fraction: f64,
    /// Whether a general self-intersection search has been performed.
    pub self_intersections_checked: bool,
    /// Frozen feature values at the end of the original recording.
    pub final_triangle_shape: [f64; 2],
    /// Frozen integrated rotation proxy used by the growth grammar.
    pub accumulated_turn: f64,
}

/// Closed material boundary with independently evaluated smooth shading normals.
pub struct BuiltShell {
    /// The generated solid boundary, in authored construction coordinates.
    pub mesh: Mesh,
    /// One finite unit shading normal for every vertex.
    pub normals: Vec<V3>,
    /// Explicit source-mapping and sampling measurements.
    pub diagnostics: Diagnostics,
}

/// Construct a closed shell wall while preserving the completed surface prefix.
pub fn build(source: &OrbitSeries, recipe: &Recipe) -> SilkResult<BuiltShell> {
    recipe.validate()?;
    let surface = Surface::new(source, recipe)?;
    let end = construction_end(recipe.source_fraction, recipe.seed_fraction);
    let mut rows: Vec<f64> = (0..=recipe.longitudinal_segments)
        .map(|i| i as f64 / recipe.longitudinal_segments as f64)
        .take_while(|u| *u <= end)
        .collect();
    if rows.last().is_none_or(|u| end - *u > 1e-12) {
        rows.push(end);
    }
    let nv = recipe.transverse_segments;
    let nl = recipe.lip_segments;
    let mut qs = Vec::new();
    qs.extend((0..=nv).map(|j| j as f64 / nv as f64));
    qs.extend((1..nl).map(|j| 1.0 + j as f64 / nl as f64));
    qs.extend((0..=nv).map(|j| 2.0 + j as f64 / nv as f64));
    qs.extend((1..nl).map(|j| 3.0 + j as f64 / nl as f64));
    let nr = qs.len();
    // Every sample is independent; indexed collection preserves canonical order
    // and gives identical bits regardless of worker count.
    let samples: Vec<(V3, V3)> = (0..rows.len() * nr)
        .into_par_iter()
        .map(|index| {
            let u = rows[index / nr];
            let q = qs[index % nr];
            (surface.wall(u, q), surface.normal(u, q))
        })
        .collect();
    let (vertices, mut normals): (Vec<V3>, Vec<V3>) = samples.into_iter().unzip();
    let mut mesh = Mesh { vertices, triangles: Vec::new() };
    for i in 0..rows.len() - 1 {
        for j in 0..nr {
            let a = (i * nr + j) as u32;
            let b = ((i + 1) * nr + j) as u32;
            let c = ((i + 1) * nr + (j + 1) % nr) as u32;
            let d = (i * nr + (j + 1) % nr) as u32;
            mesh.triangles.extend([[a, b, c], [a, c, d]]);
        }
    }
    for (row, flip) in [(0, false), (rows.len() - 1, true)] {
        let u = rows[row];
        let base = row * nr;
        let middle = mesh.vertices.len();
        for j in 0..=nv {
            let v = 2.0 * j as f64 / nv as f64 - 1.0;
            let (p, n, t) = surface.frame(u, v);
            mesh.vertices.push(p);
            normals.push(t.cross(n).normalized() * if flip { 1.0 } else { -1.0 });
        }
        let mut emit = |tri: [usize; 3]| {
            let mut t = tri.map(|v| v as u32);
            if flip {
                t.swap(1, 2);
            }
            mesh.triangles.push(t);
        };
        for j in 0..nv {
            let a = base + j;
            let b = base + j + 1;
            let m = middle + j;
            let n = middle + j + 1;
            let c = base + 2 * nv + nl - j;
            let d = c - 1;
            emit([a, b, n]);
            emit([a, n, m]);
            emit([m, n, d]);
            emit([m, d, c]);
        }
        for j in 0..nl {
            emit([middle + nv, base + nv + j, base + nv + j + 1]);
        }
        for j in 0..nl {
            emit([middle, base + 2 * nv + nl + j, base + (2 * nv + nl + j + 1) % nr]);
        }
    }
    if mesh.vertices.iter().any(|p| !p.is_finite())
        || normals.iter().any(|n| !n.is_finite() || (n.length() - 1.0).abs() > 1e-8)
    {
        return Err("growth surface contains a degenerate position or derivative".into());
    }
    mesh.validate()?;
    let s = surface.signals.last().ok_or("empty source features")?;
    Ok(BuiltShell {
        mesh,
        normals,
        diagnostics: Diagnostics {
            rows: rows.len(),
            ring_vertices: nr,
            construction_end: end,
            seed_fraction: recipe.seed_fraction,
            self_intersections_checked: false,
            final_triangle_shape: [s.a, s.b],
            accumulated_turn: s.turn,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silk::OrbitData;

    fn source() -> OrbitSeries {
        let samples = (0..129)
            .map(|i| {
                let a = TAU * f64::from(i) / 128.0;
                std::array::from_fn(|j| {
                    let p = a + TAU * j as f64 / 3.0;
                    V3::new(p.cos(), p.sin(), 0.15 * (2.0 * p).sin())
                })
            })
            .collect();
        OrbitSeries::new(&OrbitData {
            seed: "shell-test".into(),
            dt: 0.01,
            masses: [1.0; 3],
            samples,
            provenance: serde_json::json!({}),
        })
        .unwrap()
    }

    #[test]
    fn completed_and_seed_walls_are_closed_oriented_solids() {
        let source = source();
        let mut r = Recipe {
            longitudinal_segments: 64,
            transverse_segments: 32,
            feature_samples: 128,
            ..Recipe::default()
        };
        for time in [0.0, 0.37, 1.0] {
            r.source_fraction = time;
            let b = build(&source, &r).unwrap();
            let a = b.mesh.validate().unwrap();
            assert_eq!(a.components, 1);
            assert_eq!(a.euler_characteristic, 2);
            assert!(a.signed_volume > 0.0);
        }
    }

    #[test]
    fn deposited_rings_do_not_move_when_growth_advances() {
        let source = source();
        let mut r = Recipe {
            longitudinal_segments: 64,
            transverse_segments: 32,
            feature_samples: 128,
            source_fraction: 0.31,
            ..Recipe::default()
        };
        let a = build(&source, &r).unwrap();
        r.source_fraction = 0.8;
        let b = build(&source, &r).unwrap();
        let count = (a.diagnostics.rows - 1) * a.diagnostics.ring_vertices;
        assert_eq!(&a.mesh.vertices[..count], &b.mesh.vertices[..count]);
        assert_eq!(&a.normals[..count], &b.normals[..count]);
        assert!(
            (source_time(a.diagnostics.construction_end, r.seed_fraction) - 0.31).abs() < 1e-14
        );
        assert!((source_time(b.diagnostics.construction_end, r.seed_fraction) - 0.8).abs() < 1e-14);
    }

    #[test]
    fn worker_count_does_not_change_geometry_or_normals() {
        let source = source();
        let recipe = Recipe {
            longitudinal_segments: 64,
            transverse_segments: 32,
            feature_samples: 128,
            ..Recipe::default()
        };
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| build(&source, &recipe).unwrap())
        };
        let a = run(1);
        let b = run(4);
        assert_eq!(a.mesh.vertices, b.mesh.vertices);
        assert_eq!(a.mesh.triangles, b.mesh.triangles);
        assert_eq!(a.normals, b.normals);
    }

    #[test]
    fn triangle_rotation_survives_constant_pair_distances() {
        let s = source();
        let r = Recipe { feature_samples: 128, ..Recipe::default() };
        let f = Surface::new(&s, &r).unwrap();
        assert!(f.signals.last().unwrap().turn.abs() > 0.5);
    }

    #[test]
    fn unsupported_sampling_and_unknown_controls_are_errors() {
        let mut r = Recipe { aperture_degrees: 360.0, ..Recipe::default() };
        assert!(r.validate().is_err());
        r = Recipe::default();
        r.ribs = 120.0;
        r.rib_relief = 0.005;
        assert!(r.validate().is_err());
        assert!(serde_json::from_str::<Recipe>(r#"{"secret_warp":true}"#).is_err());
        r = Recipe { lobe_response: 0.7, triad_relief: 0.7, ..Recipe::default() };
        assert!(r.validate().is_err());
    }

    #[test]
    fn fixed_source_clock_is_monotone_invertible_and_flat_at_the_seed_join() {
        let seed = 0.035;
        assert_eq!(construction_end(0.0, seed), seed);
        assert_eq!(construction_end(1.0, seed), 1.0);
        let mut previous = seed;
        for time in [0.0, 1e-8, 0.001, 0.03, 0.2, 0.37, 0.8, 0.999, 1.0] {
            let u = construction_end(time, seed);
            assert!(u >= previous);
            assert!((source_time(u, seed) - time).abs() < 2e-14);
            previous = u;
        }
        let h = 1e-5;
        assert_eq!(source_time(seed - h, seed), 0.0);
        assert!(source_time(seed + h, seed) / (h * h) < 1e-3);
    }

    #[test]
    fn pchip_preserves_ranges_plateaus_and_monotone_travel() {
        let travel = [0.0, 0.12, 0.12, 0.75, 1.0];
        let sizes = [0.0, 0.4, -0.1, -0.1, 0.2];
        let signals: Vec<_> = (0..travel.len())
            .map(|i| Signal { travel: travel[i], size: sizes[i], ..Signal::default() })
            .collect();
        let slopes = signal_slopes(&signals);
        let mut previous = 0.0;
        for interval in 0..signals.len() - 1 {
            for sample in 0..=100 {
                let value = signals[interval].interpolate(
                    signals[interval + 1],
                    slopes[interval],
                    slopes[interval + 1],
                    f64::from(sample) / 100.0,
                );
                assert!(value.travel + 1e-14 >= previous);
                assert!(
                    value.travel >= travel[interval] - 1e-14
                        && value.travel <= travel[interval + 1] + 1e-14
                );
                assert!(value.size >= sizes[interval].min(sizes[interval + 1]) - 1e-14);
                assert!(value.size <= sizes[interval].max(sizes[interval + 1]) + 1e-14);
                previous = value.travel;
            }
        }
        assert_eq!(slopes[1].travel, 0.0);
        assert_eq!(slopes[2].travel, 0.0);
    }

    #[test]
    fn seed_join_has_no_source_feature_or_rib_amplitude_velocity_jump() {
        let source = source();
        let recipe = Recipe {
            feature_samples: 128,
            longitudinal_segments: 512,
            ribs: 8.0,
            rib_relief: 0.002,
            ..Recipe::default()
        };
        let surface = Surface::new(&source, &recipe).unwrap();
        let u = recipe.seed_fraction;
        let h = 1e-4;
        let before = surface.signal(u - h).components();
        let at = surface.signal(u).components();
        let after = surface.signal(u + h).components();
        assert_eq!(before, at);
        for (a, b) in at.into_iter().zip(after) {
            assert!((b - a).abs() / h < 1e-5);
            assert!((b - a).abs() / (h * h) < 0.01);
        }
        for position in [0.0, u * 0.5, u] {
            assert_eq!(surface.rib_amplitude(position), 0.0);
        }
        assert!(surface.rib_amplitude(u + h) / (h * h) < 0.001);
        assert!(surface.rib_amplitude(0.7) > 0.0);
    }

    #[test]
    fn source_turn_rotates_the_profile_without_changing_the_authored_axis() {
        let source = source();
        let quiet = Recipe { feature_samples: 128, turn_response: 0.0, ..Recipe::default() };
        let turning = Recipe { turn_response: 1.2, ..quiet.clone() };
        let a = Surface::new(&source, &quiet).unwrap();
        let b = Surface::new(&source, &turning).unwrap();
        assert_eq!(a.center(0.63), b.center(0.63));
        assert!((a.point(0.63, 0.4) - b.point(0.63, 0.4)).length() > 0.01);
    }

    #[test]
    fn local_rib_phase_is_monotone_and_nominal_sampling_alone_is_rejected() {
        let source = source();
        let recipe = Recipe {
            longitudinal_segments: 1024,
            feature_samples: 128,
            ribs: 8.0,
            rib_relief: 0.002,
            ..Recipe::default()
        };
        let surface = Surface::new(&source, &recipe).unwrap();
        let mut previous = surface.rib_phase_cycles(0.0);
        for i in 1..=4096 {
            let current = surface.rib_phase_cycles(f64::from(i) / 4096.0);
            assert!(current >= previous);
            previous = current;
        }
        assert!(surface.maximum_rib_phase_advance() <= 0.1);
        let underresolved = Recipe { longitudinal_segments: 640, ribs: 64.0, ..recipe };
        underresolved.validate().unwrap();
        match Surface::new(&source, &underresolved) {
            Ok(_) => panic!("nominal ten samples per rib must not bypass the local phase gate"),
            Err(error) => assert!(error.to_string().contains("rib phase advances")),
        }
    }

    #[test]
    fn radial_projection_frame_cannot_use_zero_turn_or_negative_rise() {
        let mut recipe = Recipe { turns: 0.0, ..Recipe::default() };
        assert!(recipe.validate().is_err());
        recipe = Recipe::default();
        recipe.rise = -0.35;
        assert!(recipe.validate().is_err());
    }

    #[test]
    fn authored_seed_growth_has_finite_bounded_endpoint_derivatives() {
        let source = source();
        let recipe = Recipe { feature_samples: 128, ..Recipe::default() };
        let surface = Surface::new(&source, &recipe).unwrap();
        for h in [1e-3, 1e-4, 1e-5] {
            let c0 = surface.center(0.0);
            let c1 = surface.center(h);
            let c2 = surface.center(2.0 * h);
            let first = (-c0 * 3.0 + c1 * 4.0 - c2) / (2.0 * h);
            let second = (c2 - c1 * 2.0 + c0) / (h * h);
            assert!(first.is_finite() && first.length() < 10.0);
            assert!(second.is_finite() && second.length() < 100.0);
            let width_second =
                (surface.width(2.0 * h) - 2.0 * surface.width(h) + surface.width(0.0)) / (h * h);
            assert!((width_second - 1.5 * (recipe.rim_radius - 0.045)).abs() < 1e-5);
            let thickness_first = (-3.0 * surface.half_thickness(0.0)
                + 4.0 * surface.half_thickness(h)
                - surface.half_thickness(2.0 * h))
                / (2.0 * h);
            let thickness_second = (surface.half_thickness(2.0 * h)
                - 2.0 * surface.half_thickness(h)
                + surface.half_thickness(0.0))
                / (h * h);
            assert!(thickness_first.abs() < 1e-6);
            assert!(thickness_second.is_finite() && thickness_second.abs() < 0.1);
        }
        assert_eq!(surface.width(0.0), 0.045);
        assert!((surface.width(1.0) - recipe.rim_radius).abs() < 1e-14);
        assert_eq!(surface.half_thickness(1.0), recipe.thickness * 0.5);
    }

    #[test]
    fn ribs_preserve_the_base_surface_and_rounded_lips_and_only_increase_wall_gap() {
        let source = source();
        let quiet = Recipe {
            feature_samples: 128,
            longitudinal_segments: 2048,
            ribs: 64.0,
            rib_relief: 0.0,
            ..Recipe::default()
        };
        let ribbed = Recipe { rib_relief: 0.006, ..quiet.clone() };
        let base = Surface::new(&source, &quiet).unwrap();
        let relief = Surface::new(&source, &ribbed).unwrap();
        let mut increased_gap = false;
        for u in [0.0, quiet.seed_fraction, 0.19, 0.47, 0.83, 1.0] {
            for v in [-1.0, -0.65, 0.0, 0.4, 1.0] {
                assert_eq!(base.point(u, v), relief.point(u, v));
                assert_eq!(base.frame(u, v), relief.frame(u, v));
                let (p, normal, _) = relief.frame(u, v);
                let outer = relief.wall(u, (v + 1.0) * 0.5);
                let inner = relief.wall(u, (5.0 - v) * 0.5);
                let window = (1.0 - v * v).powi(2);
                let expected = 2.0 * relief.half_thickness(u) + 1.5 * relief.rib_height(u) * window;
                let gap = (outer - inner).dot(normal);
                assert!(gap > 0.0);
                assert!((gap - expected).abs() < 2e-13);
                assert!((outer - p).dot(normal) > 0.0);
                assert!((inner - p).dot(normal) < 0.0);
                increased_gap |= gap > 2.0 * base.half_thickness(u) + 1e-5;
            }
            for q in [0.0, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 3.25, 3.5, 3.75] {
                assert_eq!(base.wall(u, q), relief.wall(u, q));
            }
        }
        assert!(increased_gap);
    }
}
