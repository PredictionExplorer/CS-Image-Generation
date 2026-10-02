//! From the canvas track to three tidally stretched bodies moving through the fluid.
//!
//! The ember edition stirs the fluid with the three bodies of the selected orbit, seen exactly as
//! the main edition shows them: `view` turns the raw integrator positions into the canvas track
//! `pos[body][knot]` (`N` knots per body, knot `N-1` is the final step) on the canvas
//! `[-aspect, aspect] × [-1, 1]`. This module maps the knots to fluid time and gives each body
//! the elliptical outline that the tidal field of the other two stretches it into ([`Tidal`],
//! [`Shape`]). The time map ports the museum-lab `wake/ns.py` (`Bodies`,
//! `median_fraction_speed`); docs/ember-design.md §3 is the binding recipe.
//!
//! # Time map
//!
//! Between knots the bodies move on straight segments, parametrised by the source fraction
//! `f ∈ [0, 1]`: `fi = f·(N-1)`, `left = min(⌊fi⌋, N-2)`, `w = fi - left`,
//! `pos(f) = pos_left + w·(pos_{left+1} - pos_left)`, with the outgoing segment's velocity
//! `dpos/df = (pos_{left+1} - pos_left)·(N-1)`. Fluid time is `t = f·T`, and the duration
//!
//! ```text
//! T = median_{j, b} |dpos_b/df (j / 200000)| / reference_speed,     j = 0..=200000,
//! ```
//!
//! makes the median body speed exactly the fluid's reference speed (hence the Reynolds number).
//! Knot `k` sits at fluid time `t_k = T·k/(N-1)`.
//!
//! # Shapes
//!
//! Each body is an ellipse of the disc's area, stretched along the principal axis of the tidal
//! field of the other two (docs/ember-design.md §3.10): a disc where the field is isotropic, up
//! to `max_aspect : 1` at the orbit's closest moments. Its axes turn and stretch with the field;
//! the rates come from forward differences over `10⁻⁴` fluid units, and its material follows the
//! irrotational flow that moves the outline ([`Shape::deformation_velocity`]).
//!
//! # Determinism
//!
//! Everything is sequential, fixed-order `f64` arithmetic using only `+ - × ÷ √` and comparisons;
//! the median and the tidal reference are order statistics (exact for any selection algorithm).
//! Given the same input bits, every CPU produces the same duration, positions and shapes.

use std::fmt;

use super::config::EmberConfig;
use super::error::{EmberError, EmberResult};
use super::math::{max, min};

/// Number of bodies of a three-body orbit.
const BODIES: usize = 3;

/// Uniform source-fraction intervals of the median-speed estimate: `200_001` samples per body.
const MEDIAN_INTERVALS: usize = 200_000;

/// Uniform fluid-time intervals of the speed look-ahead table: `400_001` entries on `[0, T]`
/// (the prototype's `linspace(0, T, 400001)`).
const SPEED_INTERVALS: usize = 400_000;

/// Table entries before the look-ahead start that the look-ahead speed still covers.
const LOOK_BEHIND: usize = 2;

/// Table entries, starting at the look-ahead start, that the look-ahead speed covers (a horizon
/// of `T/1000` fluid time).
const LOOK_AHEAD: usize = 400;

/// Entries per block of the block-maximum index over the speed table.
const SPEED_BLOCK: usize = 64;

/// Uniform fluid-time intervals on `[0, T]` at which every body's tidal anisotropy is sampled
/// for its reference quantile (`3 × 4001` samples).
const TIDAL_INTERVALS: usize = 4000;

/// Fluid-time step of the forward differences that give a shape's turning and stretch rates.
const SHAPE_RATE_STEP: f64 = 1e-4;

/// Position, velocity and shape of one body in world coordinates at one fluid time.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct BodyState {
    /// World position `(x, y)` of the centre.
    pub position: [f64; 2],
    /// World velocity `(vx, vy)` of the centre, in world units per fluid time unit.
    pub velocity: [f64; 2],
    /// The body's elliptical outline and how it deforms (docs/ember-design.md §3.10).
    pub shape: Shape,
}

/// A body's outline: an ellipse of constant area that the tidal field of the other two bodies
/// stretches (a disc when it is unstretched), and the rates at which it turns and stretches.
///
/// The body's material does not rotate rigidly with its axes. It moves with the irrotational,
/// area-preserving flow that carries the elliptical boundary ([`Shape::deformation_velocity`]),
/// as a fluid star's tidal bulge does: the bulge travels round the star, its matter does not.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Shape {
    /// Semi-axes `(a, b)`, `a ≥ b > 0`, along the body's own axes.
    pub semi: [f64; 2],
    /// Unit vector `(cos θ, sin θ)` of the first (long) axis.
    pub axis: [f64; 2],
    /// Rate `dθ/dt` at which the axes turn (counter-clockwise positive).
    pub spin: f64,
    /// Stretch rate `d ln a/dt` of the long semi-axis (area kept: `d ln b/dt = -d ln a/dt`).
    pub strain: f64,
}

impl Shape {
    /// A rigid disc of radius `radius`.
    #[cfg(test)]
    pub(crate) fn disc(radius: f64) -> Self {
        Self { semi: [radius, radius], axis: [1.0, 0.0], spin: 0.0, strain: 0.0 }
    }

    /// Body-frame coordinates of a world offset `d` from the centre.
    #[inline(always)]
    pub(crate) fn to_body(self, d: [f64; 2]) -> [f64; 2] {
        let [c, s] = self.axis;
        [c * d[0] + s * d[1], -s * d[0] + c * d[1]]
    }

    /// Signed distance of a world offset `d` from the outline, negative inside: Taubin's
    /// first-order approximation `F/|∇F|` of `F = sqrt((x/a)² + (y/b)²) - 1`. It is exact for a
    /// disc and accurate to second order near the outline, which is where the mask's tanh edge
    /// is evaluated.
    #[inline(always)]
    pub(crate) fn signed_distance(&self, d: [f64; 2]) -> f64 {
        let [x, y] = self.to_body(d);
        let [a, b] = self.semi;
        let (xa, yb) = (x / a, y / b);
        let q = (xa * xa + yb * yb).sqrt();
        if q < CENTRE_EPSILON {
            // At the centre the gradient vanishes; the depth there is the short semi-axis.
            return -min(a, b);
        }
        let (gx, gy) = (x / (a * a), y / (b * b));
        (q - 1.0) * q / (gx * gx + gy * gy).sqrt()
    }

    /// World velocity, relative to the centre's, of the body's material at world offset `d`: the
    /// irrotational, divergence-free flow `φ = k·x'y' + ½·e·(x'² - y'²)` in body coordinates, with
    /// `k = Ω·(a² - b²)/(a² + b²)` for axes turning at `Ω` and `e = d ln a/dt`. It moves the
    /// elliptical outline exactly (the kinematic boundary condition) and keeps its area.
    #[inline(always)]
    pub(crate) fn deformation_velocity(&self, d: [f64; 2]) -> [f64; 2] {
        let [a, b] = self.semi;
        let k = self.spin * (a * a - b * b) / (a * a + b * b);
        let e = self.strain;
        let [x, y] = self.to_body(d);
        let (ux, uy) = (k * y + e * x, k * x - e * y);
        let [c, s] = self.axis;
        [c * ux - s * uy, s * ux + c * uy]
    }

    /// Whether the world offset `d` from the centre lies strictly inside the outline:
    /// `(x'/a)² + (y'/b)² < 1` in body coordinates (never, for an outline of zero size).
    #[inline(always)]
    pub(crate) fn contains(&self, d: [f64; 2]) -> bool {
        let [a, b] = self.semi;
        if !(a > 0.0 && b > 0.0) {
            return false;
        }
        let [x, y] = self.to_body(d);
        let (xa, yb) = (x / a, y / b);
        xa * xa + yb * yb < 1.0
    }

    /// Largest speed of [`Shape::deformation_velocity`] on or inside the outline:
    /// `(|k| + |e|)·a`.
    pub(crate) fn deformation_speed(&self) -> f64 {
        let [a, b] = self.semi;
        let k = self.spin * (a * a - b * b) / (a * a + b * b);
        (k.abs() + self.strain.abs()) * a
    }

    /// Long semi-axis.
    pub(crate) fn extent(&self) -> f64 {
        max(self.semi[0], self.semi[1])
    }
}

/// Below this normalised radius the signed distance takes its centre value (avoids 0/0).
const CENTRE_EPSILON: f64 = 1e-12;

/// Planning figures of a track ([`BodyTrack::survey`]): what an orbit will cost and how it sits
/// on the canvas, known before any fluid is simulated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TrackSurvey {
    /// Fluid steps the solver's step rule takes on the bodies' own speeds (a lower bound of a
    /// render's: the stirred water is faster in places).
    pub fluid_steps: f64,
    /// Fastest material speed of a body, in units of the reference speed.
    pub peak_speed: f64,
    /// Smallest distance of a body's centre from the canvas edge, over the knots.
    pub edge_clearance: f64,
    /// Fraction of the orbit (sampled at uniform times) during which two bodies' centres are
    /// closer than two body radii.
    pub overlap_fraction: f64,
}

/// Motion of the three bodies through the fluid.
pub(crate) trait BodyMotion: Sync {
    /// The three bodies at fluid time `t`.
    fn bodies_at(&self, t: f64) -> [BodyState; 3];

    /// The bodies' look-ahead speed at `t` for the solver's CFL condition: the fastest speed of
    /// any body's material (centre speed plus deformation speed) the motion expects over a short
    /// window starting at (or just before) `t`.
    ///
    /// For a [`BodyTrack`] this is the prototype's rule: the maximum of the speed table (the
    /// fastest body's speed sampled at `400_001` uniform times) over the table entries of the
    /// look-ahead window. It is not a strict bound on the continuous motion: the window is
    /// `T/1000` long (shorter than the longest solver step `max_dt` when `T < 1000·max_dt`,
    /// i.e. `T < 2` with the defaults), and the table samples every segment only when the orbit
    /// has at most `400_000` segments. The solver also caps its step by the flow speed, which
    /// includes the penalised flow the bodies drag along.
    fn speed_bound(&self, t: f64) -> f64;
}

/// The orbit as three moving bodies: the canvas knots, the orbit-to-fluid time map and the tidal
/// model that shapes the bodies.
///
/// Invariants (established by [`BodyTrack::new`]): exactly three bodies with `N ≥ 2` finite
/// knots each, `0 < T < ∞`, and a finite speed table of `400_001` entries. The table
/// samples the fastest body's speed at uniform times; [`BodyMotion::speed_bound`] is its maximum
/// over a look-ahead window (the prototype's CFL rule, not a strict bound on the motion).
pub(crate) struct BodyTrack {
    /// Fluid time `T` of the last knot.
    duration: f64,
    /// The tidal model that shapes the bodies.
    tidal: Tidal,
    /// Knots per body, `N ≥ 2`.
    knots: usize,
    /// `N - 1` as `f64` (exact for any realistic `N`).
    last: f64,
    /// Canvas `x` of every knot, `xs[body][knot]` (structure of arrays).
    xs: [Vec<f64>; BODIES],
    /// Canvas `y` of every knot, `ys[body][knot]`.
    ys: [Vec<f64>; BODIES],
    /// `speed[i]` = largest body speed at `t_i = T·i/400000`, `i ∈ [0, 400000]` (samples: with
    /// more than `400_000` segments, some segments are never sampled).
    speed: Vec<f64>,
    /// `speed_blocks[k]` = max of `speed[64k .. 64k + 64]` (clipped to the table).
    speed_blocks: Vec<f64>,
}

impl fmt::Debug for BodyTrack {
    /// A summary; the knot arrays are far too large to print.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BodyTrack")
            .field("duration", &self.duration)
            .field("knots", &self.knots)
            .finish_non_exhaustive()
    }
}

impl BodyTrack {
    /// The track of three bodies whose canvas positions at the recorded knots are
    /// `track[body][knot]` (`View::canvas_track`), with the fluid duration derived from the median
    /// body speed.
    ///
    /// Errors: [`EmberError::DegenerateOrbit`] for unequal or too short recordings, non-finite
    /// coordinates or a median body speed of zero; [`EmberError::InvalidConfig`] for a
    /// non-positive reference speed or mass.
    pub(crate) fn new(
        track: [Vec<[f64; 2]>; BODIES],
        masses: [f64; 3],
        config: &EmberConfig,
    ) -> EmberResult<Self> {
        let reference_speed = config.fluid.reference_speed;
        require_positive(reference_speed, "fluid.reference_speed")?;
        let knots = track[0].len();
        if track.iter().any(|body| body.len() != knots) {
            return degenerate(format!(
                "bodies have different recording lengths ({}, {}, {})",
                track[0].len(),
                track[1].len(),
                track[2].len()
            ));
        }
        if knots < 2 {
            return degenerate(format!("{knots} recorded knots; at least 2 are needed"));
        }
        for (body, points) in track.iter().enumerate() {
            if let Some(knot) = points.iter().position(|p| !(p[0].is_finite() && p[1].is_finite()))
            {
                return degenerate(format!(
                    "body {body} has a non-finite canvas position {:?} at knot {knot}",
                    points[knot]
                ));
            }
        }
        let xs = track.each_ref().map(|points| points.iter().map(|p| p[0]).collect());
        let ys = track.each_ref().map(|points| points.iter().map(|p| p[1]).collect());
        drop(track);

        let tidal = Tidal::new(masses, config)?;
        let mut track = Self {
            duration: 1.0,
            tidal,
            knots,
            last: (knots - 1) as f64,
            xs,
            ys,
            speed: Vec::new(),
            speed_blocks: Vec::new(),
        };
        let median = track.median_fraction_speed();
        let duration = median / reference_speed;
        if !(duration.is_finite() && duration > 0.0) {
            return degenerate(format!(
                "median body speed {median} per orbit fraction gives the duration {duration}"
            ));
        }
        track.duration = duration;
        track.tidal.reference = track.sample_tidal_reference();
        track.build_speed_table()?;
        Ok(track)
    }

    /// Fluid time of the last recorded knot (`T`).
    pub(crate) fn duration(&self) -> f64 {
        self.duration
    }

    /// Planning figures of this track on the canvas of the given aspect, for a fluid grid of
    /// spacing `dx` (see [`TrackSurvey`]).
    pub(crate) fn survey(&self, aspect: f64, dx: f64, config: &EmberConfig) -> TrackSurvey {
        let fluid = &config.fluid;
        // The solver's step rule (`fluid`: h = min(cfl·dx/speed, max_dt)) over the speed table.
        let table_step = self.duration / SPEED_INTERVALS as f64;
        let (mut fluid_steps, mut peak) = (0.0, 0.0);
        for &speed in &self.speed[..SPEED_INTERVALS] {
            fluid_steps += table_step / min(fluid.cfl * dx / max(speed, 1e-6), fluid.max_dt);
            peak = max(peak, speed);
        }
        let mut edge_clearance = f64::INFINITY;
        for (xs, ys) in self.xs.iter().zip(&self.ys) {
            for (x, y) in xs.iter().zip(ys) {
                edge_clearance = min(edge_clearance, min(aspect - x.abs(), 1.0 - y.abs()));
            }
        }
        let touching = 2.0 * fluid.body_radius;
        let overlapping = (0..=TIDAL_INTERVALS)
            .filter(|&i| {
                let p = self.positions_at(self.uniform_time(i, TIDAL_INTERVALS));
                [(0, 1), (0, 2), (1, 2)].iter().any(|&(a, b)| {
                    let (dx, dy) = (p[a][0] - p[b][0], p[a][1] - p[b][1]);
                    dx * dx + dy * dy < touching * touching
                })
            })
            .count();
        TrackSurvey {
            fluid_steps,
            peak_speed: peak / fluid.reference_speed,
            edge_clearance,
            overlap_fraction: overlapping as f64 / (TIDAL_INTERVALS + 1) as f64,
        }
    }

    /// Number of recorded knots `N` (test-only: the pipeline knows `N` from the recording before
    /// the track exists).
    #[cfg(test)]
    pub(crate) fn knots(&self) -> usize {
        self.knots
    }

    /// Fluid time of recorded knot `k ∈ [0, N)`: `T·k/(N-1)`, evaluated as `(T·k)/(N-1)` and
    /// exact at both ends (`0` and `T`; see [`Self::uniform_time`]). Strictly increasing in `k`
    /// (consecutive times differ by `T/(N-1)`, far above the rounding error for any `N < 2⁵⁰`).
    pub(crate) fn knot_time(&self, k: usize) -> f64 {
        self.uniform_time(k, self.knots - 1)
    }

    /// `T·i/n` for `0 ≤ i ≤ n`, evaluated as `(T·i)/n`, except that `i = n` returns `T` itself.
    ///
    /// `(T·n)/n` misses `T` by one ulp for a few percent of `T` values, and the endpoint must be
    /// exact: the final knot is the still, rendered at the solver's final time `T`, where
    /// `bodies_at` must evaluate the last segment at `w = 1` (the prototype's `linspace` also
    /// stores its endpoint exactly). The map stays non-decreasing: for `i < n < 2⁵²`,
    /// `(T·i)/n ≤ T·(1 - 1/n)(1 + 2⁻⁵³)² < T`, so rounding keeps it at or below `T`.
    fn uniform_time(&self, i: usize, n: usize) -> f64 {
        if i == n { self.duration } else { self.duration * (i as f64) / n as f64 }
    }

    /// The orbit's reference tidal anisotropy `Δ_ref` (see [`Tidal`]).
    pub(crate) fn tidal_reference(&self) -> f64 {
        self.tidal.reference
    }

    /// World position of `body` at recorded knot `knot` (both in range).
    #[cfg(test)]
    fn knot_position(&self, body: usize, knot: usize) -> [f64; 2] {
        [self.xs[body][knot], self.ys[body][knot]]
    }

    /// Segment containing source fraction `f ∈ [0, 1]`: `(left, w)` with `fi = f·(N-1)`,
    /// `left = min(⌊fi⌋, N-2)` and `w = fi - left`.
    fn segment(&self, f: f64) -> (usize, f64) {
        let fi = f * self.last;
        let left = (fi.floor() as usize).min(self.knots - 2);
        (left, fi - left as f64)
    }

    /// Source fraction of fluid time `t`: `clamp(t/T, 0, 1)` (a NaN or `-0` maps to `+0`).
    fn fraction(&self, t: f64) -> f64 {
        let f = t / self.duration;
        if f > 1.0 {
            1.0
        } else if f > 0.0 {
            f
        } else {
            0.0
        }
    }

    /// Displacement of `body` along segment `left`: `pos_{left+1} - pos_left`.
    fn chord(&self, body: usize, left: usize) -> [f64; 2] {
        let (x, y) = (&self.xs[body], &self.ys[body]);
        [x[left + 1] - x[left], y[left + 1] - y[left]]
    }

    /// `body` at fluid time `t` (docs/ember-design.md §3.7): linear between knots, with the
    /// outgoing segment's velocity `(pos_{left+1} - pos_left)·(N-1)/T`.
    fn state_at(&self, body: usize, t: f64) -> BodyState {
        let (left, w) = self.segment(self.fraction(t));
        let d = self.chord(body, left);
        BodyState {
            position: [self.xs[body][left] + w * d[0], self.ys[body][left] + w * d[1]],
            velocity: [d[0] * self.last / self.duration, d[1] * self.last / self.duration],
            shape: Shape::default(),
        }
    }

    /// Median over the `3 × 200001` samples `f_j = j/200000` of `|dpos/df|`
    /// (docs/ember-design.md §3.6).
    fn median_fraction_speed(&self) -> f64 {
        let mut speeds = Vec::with_capacity(BODIES * (MEDIAN_INTERVALS + 1));
        for j in 0..=MEDIAN_INTERVALS {
            let (left, _) = self.segment(j as f64 / MEDIAN_INTERVALS as f64);
            for body in 0..BODIES {
                let d = self.chord(body, left);
                let (vx, vy) = (d[0] * self.last, d[1] * self.last);
                speeds.push((vx * vx + vy * vy).sqrt());
            }
        }
        // The middle order statistic of an odd count: the same value for any selection algorithm.
        let middle = speeds.len() / 2;
        let (_, median, _) = speeds.select_nth_unstable_by(middle, f64::total_cmp);
        *median
    }

    /// World positions of the three centres at fluid time `t`.
    fn positions_at(&self, t: f64) -> [[f64; 2]; 3] {
        std::array::from_fn(|body| self.state_at(body, t).position)
    }

    /// The bodies' shapes at `t`, with their turning and stretch rates from forward differences
    /// over [`SHAPE_RATE_STEP`]. Like the positions, the shapes are clamped to `[0, T]` (a NaN
    /// maps to 0); at the end of the orbit, where the positions stop, the rates are zero.
    fn shapes_at(&self, t: f64) -> [Shape; 3] {
        let t = if t > self.duration {
            self.duration
        } else if t > 0.0 {
            t
        } else {
            0.0
        };
        let now = self.tidal.principal(self.positions_at(t));
        let later = self.tidal.principal(self.positions_at(t + SHAPE_RATE_STEP));
        std::array::from_fn(|body| {
            let (anisotropy, axis) = now[body];
            let (later_anisotropy, later_axis) = later[body];
            // The eigenvector's sign is arbitrary: align the later axis with this one.
            let later_axis = if axis[0] * later_axis[0] + axis[1] * later_axis[1] < 0.0 {
                [-later_axis[0], -later_axis[1]]
            } else {
                later_axis
            };
            let semi = self.tidal.semi_axes(anisotropy);
            let later_semi = self.tidal.semi_axes(later_anisotropy);
            Shape {
                semi,
                axis,
                spin: (axis[0] * later_axis[1] - axis[1] * later_axis[0]) / SHAPE_RATE_STEP,
                strain: (later_semi[0] - semi[0]) / (semi[0] * SHAPE_RATE_STEP),
            }
        })
    }

    /// The orbit's reference tidal anisotropy: the `stretch_quantile` order statistic of every
    /// body's anisotropy at `4001` uniform times on `[0, T]` (exact for any sort).
    fn sample_tidal_reference(&self) -> f64 {
        let mut samples = Vec::with_capacity(BODIES * (TIDAL_INTERVALS + 1));
        for i in 0..=TIDAL_INTERVALS {
            let t = self.uniform_time(i, TIDAL_INTERVALS);
            samples.extend(self.tidal.principal(self.positions_at(t)).map(|(a, _)| a));
        }
        samples.sort_unstable_by(f64::total_cmp);
        let at = ((samples.len() - 1) as f64 * self.tidal.quantile).round() as usize;
        samples[at.min(samples.len() - 1)]
    }

    /// Fluid time of speed-table entry `i`: `T·i/400000`, evaluated as `(T·i)/400000` with the
    /// last entry exactly `T` ([`Self::uniform_time`]; non-decreasing in `i`).
    fn table_time(&self, i: usize) -> f64 {
        self.uniform_time(i, SPEED_INTERVALS)
    }

    /// Fills the speed table `speed[i] = max_b (|vel_b| + deformation speed_b)(T·i/400000)` and
    /// its block maxima.
    fn build_speed_table(&mut self) -> EmberResult<()> {
        let mut speed = Vec::with_capacity(SPEED_INTERVALS + 1);
        for i in 0..=SPEED_INTERVALS {
            let t = self.table_time(i);
            let mut fastest = 0.0f64;
            for body in self.bodies_at(t) {
                let v = body.velocity;
                fastest = max(
                    fastest,
                    (v[0] * v[0] + v[1] * v[1]).sqrt() + body.shape.deformation_speed(),
                );
            }
            if !fastest.is_finite() {
                return degenerate(format!("body speed at fluid time {t} is not finite"));
            }
            speed.push(fastest);
        }
        self.speed_blocks =
            speed.chunks(SPEED_BLOCK).map(|block| block.iter().copied().fold(0.0, max)).collect();
        self.speed = speed;
        Ok(())
    }

    /// First speed-table index `i` with `T·i/400000 ≥ t` (`400001` if none, e.g. for NaN): the
    /// `searchsorted` of the prototype, found from an estimate and corrected against the exact
    /// table times.
    fn first_entry_at_or_after(&self, t: f64) -> usize {
        let end = SPEED_INTERVALS + 1;
        if t.is_nan() {
            return end;
        }
        let estimate = (t / self.duration * SPEED_INTERVALS as f64).ceil();
        let mut i = if estimate > 0.0 { (estimate as usize).min(end) } else { 0 };
        while i > 0 && self.table_time(i - 1) >= t {
            i -= 1;
        }
        while i < end && self.table_time(i) < t {
            i += 1;
        }
        i
    }

    /// Maximum of `speed[lo..hi]` (`lo < hi ≤ 400001`) via the block maxima; equal to a plain scan
    /// because a maximum of finite values does not depend on the order of comparisons.
    fn speed_max(&self, lo: usize, hi: usize) -> f64 {
        let first_block = lo.div_ceil(SPEED_BLOCK);
        let last_block = hi / SPEED_BLOCK;
        if first_block >= last_block {
            return self.speed[lo..hi].iter().copied().fold(0.0, max);
        }
        let head = self.speed[lo..first_block * SPEED_BLOCK].iter().copied().fold(0.0, max);
        let body = self.speed_blocks[first_block..last_block].iter().copied().fold(head, max);
        self.speed[last_block * SPEED_BLOCK..hi].iter().copied().fold(body, max)
    }
}

impl BodyMotion for BodyTrack {
    fn bodies_at(&self, t: f64) -> [BodyState; 3] {
        let shapes = self.shapes_at(t);
        std::array::from_fn(|body| BodyState { shape: shapes[body], ..self.state_at(body, t) })
    }

    /// `max speed[i]` over `i ∈ [max(i₀-2, 0), min(i₀+400, 400001))`, `i₀` the first table entry at
    /// or after `t` (docs/ember-design.md §3.8; the look-ahead of the prototype's CFL control).
    /// This is the maximum of the *sampled* speed table over the look-ahead window, not a strict
    /// bound on the continuous motion (see [`BodyMotion::speed_bound`]).
    fn speed_bound(&self, t: f64) -> f64 {
        let i0 = self.first_entry_at_or_after(t);
        let lo = i0.saturating_sub(LOOK_BEHIND);
        let hi = (i0 + LOOK_AHEAD).min(SPEED_INTERVALS + 1);
        self.speed_max(lo, hi)
    }
}

/// The tidal field of the other two bodies, and the elliptical shape it gives each body
/// (docs/ember-design.md §3.10).
///
/// On the canvas, body `j` exerts on body `i` the Plummer-softened tidal tensor
/// `T = w_j·(3·r·rᵀ - ρ²·I)/ρ⁵`, `r = x_j - x_i`, `ρ² = |r|² + ε²`, `w_j = m_j/m̄` (the initial
/// masses relative to their mean). A fluid body is stretched by the traceless part of the summed
/// tensor, so the stretch follows its anisotropy `Δ = λ₁ - λ₂` (the difference of its
/// eigenvalues) along the eigenvector of `λ₁`. With `x = Δ/Δ_ref` (`Δ_ref` the orbit's
/// `stretch_quantile` anisotropy) the axis ratio is
/// `A = 1 + (max_aspect - 1)·x/(1 + x)` and the area is kept: `a = R·√A`, `b = R/√A`. Where the
/// tensor is isotropic its axes are undefined, but there `Δ = 0` and the body is a disc.
#[derive(Clone, Copy, Debug)]
struct Tidal {
    /// Masses relative to their mean.
    weights: [f64; 3],
    /// `ε²` of the Plummer softening.
    softening2: f64,
    /// Largest axis ratio (1: the bodies stay discs).
    max_aspect: f64,
    /// Order statistic of the reference anisotropy (a fraction in `(0, 1]`).
    quantile: f64,
    /// Reference anisotropy `Δ_ref` (set once the orbit's duration is known; `≤ 0`: no stretch).
    reference: f64,
    /// Radius of the disc of equal area.
    radius: f64,
}

impl Tidal {
    /// The model of `config.tidal` for bodies of these initial `masses` (finite, positive).
    fn new(masses: [f64; 3], config: &EmberConfig) -> EmberResult<Self> {
        for (body, mass) in masses.iter().enumerate() {
            require_positive(*mass, &format!("mass of body {body}"))?;
        }
        let mean = (masses[0] + masses[1] + masses[2]) / 3.0;
        let t = &config.tidal;
        Ok(Self {
            weights: masses.map(|m| m / mean),
            softening2: t.softening * t.softening,
            max_aspect: t.max_aspect,
            quantile: t.stretch_quantile,
            reference: 0.0,
            radius: config.fluid.body_radius,
        })
    }

    /// Per body, the tidal anisotropy `Δ ≥ 0` and the unit stretch axis at these positions.
    fn principal(&self, positions: [[f64; 2]; 3]) -> [(f64, [f64; 2]); 3] {
        std::array::from_fn(|i| {
            let (mut p, mut q, mut s) = (0.0, 0.0, 0.0);
            for (j, weight) in self.weights.iter().enumerate() {
                if j == i {
                    continue;
                }
                let r = [positions[j][0] - positions[i][0], positions[j][1] - positions[i][1]];
                let rho2 = r[0] * r[0] + r[1] * r[1] + self.softening2;
                let f = weight / (rho2 * rho2 * rho2.sqrt());
                p += f * (3.0 * r[0] * r[0] - rho2);
                q += f * (3.0 * r[0] * r[1]);
                s += f * (3.0 * r[1] * r[1] - rho2);
            }
            let half = 0.5 * (p - s);
            let root = (half * half + q * q).sqrt();
            // Eigenvector of λ₁ = (p + s)/2 + root: the better conditioned of (λ₁ - s, q) and
            // (q, λ₁ - p); an isotropic tensor has none and gets the x axis (its Δ is 0).
            let (u, v) = (half + root, q);
            let (w, z) = (q, root - half);
            let (x, y) = if u * u + v * v >= w * w + z * z { (u, v) } else { (w, z) };
            let norm = (x * x + y * y).sqrt();
            let axis = if norm > 0.0 { [x / norm, y / norm] } else { [1.0, 0.0] };
            (2.0 * root, axis)
        })
    }

    /// Semi-axes `(a, b)` of a body under the anisotropy `Δ` (area of the disc of radius `R`).
    fn semi_axes(&self, anisotropy: f64) -> [f64; 2] {
        if !(self.reference > 0.0 && self.max_aspect > 1.0) {
            return [self.radius, self.radius];
        }
        let x = anisotropy / self.reference;
        let aspect = 1.0 + (self.max_aspect - 1.0) * x / (1.0 + x);
        let k = aspect.sqrt();
        [self.radius * k, self.radius / k]
    }
}

/// `Err(DegenerateOrbit)` with `reason`.
fn degenerate<T>(reason: String) -> EmberResult<T> {
    Err(EmberError::DegenerateOrbit { reason })
}

/// Rejects a parameter that is not finite and strictly positive.
fn require_positive(value: f64, parameter: &str) -> EmberResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(EmberError::InvalidConfig {
            parameter: parameter.to_string(),
            reason: format!("{value} must be finite and > 0"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::math;
    use super::*;

    /// Deterministic pseudo-random numbers in `[0, 1)` (64-bit LCG, top 53 bits).
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    const TAU: f64 = 2.0 * std::f64::consts::PI;

    /// A canvas track.
    type Track = [Vec<[f64; 2]>; BODIES];

    /// Three bodies on the circle of radius `radius` about the origin, a third of a turn apart,
    /// `n` knots over one revolution (`θ_k = 2πk/(n-1)`).
    fn circle(n: usize, radius: f64) -> Track {
        std::array::from_fn(|b| {
            (0..n)
                .map(|k| {
                    let theta = TAU * k as f64 / (n - 1) as f64 + TAU * b as f64 / 3.0;
                    let (s, c) = math::sin_cos(theta);
                    [radius * c, radius * s]
                })
                .collect()
        })
    }

    /// A generic (non-symmetric) track: three bodies on wobbly, drifting loops on the canvas.
    fn wobbly(n: usize) -> Track {
        std::array::from_fn(|b| {
            (0..n)
                .map(|k| {
                    let s = k as f64 / (n - 1) as f64;
                    let (s1, c1) = math::sin_cos(TAU * (1.0 + b as f64) * s + b as f64);
                    let (s2, c2) = math::sin_cos(TAU * 3.0 * s * s);
                    [0.8 * c1 + 0.1 * s2 + 0.25 * (b as f64 - 1.0), 0.55 * s1 - 0.08 * c2]
                })
                .collect()
        })
    }

    fn config() -> EmberConfig {
        EmberConfig::default()
    }

    /// Euclidean length of a 2-vector.
    fn norm(v: [f64; 2]) -> f64 {
        (v[0] * v[0] + v[1] * v[1]).sqrt()
    }

    /// Unequal masses, so the tidal weights are exercised.
    const MASSES: [f64; 3] = [1.0, 1.4, 0.7];

    #[test]
    fn circle_duration_makes_the_median_speed_the_reference_speed() {
        let n = 1201;
        let mut cfg = config();
        cfg.fluid.reference_speed = 2.5;
        let radius = 0.6;
        let track = BodyTrack::new(circle(n, radius), MASSES, &cfg).unwrap();
        // Every chord has the same length 2ρ sin(π/(N-1)), so the median fraction speed is exact.
        let fraction_speed =
            2.0 * radius * math::sin(std::f64::consts::PI / (n - 1) as f64) * (n - 1) as f64;
        let expected = fraction_speed / 2.5;
        assert!((track.duration() - expected).abs() < 1e-9 * expected, "{}", track.duration());
        // Body speed is the reference speed everywhere.
        let bodies = track.bodies_at(0.37 * track.duration());
        for body in bodies {
            assert!((norm(body.velocity) - 2.5).abs() < 1e-9, "{:?}", body.velocity);
        }
    }

    /// The knots are the canvas track's points, unchanged.
    #[test]
    fn the_track_keeps_the_canvas_knots() {
        let points = wobbly(301);
        let track = BodyTrack::new(points.clone(), MASSES, &config()).unwrap();
        for (b, body) in points.iter().enumerate() {
            for (k, point) in body.iter().enumerate() {
                assert_eq!(track.knot_position(b, k), *point);
            }
        }
        let again = BodyTrack::new(points, MASSES, &config()).unwrap();
        assert_eq!(track.duration.to_bits(), again.duration.to_bits());
        assert_eq!(track.speed, again.speed);
    }

    #[test]
    fn bodies_at_knot_times_sit_on_the_knots() {
        let n = 5001;
        let track = BodyTrack::new(wobbly(n), MASSES, &config()).unwrap();
        assert_eq!(track.knots(), n);
        assert_eq!(track.knot_time(0).to_bits(), 0.0f64.to_bits());
        assert_eq!(track.knot_time(n - 1), track.duration());
        for k in 0..n {
            let bodies = track.bodies_at(track.knot_time(k));
            for (b, body) in bodies.iter().enumerate() {
                let knot = track.knot_position(b, k);
                for (got, expected) in body.position.iter().zip(knot) {
                    assert!((got - expected).abs() < 1e-14, "{k} {b}");
                }
            }
        }
    }

    /// `(T·(N-1))/(N-1) ≠ T` for a few percent of durations; the final knot (the still) and the
    /// last speed-table entry must nevertheless sit exactly at `T`, with the times non-decreasing
    /// up to it, for every duration.
    #[test]
    fn the_final_knot_and_table_entry_are_exactly_the_duration() {
        let mut track = BodyTrack::new(wobbly(1001), MASSES, &config()).unwrap();
        let mut rng = Lcg(23);
        let mut naive_misses = 0;
        for _ in 0..20_000 {
            track.duration = 5.0 + 15.0 * rng.next();
            let t = track.duration;
            for (n, table) in [(track.knots - 1, false), (SPEED_INTERVALS, true)] {
                let time = |i| if table { track.table_time(i) } else { track.knot_time(i) };
                assert_eq!(time(n), t);
                assert_eq!(time(0).to_bits(), 0.0f64.to_bits());
                assert!(time(n - 2) <= time(n - 1) && time(n - 1) < t);
                naive_misses += usize::from(t * n as f64 / n as f64 != t);
            }
        }
        // The special case is needed (the naive formula misses the endpoint).
        assert!(naive_misses > 100, "{naive_misses}");
        // At `T` the bodies sit at the last segment's end (w = 1).
        track.duration = 7.123456789;
        let last = track.segment(track.fraction(track.knot_time(track.knots - 1)));
        assert_eq!(last, (track.knots - 2, 1.0));
    }

    #[test]
    fn bodies_move_linearly_between_knots_with_the_outgoing_velocity() {
        let n = 101;
        let track = BodyTrack::new(wobbly(n), MASSES, &config()).unwrap();
        let t_scale = track.duration() / (n - 1) as f64;
        for k in [0, 1, 50, 98, 99] {
            let t = (k as f64 + 0.25) * t_scale;
            for (b, body) in track.bodies_at(t).iter().enumerate() {
                let (p, q) = (track.knot_position(b, k), track.knot_position(b, k + 1));
                for axis in 0..2 {
                    let expected = p[axis] + 0.25 * (q[axis] - p[axis]);
                    assert!((body.position[axis] - expected).abs() < 1e-13);
                    let velocity = (q[axis] - p[axis]) / t_scale;
                    assert!(
                        (body.velocity[axis] - velocity).abs() < 1e-9 * velocity.abs().max(1.0)
                    );
                }
            }
        }
        // Clamped outside [0, T]; the final knot keeps the incoming segment's velocity.
        assert_eq!(track.bodies_at(-5.0), track.bodies_at(0.0));
        assert_eq!(track.bodies_at(f64::INFINITY), track.bodies_at(track.duration()));
        let last = track.bodies_at(track.duration());
        for (b, body) in last.iter().enumerate() {
            let knot = track.knot_position(b, n - 1);
            assert!((body.position[0] - knot[0]).abs() < 1e-15);
            assert!((body.position[1] - knot[1]).abs() < 1e-15);
        }
    }

    #[test]
    fn speed_bound_equals_its_definition() {
        let n = 777;
        let track = BodyTrack::new(wobbly(n), MASSES, &config()).unwrap();
        let times: Vec<f64> = (0..=SPEED_INTERVALS).map(|i| track.table_time(i)).collect();
        assert!(times.windows(2).all(|w| w[0] <= w[1]));
        let naive = |t: f64| {
            // First index with ts ≥ t (none for NaN), like numpy's searchsorted.
            let i0 = times.partition_point(|&ts| ts < t || t.is_nan());
            let lo = i0.saturating_sub(2);
            let hi = (i0 + 400).min(times.len());
            track.speed[lo..hi].iter().copied().fold(0.0, f64::max)
        };
        let mut rng = Lcg(11);
        let mut probes = vec![-1.0, 0.0, track.duration(), 2.0 * track.duration(), f64::NAN];
        for _ in 0..3000 {
            probes.push(rng.next() * track.duration());
            let i = (rng.next() * SPEED_INTERVALS as f64) as usize;
            probes.extend([times[i], times[i].next_up(), times[i].next_down()]);
        }
        probes.extend([times[SPEED_INTERVALS].next_down(), times[1].next_down()]);
        for t in probes {
            assert_eq!(track.speed_bound(t).to_bits(), naive(t).to_bits(), "t = {t}");
        }
        // Here every segment is sampled (777 knots), so the look-ahead covers the true body speeds
        // just ahead.
        let t = 0.4 * track.duration();
        let ahead = track.bodies_at(t + 1e-4 * track.duration());
        for body in ahead {
            assert!(norm(body.velocity) <= track.speed_bound(t));
        }
    }

    /// The table holds the fastest material speed: centre speed plus deformation speed.
    #[test]
    fn speed_table_entries_are_the_largest_body_speed() {
        let track = BodyTrack::new(wobbly(333), MASSES, &config()).unwrap();
        for i in [0, 1, 12345, 399_999, 400_000] {
            let fastest = track
                .bodies_at(track.table_time(i))
                .iter()
                .map(|b| norm(b.velocity) + b.shape.deformation_speed())
                .fold(0.0, f64::max);
            assert_eq!(track.speed[i], fastest);
        }
        assert_eq!(track.speed.len(), SPEED_INTERVALS + 1);
    }

    #[test]
    fn degenerate_tracks_are_rejected() {
        let is_degenerate = |track: Track| {
            matches!(
                BodyTrack::new(track, MASSES, &config()),
                Err(EmberError::DegenerateOrbit { .. })
            )
        };
        let good = wobbly(50);
        assert!(!is_degenerate(good.clone()));
        // Unequal and too short recordings.
        let mut unequal = good.clone();
        unequal[1].pop();
        assert!(is_degenerate(unequal));
        assert!(is_degenerate(good.clone().map(|body| body[..1].to_vec())));
        // Non-finite coordinates.
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut track = good.clone();
            track[2][17][1] = bad;
            assert!(is_degenerate(track));
        }
        // Distinct but motionless bodies: zero median speed.
        let still: Track = [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]].map(|point| vec![point; 20]);
        assert!(is_degenerate(still));
    }

    #[test]
    fn invalid_masses_and_speeds_are_config_errors() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let masses = [1.0, bad, 1.0];
            assert!(matches!(
                BodyTrack::new(wobbly(20), masses, &config()),
                Err(EmberError::InvalidConfig { parameter, .. }) if parameter == "mass of body 1"
            ));
            let mut cfg = config();
            cfg.fluid.reference_speed = bad;
            assert!(matches!(
                BodyTrack::new(wobbly(20), MASSES, &cfg),
                Err(EmberError::InvalidConfig { parameter, .. })
                    if parameter == "fluid.reference_speed"
            ));
        }
    }

    /// The planning figures against their definitions on a track whose answers are known.
    #[test]
    fn the_survey_reports_cost_clearance_and_overlap() {
        let cfg = config();
        let (aspect, dx) = (1.5, 0.002);
        // Three bodies a third of a turn apart on a circle: constant speed, never close.
        let radius = 0.6;
        let track = BodyTrack::new(circle(2401, radius), MASSES, &cfg).unwrap();
        let survey = track.survey(aspect, dx, &cfg);
        assert!((survey.edge_clearance - (1.0 - radius)).abs() < 1e-6, "{survey:?}");
        assert_eq!(survey.overlap_fraction, 0.0);
        // Speed is the reference speed (the bodies barely deform), so the step is the CFL step.
        assert!(survey.peak_speed > 0.999 && survey.peak_speed < 1.2, "{survey:?}");
        let cfl_steps = track.duration() / (cfg.fluid.cfl * dx / cfg.fluid.reference_speed);
        assert!(survey.fluid_steps >= cfl_steps * 0.999, "{survey:?} vs {cfl_steps}");
        assert!(survey.fluid_steps < cfl_steps * 1.2, "{survey:?} vs {cfl_steps}");

        // Two bodies kept closer than two radii overlap for the whole orbit.
        let mut close = circle(601, radius);
        close[1] = close[0].iter().map(|p| [p[0] + 0.5 * cfg.fluid.body_radius, p[1]]).collect();
        let survey = BodyTrack::new(close, MASSES, &cfg).unwrap().survey(aspect, dx, &cfg);
        assert_eq!(survey.overlap_fraction, 1.0);
    }

    /// A shape with generic axes, stretch and rates.
    fn stretched() -> Shape {
        let (s, c) = math::sin_cos(0.83);
        Shape { semi: [0.07, 0.03], axis: [c, s], spin: 2.7, strain: -1.9 }
    }

    /// World offset of the outline point at parameter `phi` (and the scaled point `scale`).
    fn outline(shape: &Shape, phi: f64, scale: f64) -> [f64; 2] {
        let (s, c) = math::sin_cos(phi);
        let [x, y] = [scale * shape.semi[0] * c, scale * shape.semi[1] * s];
        let [ca, sa] = shape.axis;
        [ca * x - sa * y, sa * x + ca * y]
    }

    #[test]
    fn a_disc_has_the_exact_signed_distance() {
        let disc = Shape::disc(0.05);
        let mut rng = Lcg(3);
        for _ in 0..1000 {
            let d = [rng.next() * 0.4 - 0.2, rng.next() * 0.4 - 0.2];
            let exact = norm(d) - 0.05;
            assert!((disc.signed_distance(d) - exact).abs() < 1e-15, "{d:?}");
            assert_eq!(disc.contains(d), exact < 0.0);
        }
        assert_eq!(disc.signed_distance([0.0, 0.0]), -0.05);
        assert_eq!(disc.deformation_speed(), 0.0);
        assert_eq!(disc.deformation_velocity([0.03, -0.01]), [0.0, 0.0]);
    }

    #[test]
    fn the_signed_distance_is_signed_by_the_outline_and_first_order_accurate() {
        let shape = stretched();
        assert_eq!(shape.signed_distance([0.0, 0.0]), -0.03, "the depth at the centre is b");
        assert_eq!(shape.extent(), 0.07);
        for k in 0..360 {
            let phi = TAU * f64::from(k) / 360.0;
            for scale in [0.2, 0.9, 0.999, 1.001, 1.1, 3.0] {
                let d = outline(&shape, phi, scale);
                let sd = shape.signed_distance(d);
                assert_eq!(sd < 0.0, scale < 1.0, "phi {phi}, scale {scale}");
                assert_eq!(shape.contains(d), scale < 1.0);
            }
            // Near the outline the distance is accurate to second order in the offset: step out
            // along the outward normal and compare.
            let on = outline(&shape, phi, 1.0);
            let [x, y] = shape.to_body(on);
            let [a, b] = shape.semi;
            let n_body = [x / (a * a), y / (b * b)];
            let n_len = norm(n_body);
            let [ca, sa] = shape.axis;
            let normal = [
                (ca * n_body[0] - sa * n_body[1]) / n_len,
                (sa * n_body[0] + ca * n_body[1]) / n_len,
            ];
            for h in [-1e-4, 1e-4] {
                let sd = shape.signed_distance([on[0] + h * normal[0], on[1] + h * normal[1]]);
                assert!((sd - h).abs() < 2e-3 * h.abs(), "phi {phi}: {sd} vs {h}");
            }
        }
    }

    /// The deformation flow is divergence-free and irrotational, moves the outline onto the
    /// turned, stretched outline, and never exceeds its speed bound inside the body.
    #[test]
    fn the_deformation_flow_carries_the_outline_and_keeps_the_area() {
        let shape = stretched();
        let h = 1e-6;
        let at = |d: [f64; 2]| shape.deformation_velocity(d);
        let p = [0.011, -0.017];
        let (ux1, ux0) = (at([p[0] + h, p[1]]), at([p[0] - h, p[1]]));
        let (uy1, uy0) = (at([p[0], p[1] + h]), at([p[0], p[1] - h]));
        let divergence = (ux1[0] - ux0[0] + uy1[1] - uy0[1]) / (2.0 * h);
        let vorticity = (ux1[1] - ux0[1] - uy1[0] + uy0[0]) / (2.0 * h);
        assert!(divergence.abs() < 1e-8 && vorticity.abs() < 1e-8, "{divergence} {vorticity}");

        // After a short time δ the outline has turned by Ω·δ and stretched by e·δ.
        let delta = 1e-6;
        let (sd, cd) = math::sin_cos(shape.spin * delta);
        let [ca, sa] = shape.axis;
        let e = shape.strain * delta;
        let later = Shape {
            semi: [shape.semi[0] * (1.0 + e), shape.semi[1] * (1.0 - e)],
            axis: [ca * cd - sa * sd, sa * cd + ca * sd],
            ..shape
        };
        let bound = shape.deformation_speed();
        for k in 0..720 {
            let d = outline(&shape, TAU * f64::from(k) / 720.0, 1.0);
            let u = at(d);
            let moved = [d[0] + delta * u[0], d[1] + delta * u[1]];
            assert!(later.signed_distance(moved).abs() < 1e-3 * delta * bound, "{k}");
            assert!(norm(u) <= bound * (1.0 + 1e-12), "{k}: {} > {bound}", norm(u));
            let inner = at(outline(&shape, TAU * f64::from(k) / 720.0, 0.5));
            assert!(norm(inner) <= bound);
        }
    }

    /// A body with one companion is stretched along the line to it; a heavier companion stretches
    /// it more; the anisotropy falls off as the cube of the distance.
    #[test]
    fn the_tidal_axis_points_at_the_companion() {
        let mut cfg = config();
        cfg.tidal.softening = 1e-6;
        let tidal = Tidal::new([1.0, 1.0, 1.0], &cfg).unwrap();
        let far = [1e6, 1e6];
        let (s, c) = math::sin_cos(0.6);
        let near = tidal.principal([[0.0, 0.0], [0.3 * c, 0.3 * s], far]);
        let (anisotropy, axis) = near[0];
        assert!((axis[0] * s - axis[1] * c).abs() < 1e-12, "{axis:?}");
        // Δ = 3·w/r³ (λ₁ = 2w/r³, λ₂ = -w/r³).
        assert!((anisotropy - 3.0 / (0.3f64 * 0.3 * 0.3)).abs() < 1e-6 * anisotropy);
        let farther = tidal.principal([[0.0, 0.0], [0.6 * c, 0.6 * s], far])[0].0;
        assert!((anisotropy / farther - 8.0).abs() < 1e-6);
        let heavy = Tidal::new([1.0, 2.0, 1.0], &cfg).unwrap();
        let heavier = heavy.principal([[0.0, 0.0], [0.3 * c, 0.3 * s], far])[0].0;
        assert!((heavier / anisotropy - 2.0 * 3.0 / 4.0).abs() < 1e-9, "weights m/mean");
        // Symmetric: both bodies of the pair feel the same (unweighted) stretch along the line.
        assert!((near[1].0 - anisotropy).abs() < 1e-9 * anisotropy);
    }

    #[test]
    fn semi_axes_keep_the_area_and_saturate_at_the_largest_aspect() {
        let mut tidal = Tidal::new([1.0; 3], &config()).unwrap();
        let r = config().fluid.body_radius;
        assert_eq!(tidal.semi_axes(5.0), [r, r], "no reference yet: a disc");
        tidal.reference = 2.0;
        let max_aspect = config().tidal.max_aspect;
        let mut previous = 1.0;
        for anisotropy in [0.0, 0.1, 1.0, 2.0, 5.0, 1e3, 1e12] {
            let [a, b] = tidal.semi_axes(anisotropy);
            assert!((a * b - r * r).abs() < 1e-15, "area");
            let aspect = a / b;
            assert!(aspect >= previous && aspect < max_aspect * (1.0 + 1e-9));
            previous = aspect;
        }
        let [a, b] = tidal.semi_axes(2.0);
        assert!((a / b - 0.5 * (1.0 + max_aspect)).abs() < 1e-12, "half way at the reference");
        assert_eq!(tidal.semi_axes(0.0), [r, r]);
        tidal.max_aspect = 1.0;
        assert_eq!(tidal.semi_axes(1e9), [r, r], "max_aspect 1 keeps discs");
    }

    /// The track's shapes: the reference is the configured quantile, the stretch follows it, the
    /// rates are the shapes' own time derivatives, and at the end of the orbit they stop.
    #[test]
    fn track_shapes_follow_the_tidal_field() {
        let track = BodyTrack::new(wobbly(2001), MASSES, &config()).unwrap();
        let reference = track.tidal_reference();
        assert!(reference > 0.0 && reference.is_finite());
        let mut samples = Vec::new();
        for i in 0..=TIDAL_INTERVALS {
            let t = track.uniform_time(i, TIDAL_INTERVALS);
            samples.extend(track.tidal.principal(track.positions_at(t)).map(|(a, _)| a));
        }
        let below = samples.iter().filter(|&&a| a < reference).count() as f64;
        let quantile = config().tidal.stretch_quantile;
        assert!((below / samples.len() as f64 - quantile).abs() < 1e-3, "{below}");

        let r = config().fluid.body_radius;
        let (mut stretched, mut rates) = (0, 0);
        for k in 1..400 {
            let t = track.duration() * f64::from(k) / 400.0;
            let now = track.bodies_at(t);
            let later = track.bodies_at(t + 1e-6);
            for (body, (s, l)) in now.iter().zip(&later).enumerate() {
                let shape = s.shape;
                assert!((shape.semi[0] * shape.semi[1] - r * r).abs() < 1e-15);
                assert!(shape.semi[0] >= shape.semi[1] && (norm(shape.axis) - 1.0).abs() < 1e-15);
                stretched += usize::from(shape.semi[0] / shape.semi[1] > 1.5);
                // The rates match the shape's own change (within the difference steps' error).
                let turned = shape.axis[0] * l.shape.axis[1] - shape.axis[1] * l.shape.axis[0];
                let turned =
                    if shape.axis[0] * l.shape.axis[0] + shape.axis[1] * l.shape.axis[1] < 0.0 {
                        -turned
                    } else {
                        turned
                    };
                let spin = turned / 1e-6;
                let strain = (l.shape.semi[0] - shape.semi[0]) / (shape.semi[0] * 1e-6);
                let scale = shape.spin.abs().max(1.0);
                if (spin - shape.spin).abs() < 0.05 * scale
                    && (strain - shape.strain).abs() < 0.05 * shape.strain.abs().max(1.0)
                {
                    rates += 1;
                } else {
                    println!("t {t}, body {body}: spin {spin} vs {}", shape.spin);
                }
            }
        }
        assert!(stretched > 0, "the closest moments stretch the bodies");
        assert!(rates >= 3 * 399 - 3, "rates agree away from knots: {rates}");
        for body in track.bodies_at(track.duration()) {
            assert_eq!((body.shape.spin, body.shape.strain), (0.0, 0.0));
        }
    }

    #[test]
    fn a_largest_aspect_of_one_keeps_rigid_discs() {
        let mut cfg = config();
        cfg.tidal.max_aspect = 1.0;
        let track = BodyTrack::new(wobbly(501), MASSES, &cfg).unwrap();
        let r = cfg.fluid.body_radius;
        for k in 0..=50 {
            for body in track.bodies_at(track.duration() * f64::from(k) / 50.0) {
                assert_eq!(body.shape.semi, [r, r]);
                assert_eq!((body.shape.strain, body.shape.deformation_speed()), (0.0, 0.0));
            }
        }
    }

    /// Cost of the projection and time map at the production size (1,000,000 knots) and of the
    /// per-step queries. Run with
    /// `cargo test --release --lib ember::orbit::tests::track_throughput -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark"]
    fn track_throughput() {
        use std::time::Instant;
        let positions = wobbly(1_000_000);
        let clock = Instant::now();
        let track = BodyTrack::new(positions, MASSES, &config()).unwrap();
        println!(
            "BodyTrack::new, 1M knots: {:.3} s (T = {:.4})",
            clock.elapsed().as_secs_f64(),
            track.duration()
        );
        let queries = 1_000_000usize;
        let clock = Instant::now();
        let mut sum = 0.0;
        for i in 0..queries {
            let t = track.duration() * (i as f64 + 0.5) / queries as f64;
            sum += track.speed_bound(t) + track.bodies_at(t)[1].position[0];
        }
        let per = clock.elapsed().as_secs_f64() / queries as f64;
        println!("speed_bound + bodies_at: {:.1} ns per query ({sum:.3})", per * 1e9);
    }
}
