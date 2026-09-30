//! From the recorded 3-D orbit to three tidally stretched bodies moving on the canvas.
//!
//! The ember edition stirs the fluid with the three bodies of the selected orbit, seen from the
//! plane of their principal motion. This module turns the raw integrator positions
//! `p[body][knot] ∈ ℝ³` (`N` knots per body, knot `N-1` is the final step) into three moving
//! bodies in world coordinates, maps orbit knots to fluid time, and gives each body the elliptical
//! outline that the tidal field of the other two stretches it into ([`Tidal`], [`Shape`]). It ports the museum-lab
//! `estuary/source.py` projection (`Source.read`, `Source.sample`) and the `wake/ns.py` time map
//! (`Bodies`, `median_fraction_speed`); docs/ember-design.md §3 is the binding recipe.
//!
//! # Principal plane
//!
//! 1. Bounding box of all `3N` points: `origin = (low + high)/2`, `extent = max_axis(high - low)`.
//! 2. Normalised points `q = (p - origin)/extent`; their mean `μ` and covariance
//!    `C = Σ (q - μ)(q - μ)ᵀ / 3N`, both summed in scan order (knot-major, then body) with
//!    Neumaier-compensated summation.
//! 3. Eigen-decomposition of `C` by cyclic Jacobi rotations (fixed pivot order `(0,1), (0,2),
//!    (1,2)`), eigenpairs sorted by eigenvalue, descending. The orbit must span a plane:
//!    `λ₁ > 10⁻¹² λ₀`.
//! 4. Axis signs are fixed by the *anchor rule*: along axis `a` the first point in scan order whose
//!    `|v| = |(q - μ)·e_a|` reaches `max|v|·(1 - 10⁻¹²)` must have `v > 0`. This makes the
//!    projection independent of the eigensolver's arbitrary signs.
//! 5. `P = ((q - μ)·e₀, (q - μ)·e₁)`; with `centre = (min P + max P)/2` and
//!    `half = (max P - min P)/2`, the world position of a knot is
//!    `pos = (P - centre)·scale`, `scale = fill / max(half_x/aspect, half_y)`, so the orbit spans
//!    `±fill·aspect` in `x` or `±fill` in `y` (whichever binds) on the canvas
//!    `[-aspect, aspect] × [-1, 1]`.
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
//! Given the same input bits, every CPU produces the same projection, duration, positions and
//! shapes.

use std::fmt;

use nalgebra::Vector3;

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

/// Upper bound of Jacobi sweeps (convergence is quadratic; 3×3 needs well under ten).
const JACOBI_SWEEPS: usize = 64;

/// Off-diagonal magnitude below which the Jacobi iteration stops.
const JACOBI_TOLERANCE: f64 = 1e-300;

/// The second principal variance must exceed this fraction of the first (the orbit spans a plane).
const PLANE_RATIO: f64 = 1e-12;

/// Relative tolerance of the anchor rule's "maximal projection" test.
const ANCHOR_TOLERANCE: f64 = 1e-12;

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

/// Projection parameters, recorded in the certificate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Projection {
    /// Centre of the orbit's bounding box (3-D, original units).
    pub origin: [f64; 3],
    /// Largest bounding-box side (3-D, original units).
    pub extent: f64,
    /// First two principal axes (unit vectors in normalised 3-D space).
    pub axes: [[f64; 3]; 2],
    /// World units per normalised projected unit.
    pub scale: f64,
    /// Principal variances, descending.
    pub variances: [f64; 3],
}

/// The orbit as three moving bodies: PCA-projected knots, the orbit-to-fluid time map and the
/// tidal model that shapes the bodies.
///
/// Invariants (established by [`BodyTrack::new`]): exactly three bodies with `N ≥ 2` finite
/// projected knots each, `0 < T < ∞`, and a finite speed table of `400_001` entries. The table
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
    /// Projected world `x` of every knot, `xs[body][knot]` (structure of arrays).
    xs: [Vec<f64>; BODIES],
    /// Projected world `y` of every knot, `ys[body][knot]`.
    ys: [Vec<f64>; BODIES],
    /// How the knots were projected.
    projection: Projection,
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
            .field("projection", &self.projection)
            .finish_non_exhaustive()
    }
}

impl BodyTrack {
    /// Projects `positions[body][knot]` (exactly 3 bodies, at least 2 knots) onto the canvas of the
    /// given aspect and derives the fluid duration from the median body speed.
    ///
    /// Errors: [`EmberError::DegenerateOrbit`] for a wrong body count, unequal or too short
    /// recordings, non-finite coordinates, a collapsed bounding box, an orbit that does not span a
    /// plane, or a median body speed of zero; [`EmberError::InvalidConfig`] for a non-positive
    /// aspect, fill or reference speed.
    pub(crate) fn new(
        positions: &[Vec<Vector3<f64>>],
        masses: [f64; 3],
        aspect: f64,
        config: &EmberConfig,
    ) -> EmberResult<Self> {
        let fill = config.projection.fill;
        let reference_speed = config.fluid.reference_speed;
        require_positive(aspect, "aspect")?;
        require_positive(fill, "projection.fill")?;
        require_positive(reference_speed, "fluid.reference_speed")?;
        let (bodies, knots) = validate_shape(positions)?;

        let (origin, extent) = bounding_box(bodies)?;
        let normalise = |p: &Vector3<f64>| {
            [(p.x - origin[0]) / extent, (p.y - origin[1]) / extent, (p.z - origin[2]) / extent]
        };
        let count = (BODIES * knots) as f64;

        // Mean and covariance of the normalised points, compensated, in scan order.
        let mut sums = [Neumaier::default(); 3];
        for p in scan(bodies) {
            let q = normalise(p);
            for (sum, value) in sums.iter_mut().zip(q) {
                sum.add(value);
            }
        }
        let mean = sums.map(|sum| sum.total() / count);
        let centred = |p: &Vector3<f64>| {
            let q = normalise(p);
            [q[0] - mean[0], q[1] - mean[1], q[2] - mean[2]]
        };
        let mut products = [Neumaier::default(); 6];
        for p in scan(bodies) {
            let c = centred(p);
            let terms =
                [c[0] * c[0], c[0] * c[1], c[0] * c[2], c[1] * c[1], c[1] * c[2], c[2] * c[2]];
            for (sum, term) in products.iter_mut().zip(terms) {
                sum.add(term);
            }
        }
        let [c00, c01, c02, c11, c12, c22] = products.map(|sum| sum.total() / count);
        let covariance = [[c00, c01, c02], [c01, c11, c12], [c02, c12, c22]];

        let (variances, vectors) = principal_axes(&covariance);
        if !(variances.iter().all(|v| v.is_finite())
            && variances[0] > 0.0
            && variances[1] > variances[0] * PLANE_RATIO)
        {
            return degenerate(format!(
                "the orbit does not span a plane (principal variances {variances:?})"
            ));
        }

        // Hygiene: unit e₀, and e₁ orthogonalised against it (Gram–Schmidt).
        let e0 = normalised(vectors[0]);
        let e1 = normalised(sub(vectors[1], scaled(e0, dot(e0, vectors[1]))));
        let mut axes = [e0, e1];
        orient_by_anchor(bodies, &centred, &mut axes)?;

        // Project, then centre and scale onto the canvas.
        let mut xs: [Vec<f64>; BODIES] = std::array::from_fn(|_| Vec::with_capacity(knots));
        let mut ys: [Vec<f64>; BODIES] = std::array::from_fn(|_| Vec::with_capacity(knots));
        let (mut low, mut high) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for (k, p) in scan(bodies).enumerate() {
            let c = centred(p);
            let projected = [dot(c, axes[0]), dot(c, axes[1])];
            for axis in 0..2 {
                low[axis] = min(low[axis], projected[axis]);
                high[axis] = max(high[axis], projected[axis]);
            }
            let body = k % BODIES;
            xs[body].push(projected[0]);
            ys[body].push(projected[1]);
        }
        let centre = [0.5 * low[0] + 0.5 * high[0], 0.5 * low[1] + 0.5 * high[1]];
        let half = [(high[0] - low[0]) * 0.5, (high[1] - low[1]) * 0.5];
        let scale = fill / max(half[0] / aspect, half[1]);
        if !(scale.is_finite() && scale > 0.0) {
            return degenerate(format!("projection scale {scale} is not finite and positive"));
        }
        for (column, offset) in [(&mut xs, centre[0]), (&mut ys, centre[1])] {
            for values in column.iter_mut() {
                for value in values.iter_mut() {
                    *value = (*value - offset) * scale;
                }
            }
        }

        let tidal = Tidal::new(masses, config)?;
        let mut track = Self {
            duration: 1.0,
            tidal,
            knots,
            last: (knots - 1) as f64,
            xs,
            ys,
            projection: Projection { origin, extent, axes, scale, variances },
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

    /// Projection parameters.
    pub(crate) fn projection(&self) -> &Projection {
        &self.projection
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

/// Checks the recording's shape: exactly three bodies with the same number `N ≥ 2` of knots.
fn validate_shape(
    positions: &[Vec<Vector3<f64>>],
) -> EmberResult<([&[Vector3<f64>]; BODIES], usize)> {
    let [a, b, c] = positions else {
        return degenerate(format!("expected 3 bodies, got {}", positions.len()));
    };
    let knots = a.len();
    if b.len() != knots || c.len() != knots {
        return degenerate(format!(
            "bodies have different recording lengths ({}, {}, {})",
            a.len(),
            b.len(),
            c.len()
        ));
    }
    if knots < 2 {
        return degenerate(format!("{knots} recorded knots; at least 2 are needed"));
    }
    Ok(([a, b, c], knots))
}

/// All `3N` points in scan order: knot-major, then body.
fn scan(bodies: [&[Vector3<f64>]; BODIES]) -> impl Iterator<Item = &Vector3<f64>> {
    let [a, b, c] = bodies;
    a.iter().zip(b).zip(c).flat_map(|((p, q), r)| [p, q, r])
}

/// Centre and largest side of the bounding box of all points; rejects non-finite coordinates and
/// a collapsed box.
fn bounding_box(bodies: [&[Vector3<f64>]; BODIES]) -> EmberResult<([f64; 3], f64)> {
    let (mut low, mut high) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for (index, p) in scan(bodies).enumerate() {
        let p = [p.x, p.y, p.z];
        if !p.iter().all(|v| v.is_finite()) {
            return degenerate(format!(
                "body {} has a non-finite position {p:?} at knot {}",
                index % BODIES,
                index / BODIES
            ));
        }
        for axis in 0..3 {
            low[axis] = min(low[axis], p[axis]);
            high[axis] = max(high[axis], p[axis]);
        }
    }
    let origin = std::array::from_fn(|axis| low[axis] * 0.5 + high[axis] * 0.5);
    let extent = (0..3).map(|axis| high[axis] - low[axis]).fold(0.0, max);
    if !(extent.is_finite() && extent > 0.0) {
        return degenerate(format!("bounding box extent {extent} is not finite and positive"));
    }
    Ok((origin, extent))
}

/// Flips each axis so that its anchor (the first point in scan order whose projection magnitude
/// reaches `max·(1 - 10⁻¹²)`) projects positively.
fn orient_by_anchor(
    bodies: [&[Vector3<f64>]; BODIES],
    centred: &impl Fn(&Vector3<f64>) -> [f64; 3],
    axes: &mut [[f64; 3]; 2],
) -> EmberResult<()> {
    let mut largest = [0.0f64; 2];
    for p in scan(bodies) {
        let c = centred(p);
        for (m, axis) in largest.iter_mut().zip(axes.iter()) {
            *m = max(*m, dot(c, *axis).abs());
        }
    }
    if !(largest[0] > 0.0 && largest[1] > 0.0) {
        return degenerate("no point projects onto a principal axis".into());
    }
    let threshold = largest.map(|m| m * (1.0 - ANCHOR_TOLERANCE));
    let mut anchor: [Option<f64>; 2] = [None, None];
    for p in scan(bodies) {
        let c = centred(p);
        for a in 0..2 {
            if anchor[a].is_none() {
                let v = dot(c, axes[a]);
                if v.abs() >= threshold[a] {
                    anchor[a] = Some(v);
                }
            }
        }
        if anchor.iter().all(Option::is_some) {
            break;
        }
    }
    for (axis, v) in axes.iter_mut().zip(anchor) {
        // The anchor exists: the point attaining `largest` passes its own threshold.
        if v.is_some_and(|v| v < 0.0) {
            *axis = axis.map(|x| -x);
        }
    }
    Ok(())
}

/// Eigen-decomposition of a symmetric 3×3 matrix by cyclic Jacobi rotations.
///
/// Pivots are visited in the fixed order `(0,1), (0,2), (1,2)`. Each rotation annihilates `a_pq`:
/// `θ = (a_qq - a_pp)/(2 a_pq)`, `t = sign(θ)/(|θ| + √(θ²+1))` (the smaller root of
/// `t² + 2θt - 1 = 0`, `sign(0) = +1`), `c = 1/√(t²+1)`, `s = t·c`, and `A ← JᵀAJ`, `V ← VJ` with
/// `J_pp = J_qq = c`, `J_pq = s`, `J_qp = -s`. Sweeps stop once every off-diagonal magnitude is at
/// most `10⁻³⁰⁰` (or after 64 sweeps). For `|θ| > ~10¹⁵⁴`, `θ²` overflows, `t` becomes `0` and
/// `a_pq` (relatively below `10⁻¹⁵⁴` of the diagonal gap) is simply dropped.
///
/// Returns eigenvalues sorted descending (`total_cmp`, ties by original index) and the matching
/// unit eigenvectors.
fn principal_axes(matrix: &[[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = *matrix;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..JACOBI_SWEEPS {
        if a[0][1].abs() <= JACOBI_TOLERANCE
            && a[0][2].abs() <= JACOBI_TOLERANCE
            && a[1][2].abs() <= JACOBI_TOLERANCE
        {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let apq = a[p][q];
            if apq == 0.0 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
            let sign = if theta < 0.0 { -1.0 } else { 1.0 };
            let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            let r = 3 - p - q;
            let (arp, arq) = (a[r][p], a[r][q]);
            let (new_rp, new_rq) = (c * arp - s * arq, s * arp + c * arq);
            a[r][p] = new_rp;
            a[p][r] = new_rp;
            a[r][q] = new_rq;
            a[q][r] = new_rq;
            a[p][p] -= t * apq;
            a[q][q] += t * apq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;
            for row in &mut v {
                let (vp, vq) = (row[p], row[q]);
                row[p] = c * vp - s * vq;
                row[q] = s * vp + c * vq;
            }
        }
    }
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]).then(i.cmp(&j)));
    let values = order.map(|i| a[i][i]);
    let vectors = order.map(|i| [v[0][i], v[1][i], v[2][i]]);
    (values, vectors)
}

/// Neumaier's compensated summation (improved Kahan–Babuška): sequential and exactly
/// reproducible, with an error bound independent of the number of terms.
#[derive(Clone, Copy, Debug, Default)]
struct Neumaier {
    /// Running (rounded) sum.
    sum: f64,
    /// Accumulated rounding errors of `sum`.
    compensation: f64,
}

impl Neumaier {
    /// Adds `x`, capturing the rounding error of the addition exactly.
    fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.compensation += (self.sum - t) + x;
        } else {
            self.compensation += (x - t) + self.sum;
        }
        self.sum = t;
    }

    /// The compensated total.
    fn total(self) -> f64 {
        self.sum + self.compensation
    }
}

/// `a·b` (left to right).
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `a - b`.
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// `s·a`.
fn scaled(a: [f64; 3], s: f64) -> [f64; 3] {
    a.map(|x| x * s)
}

/// `a/|a|`.
fn normalised(a: [f64; 3]) -> [f64; 3] {
    let norm = dot(a, a).sqrt();
    a.map(|x| x / norm)
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

    /// An orthonormal pair `(u, v)` spanning a generically tilted plane, and its normal.
    fn tilted_plane() -> ([f64; 3], [f64; 3], [f64; 3]) {
        let (sa, ca) = math::sin_cos(0.7);
        let (sb, cb) = math::sin_cos(-1.1);
        let u = [ca, sa * cb, sa * sb];
        let raw = [-0.3, 0.8, -0.45];
        let v = normalised(sub(raw, scaled(u, dot(u, raw))));
        let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        (u, v, n)
    }

    /// Three bodies on the ellipse `o + A cos θ u + B sin θ v`, phases `2πb/3`, `N` knots over one
    /// revolution (`θ_k = 2πk/(N-1)`).
    fn ellipse(n: usize, major: f64, minor: f64) -> Vec<Vec<Vector3<f64>>> {
        let (u, v, _) = tilted_plane();
        let o = [120.0, -35.0, 7.5];
        (0..3usize)
            .map(|b| {
                (0..n)
                    .map(|k| {
                        let theta = TAU * k as f64 / (n - 1) as f64 + TAU * b as f64 / 3.0;
                        let (s, c) = math::sin_cos(theta);
                        Vector3::from(std::array::from_fn(|i| {
                            o[i] + major * c * u[i] + minor * s * v[i]
                        }))
                    })
                    .collect()
            })
            .collect()
    }

    fn config() -> EmberConfig {
        EmberConfig::default()
    }

    /// Euclidean length of a 2-vector.
    fn norm(v: [f64; 2]) -> f64 {
        (v[0] * v[0] + v[1] * v[1]).sqrt()
    }

    const ASPECT: f64 = 1.5;

    /// Unequal masses, so the tidal weights are exercised.
    const MASSES: [f64; 3] = [1.0, 1.4, 0.7];

    #[test]
    fn jacobi_diagonalises_symmetric_matrices() {
        let mut rng = Lcg(7);
        for case in 0..200 {
            let mut m = [[0.0; 3]; 3];
            for (i, j) in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)] {
                let x = rng.next() * 2.0 - 1.0;
                m[i][j] = x;
                m[j][i] = x;
            }
            if case % 5 == 0 {
                // Repeated eigenvalue.
                m = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, rng.next()]];
            }
            let (values, vectors) = principal_axes(&m);
            assert!(values[0] >= values[1] && values[1] >= values[2]);
            for (i, e) in vectors.iter().enumerate() {
                for (j, f) in vectors.iter().enumerate() {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!((dot(*e, *f) - expected).abs() < 1e-14, "orthonormal {i} {j}");
                }
                let me = [dot(m[0], *e), dot(m[1], *e), dot(m[2], *e)];
                for k in 0..3 {
                    assert!((me[k] - values[i] * e[k]).abs() < 1e-14, "A e = λ e ({case})");
                }
            }
        }
    }

    #[test]
    fn jacobi_on_a_diagonal_matrix_only_sorts() {
        let (values, vectors) =
            principal_axes(&[[1.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 2.0]]);
        assert_eq!(values, [3.0, 2.0, 1.0]);
        assert_eq!(vectors, [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]);
    }

    #[test]
    fn neumaier_recovers_cancelled_terms() {
        let mut sum = Neumaier::default();
        for x in [1.0, 1e100, 1.0, -1e100] {
            sum.add(x);
        }
        assert_eq!(sum.total(), 2.0);
    }

    #[test]
    fn ellipse_in_a_tilted_plane_recovers_the_plane_axes() {
        let n = 1201;
        let (u, v, _) = tilted_plane();
        let track = BodyTrack::new(&ellipse(n, 30.0, 12.0), MASSES, ASPECT, &config()).unwrap();
        let p = track.projection();
        assert!((dot(p.axes[0], u).abs() - 1.0).abs() < 1e-12, "{:?}", p.axes);
        assert!((dot(p.axes[1], v).abs() - 1.0).abs() < 1e-12, "{:?}", p.axes);
        // Principal variances A²/2 and B²/2 (normalised), nothing out of the plane.
        assert!((p.variances[0] / p.variances[1] - 6.25).abs() < 1e-9);
        assert!(p.variances[2].abs() < 1e-12 * p.variances[0]);
        assert_eq!(p.origin.map(|o| (o * 1e6).round() / 1e6), [120.0, -35.0, 7.5]);
        // The major axis binds: x spans ±fill·aspect, y spans ±fill·B/A·aspect.
        let fill = config().projection.fill;
        let (mut x_max, mut y_max) = (0.0f64, 0.0f64);
        for b in 0..3 {
            for k in 0..n {
                let [x, y] = track.knot_position(b, k);
                x_max = x_max.max(x.abs());
                y_max = y_max.max(y.abs());
            }
        }
        assert!((x_max - fill * ASPECT).abs() < 1e-12, "{x_max}");
        assert!((y_max - fill * ASPECT * 12.0 / 30.0).abs() < 1e-12, "{y_max}");
        assert!((p.scale * 30.0 / p.extent - fill * ASPECT).abs() < 1e-12);
    }

    #[test]
    fn circle_duration_makes_the_median_speed_the_reference_speed() {
        let n = 1201;
        let (_, _, normal) = tilted_plane();
        let mut cfg = config();
        cfg.fluid.reference_speed = 2.5;
        let track = BodyTrack::new(&ellipse(n, 20.0, 20.0), MASSES, ASPECT, &cfg).unwrap();
        // The axes span the orbit's plane (their direction within it is arbitrary for a circle).
        for axis in track.projection().axes {
            assert!(dot(axis, normal).abs() < 1e-12);
        }
        // Every chord has the same length 2ρ sin(π/(N-1)), so the median fraction speed is exact.
        // The three bodies sit at phases 0, 2π/3, 4π/3: their centroid is the circle's centre.
        let at = |b| track.knot_position(b, 17);
        let centre =
            [(at(0)[0] + at(1)[0] + at(2)[0]) / 3.0, (at(0)[1] + at(1)[1] + at(2)[1]) / 3.0];
        let (dx, dy) = (at(1)[0] - centre[0], at(1)[1] - centre[1]);
        let rho = (dx * dx + dy * dy).sqrt();
        let fraction_speed =
            2.0 * rho * math::sin(std::f64::consts::PI / (n - 1) as f64) * (n - 1) as f64;
        let expected = fraction_speed / 2.5;
        assert!((track.duration() - expected).abs() < 1e-9 * expected, "{}", track.duration());
        // Body speed is the reference speed everywhere.
        let bodies = track.bodies_at(0.37 * track.duration());
        for body in bodies {
            assert!((norm(body.velocity) - 2.5).abs() < 1e-9, "{:?}", body.velocity);
        }
    }

    #[test]
    fn anchor_rule_fixes_the_axis_signs() {
        let n = 1201;
        for mirrored in [false, true] {
            let mut positions = ellipse(n, 30.0, 12.0);
            if mirrored {
                // Point reflection through the centre flips every eigenvector sign candidate.
                for body in &mut positions {
                    for p in body.iter_mut() {
                        *p = Vector3::new(240.0, -70.0, 15.0) - *p;
                    }
                }
            }
            let track = BodyTrack::new(&positions, MASSES, ASPECT, &config()).unwrap();
            // Axis 0: the first maximal point is body 0 at knot 0 (θ = 0 or π after reflection).
            assert!(track.knot_position(0, 0)[0] > 1.0);
            // Axis 1: the first maximal point is body 2 at θ = 3π/2, knot (N-1)/12.
            assert!(track.knot_position(2, (n - 1) / 12)[1] > 0.3);
            // Both orientations give the same picture.
            let reference =
                BodyTrack::new(&ellipse(n, 30.0, 12.0), MASSES, ASPECT, &config()).unwrap();
            for b in 0..3 {
                for k in (0..n).step_by(37) {
                    let (p, q) = (track.knot_position(b, k), reference.knot_position(b, k));
                    assert!((p[0] - q[0]).abs() < 1e-12 && (p[1] - q[1]).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn projection_is_bit_reproducible() {
        let a = BodyTrack::new(&ellipse(301, 30.0, 12.0), MASSES, ASPECT, &config()).unwrap();
        let b = BodyTrack::new(&ellipse(301, 30.0, 12.0), MASSES, ASPECT, &config()).unwrap();
        assert_eq!(a.xs, b.xs);
        assert_eq!(a.ys, b.ys);
        assert_eq!(a.duration.to_bits(), b.duration.to_bits());
        assert_eq!(a.projection, b.projection);
    }

    /// A generic (non-symmetric) orbit: three bodies on wobbly, drifting loops.
    fn wobbly(n: usize) -> Vec<Vec<Vector3<f64>>> {
        (0..3usize)
            .map(|b| {
                (0..n)
                    .map(|k| {
                        let s = k as f64 / (n - 1) as f64;
                        let (s1, c1) = math::sin_cos(TAU * (1.0 + b as f64) * s + b as f64);
                        let (s2, c2) = math::sin_cos(TAU * 3.0 * s * s);
                        Vector3::new(
                            3.0 * c1 + 0.4 * s2 + b as f64,
                            2.0 * s1 - 0.3 * c2,
                            0.5 * c1 * s2 + 0.1 * s,
                        )
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn bodies_at_knot_times_sit_on_the_knots() {
        let n = 5001;
        let track = BodyTrack::new(&wobbly(n), MASSES, ASPECT, &config()).unwrap();
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
        let mut track = BodyTrack::new(&wobbly(1001), MASSES, ASPECT, &config()).unwrap();
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
        let track = BodyTrack::new(&wobbly(n), MASSES, ASPECT, &config()).unwrap();
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
        let track = BodyTrack::new(&wobbly(n), MASSES, ASPECT, &config()).unwrap();
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
        let track = BodyTrack::new(&wobbly(333), MASSES, ASPECT, &config()).unwrap();
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
    fn degenerate_orbits_are_rejected() {
        let is_degenerate = |positions: &[Vec<Vector3<f64>>]| {
            matches!(
                BodyTrack::new(positions, MASSES, ASPECT, &config()),
                Err(EmberError::DegenerateOrbit { .. })
            )
        };
        let good = wobbly(50);
        assert!(!is_degenerate(&good));
        // Wrong body count, unequal and too short recordings.
        assert!(is_degenerate(&good[..2]));
        assert!(is_degenerate(&[good.clone(), vec![good[0].clone()]].concat()));
        let mut unequal = good.clone();
        unequal[1].pop();
        assert!(is_degenerate(&unequal));
        let single: Vec<Vec<Vector3<f64>>> = good.iter().map(|b| b[..1].to_vec()).collect();
        assert!(is_degenerate(&single));
        // Non-finite coordinates.
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut positions = good.clone();
            positions[2][17].y = bad;
            assert!(is_degenerate(&positions));
        }
        // Constant orbit: collapsed bounding box.
        let constant = vec![vec![Vector3::new(1.0, 2.0, 3.0); 10]; 3];
        assert!(is_degenerate(&constant));
        // Collinear orbit: no plane.
        let line: Vec<Vec<Vector3<f64>>> = (0..3usize)
            .map(|b| {
                (0..40usize)
                    .map(|k| {
                        let s = math::sin(k as f64 * 0.37 + b as f64);
                        Vector3::new(1.0 + 2.0 * s, -3.0 + s, 0.5 * s)
                    })
                    .collect()
            })
            .collect();
        assert!(is_degenerate(&line));
        // Distinct but motionless bodies: zero median speed.
        let still: Vec<Vec<Vector3<f64>>> = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            .iter()
            .map(|p| vec![Vector3::from(*p); 20])
            .collect();
        assert!(is_degenerate(&still));
    }

    #[test]
    fn invalid_canvas_parameters_are_config_errors() {
        let positions = wobbly(20);
        for aspect in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                BodyTrack::new(&positions, MASSES, aspect, &config()),
                Err(EmberError::InvalidConfig { .. })
            ));
        }
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let masses = [1.0, bad, 1.0];
            assert!(matches!(
                BodyTrack::new(&positions, masses, ASPECT, &config()),
                Err(EmberError::InvalidConfig { parameter, .. }) if parameter == "mass of body 1"
            ));
        }
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
        let track = BodyTrack::new(&wobbly(2001), MASSES, ASPECT, &config()).unwrap();
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
        let track = BodyTrack::new(&wobbly(501), MASSES, ASPECT, &cfg).unwrap();
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
        let track = BodyTrack::new(&positions, MASSES, ASPECT, &config()).unwrap();
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
