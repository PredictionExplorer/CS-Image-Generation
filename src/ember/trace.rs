//! Backward characteristics with gated soak-zone contacts.
//!
//! Ink is a property of water: a parcel carries ink from the last time it brushed past a body
//! where the water was spinning fast enough (the body's own boundary layer). To know a parcel's
//! ink at the end of a frame interval we follow its pathline *backwards* through the interval's
//! velocity snapshots and record, per body, the **latest** time at which it was inside that body's
//! gated soak zone. This ports the museum-lab ledger tracer (`wake3/trace_triton.py`,
//! `wake2/lagdye.py`: `_seg_iv`, `_fetch`, `_fetch1c`) statement for statement;
//! docs/ember-design.md §5.2 is the binding recipe.
//!
//! # Pathlines
//!
//! Snapshots sit at ascending times `τ_0 < … < τ_S`. Over the interval `t0 = τ_{s-1}`,
//! `t1 = τ_s` (`dt = t1 - t0`, `h = -dt`) the velocity is bilinear in space on the periodic
//! fluid grid (`f32` samples, `f64` arithmetic) and linear in time, and one classical RK4 step
//! integrates `dx/dt = u(x, t)` backwards:
//!
//! ```text
//! k1 = U1(x)                        x2 = x + h/2·k1
//! k2 = ½(U1(x2) + U0(x2))           x3 = x + h/2·k2
//! k3 = ½(U1(x3) + U0(x3))           k4 = U0(x + h·k3)
//! xn = x + h/6·(k1 + 2k2 + 2k3 + k4)
//! ```
//!
//! Positions are carried unwrapped (world coordinates); only the lookups wrap.
//!
//! # Soak zone and gate
//!
//! Body `i`'s soak zone is its elliptical outline (semi-axes `a_i, b_i` along its axis, see
//! `orbit::Shape`) grown by the soak depth `d`, taken as the ellipse of semi-axes
//! `a_i + d, b_i + d` (exact for a disc). Its *soak frame* is the symmetric map
//! `F = R·diag(1/(a_i + d), 1/(b_i + d))·Rᵀ` (`R` the rotation onto the body's axes; the
//! reciprocals are computed once per body state), which takes the zone onto the unit circle.
//! Being symmetric, `F` does not depend on the sign of the axis and stays continuous where the
//! body is a disc and its axis is arbitrary, so the frames at the two ends of a step always
//! describe the same orientation. Within a step
//! the parcel moves in body `i`'s soak frame along the straight segment
//! `a = F_1(x - c_i(t1)) → b = F_0(xn - c_i(t0))` between the frames at the step ends,
//! parametrised by `s ∈ [0, 1]` (`s = 0` at `t1`, `s = 1` at `t0`). It is inside the zone
//! `|a + s(b - a)| < 1` on the root interval `[s0, s1]` of `A s² + 2B s + C = 0` with
//! `A = |b - a|²`, `B = a·(b - a)`, `C = |a|² - 1`.
//! The gate admits only fast-spinning water: with `|ω|` linear in `s` between its Catmull-Rom
//! samples at the two step ends (`a1 = |ω(t1, x)|`, `a0 = |ω(t0, xn)|`), `|ω| > ω_c` on an
//! interval `[g0, g1]`. The contact is `[s_lo, s_hi] = [max(s0, g0), min(s1, g1)]`, i.e. the
//! times `[t1 - s_hi·dt, t1 - s_lo·dt]`.
//!
//! # Record
//!
//! Tracing runs backwards, so the first contact found is the latest: body `i` records once,
//! `t*_i = min(t1 - s_lo·dt, t_valve)`, if the contact reaches back to the valve
//! (`t1 - s_hi·dt ≤ t_valve`). Records before the pre-roll (`t*_i < t_on`) are dropped at the end.
//!
//! # Exact fast paths
//!
//! [`Tracer`] implements the eager definition above with shortcuts that provably return the same
//! bits (a test compares it with a literal reference implementation):
//!
//! * the vorticity gate is evaluated only when some zone interval is non-empty (a hit needs
//!   `s1 ≥ s_hi > s_lo ≥ s0`); `ω(t1, x)` is recomputed when it was not carried, which is the same
//!   Catmull-Rom sample at the same point and snapshot;
//! * a step does no contact work once every body has a record, when `t1 - dt > t_valve` (then
//!   `t_lo ≥ t1 - dt` by monotone rounding, so nothing can record), or when `t1 < t_on` (every
//!   record of this and all earlier steps would be `≤ t1 < t_on` and dropped);
//! * a zone test with `disc ≤ 0` (or `A < 10⁻²⁰` and `C ≥ 0`) stops before the square root.
//!
//! A far-from-body pre-filter for the zone test was measured to be worth at most ~9% of the trace
//! and would need a rounding-error proof to stay bit-exact, so every pending body is tested.
//!
//! # Performance
//!
//! Each RK4 stage depends on the previous one through a lookup, so a single pathline is
//! latency-bound. [`Tracer::trace_lanes`] advances several independent parcels in lock step,
//! stage by stage, which lets the out-of-order core overlap their chains (about 2.3× on an
//! Apple M4 with four lanes). Each lane executes exactly the single-parcel operations, so the
//! results do not depend on how parcels are grouped.

use super::fluid::{FluidGrid, Snapshot};
use super::math::{clamp_unit, max, min};
use super::orbit::{BodyState, Shape};

/// Squared step displacement (relative to a body) below which the parcel counts as not moving.
const STILL_SEGMENT: f64 = 1e-20;

/// Denominator of the gate's crossing fraction when `|ω|` is equal at both step ends.
const GATE_EPSILON: f64 = 1e-12;

/// When and where a parcel picks up ink.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ContactRules {
    /// Depth of the soak zone beyond every body's outline.
    pub soak_depth: f64,
    /// Vorticity magnitude gate; `<= 0` disables the gate.
    pub vorticity_gate: f64,
    /// Pre-roll: a body's record (its latest accepted contact in the window) earlier than this
    /// fluid time is dropped.
    pub t_on: f64,
    /// Valve: contacts count only up to this fluid time. A contact entirely after it is ignored;
    /// one straddling it records `t_valve`.
    pub t_valve: f64,
}

/// The flow over one frame interval: `S + 1` snapshots at ascending times `τ_0 < … < τ_S` and the
/// three bodies (centre and outline) at each snapshot time.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FlowWindow<'a> {
    /// The fluid grid all snapshots live on.
    pub grid: FluidGrid,
    /// Snapshots at strictly ascending times: at least one. The pipeline always passes `S ≥ 1`
    /// intervals (docs/ember-design.md §5.1); a single snapshot is an empty window, through which
    /// tracing is the identity.
    pub snapshots: &'a [Snapshot],
    /// `bodies[s][b]` = body `b` at `snapshots[s].time`.
    pub bodies: &'a [[BodyState; 3]],
}

/// Result of tracing one parcel back through a window.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Trace {
    /// World position of the parcel at the window start `τ_0` (unwrapped).
    pub origin: [f64; 2],
    /// Per body, the latest contact time within the window that the rules accept.
    pub contact: [Option<f64>; 3],
}

/// Traces the parcel at `start` (world coordinates, time `τ_S`) back to `τ_0` (the one-shot form
/// of docs/ember-design.md §5.2).
///
/// Test-only convenience wrapper around [`Tracer`]: production code traces many parcels through
/// each window and builds the [`Tracer`] once ([`super::ink::remap`]).
///
/// # Panics
///
/// If the window is malformed (see [`Tracer::new`]).
#[cfg(test)]
pub(crate) fn trace_back(window: &FlowWindow<'_>, rules: &ContactRules, start: [f64; 2]) -> Trace {
    Tracer::new(window, rules).trace(start)
}

/// Catmull-Rom (cardinal, tension ½) weights of the taps `f-1, f, f+1, f+2` at fractional offset
/// `t ∈ [0, 1)` from tap `f`:
///
/// ```text
/// w0 = -½t³ + t² - ½t    w1 = 3/2 t³ - 5/2 t² + 1    w2 = -3/2 t³ + 2t² + ½t    w3 = ½t³ - ½t²
/// ```
///
/// (evaluated left to right, as the prototype's `_cr`). They sum to 1 and reproduce linear data.
#[inline(always)]
pub(crate) fn catmull_rom_weights(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

/// Parameter interval `[s0, s1]` (with `s1 > s0`) on which the segment `a → b` (relative to a
/// disc centre) lies inside the disc of squared radius `r2`, or `None` when that interval is
/// empty. Exactly the prototype's `_seg_iv` (docs/ember-design.md §5.2 step 3), with early exits
/// where its result is an empty interval regardless of the remaining arithmetic.
#[inline(always)]
fn segment_interval(a: [f64; 2], b: [f64; 2], r2: f64) -> Option<(f64, f64)> {
    let d = [b[0] - a[0], b[1] - a[1]];
    let big_a = d[0] * d[0] + d[1] * d[1];
    let big_c = a[0] * a[0] + a[1] * a[1] - r2;
    if big_a < STILL_SEGMENT {
        // The prototype's `still` branch: s0 = 0, s1 = (C < 0 ? 1 : 0).
        return (big_c < 0.0).then_some((0.0, 1.0));
    }
    let big_b = a[0] * d[0] + a[1] * d[1];
    let disc = big_b * big_b - big_a * big_c;
    if disc > 0.0 {
        let root = disc.sqrt();
        // A ≥ 1e-20 here, so the prototype's max(A, 1e-24) is A.
        let s0 = clamp_unit((-big_b - root) / big_a);
        let s1 = clamp_unit((-big_b + root) / big_a);
        (s1 > s0).then_some((s0, s1))
    } else {
        // disc ≤ 0 (or NaN): the prototype sets s1 = s0, an empty interval.
        None
    }
}

/// The gate interval `[g0, g1]` of a step on which `|ω|`, linear in `s` from `|o1|` (`s = 0`,
/// `t1`) to `|o0|` (`s = 1`, `t0`), exceeds `wc` (docs/ember-design.md §5.2 step 2; empty when
/// `g1 ≤ g0`).
#[inline(always)]
fn gate_interval(o1: f64, o0: f64, wc: f64) -> (f64, f64) {
    let a1 = o1.abs();
    let a0 = o0.abs();
    let sc = clamp_unit((wc - a1) / if a0 == a1 { GATE_EPSILON } else { a0 - a1 });
    let g0 = if a1 > wc {
        0.0
    } else if a0 > wc {
        sc
    } else {
        1.0
    };
    let g1 = if a1 > wc {
        if a0 > wc { 1.0 } else { sc }
    } else if a0 > wc {
        1.0
    } else {
        0.0
    };
    (g0, g1)
}

/// Periodic lattice geometry of the fluid grid, for lookups.
#[derive(Clone, Copy, Debug)]
struct Lattice {
    /// World `x` of column 0.
    x0: f64,
    /// World `y` of row 0.
    y0: f64,
    /// `1/dx`.
    inv_dx: f64,
    /// Columns.
    nx: usize,
    /// Rows.
    ny: usize,
}

/// Bilinear interpolation stencil: four node indices and the fractional weights.
#[derive(Clone, Copy, Debug)]
struct Bilinear {
    /// Nodes `(i, j)`, `(i, j+1)`, `(i+1, j)`, `(i+1, j+1)` (periodic), row-major indices.
    nodes: [usize; 4],
    /// Fractional column offset `wx`.
    wx: f64,
    /// `1 - wx`.
    ox: f64,
    /// Fractional row offset `wy`.
    wy: f64,
    /// `1 - wy`.
    oy: f64,
}

impl Bilinear {
    /// `(f00(1-wx) + f01 wx)(1-wy) + (f10(1-wx) + f11 wx) wy` (the prototype's `_fetch`).
    #[inline(always)]
    fn sample(&self, field: &[f32]) -> f64 {
        let [a, b, c, d] = self.nodes.map(|n| f64::from(field[n]));
        (a * self.ox + b * self.wx) * self.oy + (c * self.ox + d * self.wx) * self.wy
    }
}

impl Lattice {
    /// Geometry of `grid`.
    fn new(grid: &FluidGrid) -> Self {
        Self { x0: grid.x0(), y0: grid.y0(), inv_dx: 1.0 / grid.dx, nx: grid.nx, ny: grid.ny }
    }

    /// Continuous grid coordinates `(gx, gy)` of a world point (nodes at integers).
    #[inline(always)]
    fn coordinates(&self, p: [f64; 2]) -> (f64, f64) {
        ((p[0] - self.x0) * self.inv_dx, (p[1] - self.y0) * self.inv_dx)
    }

    /// Periodic indices `(f mod n, (f+1) mod n)` of an integral `f`.
    ///
    /// Total for every `f64`: the cast saturates (`±∞` and `|f| ≥ 2⁶³` become `i64::MIN/MAX`,
    /// NaN becomes 0) and the fast-path test cannot overflow, so a runaway or non-finite position
    /// yields in-range (meaningless) indices instead of a panic. Its sample is then either
    /// non-finite (the weights are NaN) or taken from far outside the flow; either way the traced
    /// origin shows it, and [`super::ink::remap`] counts non-finite origins.
    #[inline(always)]
    fn pair(f: f64, n: usize) -> (usize, usize) {
        let (i, n) = (f as i64, n as i64);
        if (0..n - 1).contains(&i) {
            (i as usize, i as usize + 1)
        } else {
            let i = i.rem_euclid(n);
            (i as usize, if i + 1 == n { 0 } else { i as usize + 1 })
        }
    }

    /// Periodic indices `(f-1, f, f+1, f+2) mod n` of an integral `f` (total like [`Self::pair`]).
    #[inline(always)]
    fn quad(f: f64, n: usize) -> [usize; 4] {
        let (i, n) = (f as i64, n as i64);
        if (1..n - 2).contains(&i) {
            let i = i as usize;
            [i - 1, i, i + 1, i + 2]
        } else {
            [-1, 0, 1, 2].map(|k: i64| i.wrapping_add(k).rem_euclid(n) as usize)
        }
    }

    /// Bilinear stencil at world point `p` (periodic).
    #[inline(always)]
    fn bilinear(&self, p: [f64; 2]) -> Bilinear {
        let (gx, gy) = self.coordinates(p);
        let (fx, fy) = (gx.floor(), gy.floor());
        let (wx, wy) = (gx - fx, gy - fy);
        let (c0, c1) = Self::pair(fx, self.nx);
        let (r0, r1) = Self::pair(fy, self.ny);
        let (r0, r1) = (r0 * self.nx, r1 * self.nx);
        Bilinear { nodes: [r0 + c0, r0 + c1, r1 + c0, r1 + c1], wx, ox: 1.0 - wx, wy, oy: 1.0 - wy }
    }

    /// Periodic Catmull-Rom (bicubic) sample of `field` at world point `p` (the prototype's
    /// `_fetch1c`): rows `Σ_i a_i f[r][c_i]`, then `Σ_j b_j row_j`, left to right.
    #[inline(always)]
    fn catmull_rom(&self, field: &[f32], p: [f64; 2]) -> f64 {
        let (gx, gy) = self.coordinates(p);
        let (fx, fy) = (gx.floor(), gy.floor());
        let a = catmull_rom_weights(gx - fx);
        let b = catmull_rom_weights(gy - fy);
        let cols = Self::quad(fx, self.nx);
        let rows = Self::quad(fy, self.ny);
        let row = |r: usize| {
            let base = r * self.nx;
            f64::from(field[base + cols[0]]) * a[0]
                + f64::from(field[base + cols[1]]) * a[1]
                + f64::from(field[base + cols[2]]) * a[2]
                + f64::from(field[base + cols[3]]) * a[3]
        };
        row(rows[0]) * b[0] + row(rows[1]) * b[1] + row(rows[2]) * b[2] + row(rows[3]) * b[3]
    }
}

/// Maps a world offset from a body's centre into the body's soak frame, where its soak zone (the
/// outline's semi-axes plus the soak depth) is the unit circle: the symmetric matrix
/// `R·diag(i₀, i₁)·Rᵀ` with `i₀ = 1/(a + d)`, `i₁ = 1/(b + d)` and `R` the rotation onto the
/// axis `(c, s)`. It is the same for the axis and its negative, and `i₀·I` for a disc whatever
/// its axis.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SoakFrame {
    /// `c²·i₀ + s²·i₁`.
    xx: f64,
    /// `c·s·(i₀ - i₁)`.
    xy: f64,
    /// `s²·i₀ + c²·i₁`.
    yy: f64,
}

impl SoakFrame {
    /// The soak frame of `shape` with soak depth `soak` (`a + soak > 0` and `b + soak > 0`).
    pub(crate) fn new(shape: &Shape, soak: f64) -> Self {
        let [c, s] = shape.axis;
        let (i0, i1) = (1.0 / (shape.semi[0] + soak), 1.0 / (shape.semi[1] + soak));
        Self {
            xx: (c * c) * i0 + (s * s) * i1,
            xy: (c * s) * (i0 - i1),
            yy: (s * s) * i0 + (c * c) * i1,
        }
    }

    /// The soak-frame coordinates of the world offset `d`.
    #[inline(always)]
    fn apply(&self, d: [f64; 2]) -> [f64; 2] {
        [self.xx * d[0] + self.xy * d[1], self.xy * d[0] + self.yy * d[1]]
    }
}

/// Constants of one backward step over `[τ_{s-1}, τ_s]`.
#[derive(Clone, Copy, Debug)]
struct Step {
    /// Index `s` of the later snapshot.
    later: usize,
    /// `t1 = τ_s`.
    t1: f64,
    /// `dt = t1 - t0 > 0`.
    dt: f64,
    /// `h = -dt`.
    h: f64,
    /// `h/2`.
    half_h: f64,
    /// `h/6`.
    sixth_h: f64,
    /// Body centres at `t1`.
    centres1: [[f64; 2]; 3],
    /// Body centres at `t0`.
    centres0: [[f64; 2]; 3],
    /// Soak frames of the bodies at `t1`.
    frames1: [SoakFrame; 3],
    /// Soak frames of the bodies at `t0`.
    frames0: [SoakFrame; 3],
    /// Whether a contact in this step can still produce a surviving record.
    can_record: bool,
}

/// A flow window prepared for tracing many parcels: validated once, per-step constants
/// precomputed. Tracing is a pure function of the window, the rules and the start point.
#[derive(Debug)]
pub(crate) struct Tracer<'a> {
    /// Fluid-grid geometry.
    lattice: Lattice,
    /// The window's snapshots.
    snapshots: &'a [Snapshot],
    /// Steps in tracing order (latest interval first).
    steps: Vec<Step>,
    /// The vorticity gate `ω_c`, or `None` when disabled.
    gate: Option<f64>,
    /// Pre-roll: records before this time are dropped.
    t_on: f64,
    /// Valve: contacts after this time do not count.
    t_valve: f64,
}

impl<'a> Tracer<'a> {
    /// Prepares `window` for tracing under `rules`.
    ///
    /// # Panics
    ///
    /// If the window has no snapshot, a body row per snapshot is missing, a snapshot's fields do
    /// not match the grid, or the snapshot times are not strictly increasing (all caller bugs).
    pub(crate) fn new(window: &FlowWindow<'a>, rules: &ContactRules) -> Self {
        let snapshots = window.snapshots;
        let n = window.grid.len();
        assert!(!snapshots.is_empty(), "a flow window needs at least one snapshot");
        assert_eq!(window.bodies.len(), snapshots.len(), "one body row per snapshot");
        for snapshot in snapshots {
            assert!(
                snapshot.u.len() == n && snapshot.v.len() == n && snapshot.w.len() == n,
                "snapshot fields must match the fluid grid"
            );
        }
        let steps = (1..snapshots.len())
            .rev()
            .map(|s| {
                let t1 = snapshots[s].time;
                let t0 = snapshots[s - 1].time;
                let dt = t1 - t0;
                assert!(dt > 0.0, "snapshot times must increase strictly ({t0} then {t1})");
                let h = -dt;
                // Every contact of this step has t_lo = t1 - s_hi·dt ≥ t1 - dt (monotone
                // rounding, s_hi ≤ 1): if that exceeds the valve, nothing records.
                let after_valve = t1 - dt > rules.t_valve;
                // Every record of this and all earlier steps is ≤ t1 < t_on: all get dropped.
                let before_pre_roll = t1 < rules.t_on;
                Step {
                    later: s,
                    t1,
                    dt,
                    h,
                    half_h: h / 2.0,
                    sixth_h: h / 6.0,
                    centres1: window.bodies[s].map(|b| b.position),
                    centres0: window.bodies[s - 1].map(|b| b.position),
                    frames1: window.bodies[s].map(|b| SoakFrame::new(&b.shape, rules.soak_depth)),
                    frames0: window.bodies[s - 1]
                        .map(|b| SoakFrame::new(&b.shape, rules.soak_depth)),
                    can_record: !after_valve && !before_pre_roll,
                }
            })
            .collect();
        Self {
            lattice: Lattice::new(&window.grid),
            snapshots,
            steps,
            gate: (rules.vorticity_gate > 0.0).then_some(rules.vorticity_gate),
            t_on: rules.t_on,
            t_valve: rules.t_valve,
        }
    }

    /// One RK4 step (docs/ember-design.md §5.2 step 1) of `L` independent pathlines from `x` at
    /// `t1` back to `t0`. The lanes run in lock step, stage by stage, so the out-of-order core
    /// overlaps their (otherwise serial) lookup chains; every lane performs exactly the
    /// single-parcel arithmetic.
    #[inline(always)]
    fn advect<const L: usize>(&self, step: &Step, x: [[f64; 2]; L]) -> [[f64; 2]; L] {
        let (late, early) = (&self.snapshots[step.later], &self.snapshots[step.later - 1]);
        let (u1, v1, u0, v0) = (&late.u[..], &late.v[..], &early.u[..], &early.v[..]);
        let lattice = &self.lattice;
        let (h, half_h, sixth_h) = (step.h, step.half_h, step.sixth_h);
        let both = |p: &Bilinear| {
            [0.5 * (p.sample(u1) + p.sample(u0)), 0.5 * (p.sample(v1) + p.sample(v0))]
        };

        let p = x.map(|x| lattice.bilinear(x));
        let k1: [[f64; 2]; L] = std::array::from_fn(|i| [p[i].sample(u1), p[i].sample(v1)]);
        let x2: [[f64; 2]; L] =
            std::array::from_fn(|i| [x[i][0] + half_h * k1[i][0], x[i][1] + half_h * k1[i][1]]);
        let p = x2.map(|x| lattice.bilinear(x));
        let k2: [[f64; 2]; L] = std::array::from_fn(|i| both(&p[i]));
        let x3: [[f64; 2]; L] =
            std::array::from_fn(|i| [x[i][0] + half_h * k2[i][0], x[i][1] + half_h * k2[i][1]]);
        let p = x3.map(|x| lattice.bilinear(x));
        let k3: [[f64; 2]; L] = std::array::from_fn(|i| both(&p[i]));
        let x4: [[f64; 2]; L] =
            std::array::from_fn(|i| [x[i][0] + h * k3[i][0], x[i][1] + h * k3[i][1]]);
        let p = x4.map(|x| lattice.bilinear(x));
        let k4: [[f64; 2]; L] = std::array::from_fn(|i| [p[i].sample(u0), p[i].sample(v0)]);
        std::array::from_fn(|i| {
            [
                x[i][0] + sixth_h * (k1[i][0] + 2.0 * k2[i][0] + 2.0 * k3[i][0] + k4[i][0]),
                x[i][1] + sixth_h * (k1[i][1] + 2.0 * k2[i][1] + 2.0 * k3[i][1] + k4[i][1]),
            ]
        })
    }

    /// Dealiased vorticity of snapshot `s` at world point `p` (periodic Catmull-Rom).
    #[inline(always)]
    fn vorticity(&self, s: usize, p: [f64; 2]) -> f64 {
        self.lattice.catmull_rom(&self.snapshots[s].w, p)
    }

    /// Contact work of one parcel over one step `x (t1) → xn (t0)` (docs/ember-design.md §5.2
    /// steps 2–5): updates `record` and returns `ω(t0, xn)` when the gate was evaluated (to be
    /// carried as the next step's `ω(t1, x)`). `carried` is `ω(t1, x)` if known.
    #[inline(always)]
    fn contact(
        &self,
        step: &Step,
        x: [f64; 2],
        xn: [f64; 2],
        carried: Option<f64>,
        record: &mut [Option<f64>; 3],
    ) -> Option<f64> {
        if !step.can_record || record.iter().all(Option::is_some) {
            return None;
        }
        let mut intervals: [Option<(f64, f64)>; 3] = [None; 3];
        for (body, interval) in intervals.iter_mut().enumerate() {
            if record[body].is_none() {
                let (c1, c0) = (step.centres1[body], step.centres0[body]);
                *interval = segment_interval(
                    step.frames1[body].apply([x[0] - c1[0], x[1] - c1[1]]),
                    step.frames0[body].apply([xn[0] - c0[0], xn[1] - c0[1]]),
                    1.0,
                );
            }
        }
        if intervals.iter().all(Option::is_none) {
            return None;
        }
        let mut next_carried = None;
        let (g0, g1) = match self.gate {
            None => (0.0, 1.0),
            Some(wc) => {
                let o1 = match carried {
                    Some(o1) => o1,
                    None => self.vorticity(step.later, x),
                };
                let o0 = self.vorticity(step.later - 1, xn);
                next_carried = Some(o0);
                gate_interval(o1, o0, wc)
            }
        };
        for (record, interval) in record.iter_mut().zip(intervals) {
            let Some((s0, s1)) = interval else { continue };
            let s_lo = max(s0, g0);
            let s_hi = min(s1, g1);
            if s_hi > s_lo {
                let t_lo = step.t1 - s_hi * step.dt;
                let t_hi = step.t1 - s_lo * step.dt;
                if t_lo <= self.t_valve {
                    *record = Some(min(t_hi, self.t_valve));
                }
            }
        }
        next_carried
    }

    /// Traces the parcel at `start` (world coordinates, time `τ_S`) back to `τ_0`.
    pub(crate) fn trace(&self, start: [f64; 2]) -> Trace {
        let [trace] = self.trace_lanes([start]);
        trace
    }

    /// Traces `L` parcels at once; lane `i` returns exactly `self.trace(starts[i])`.
    ///
    /// Never panics on bad numbers: a lane whose arithmetic meets a non-finite start or velocity
    /// sample (NaN poisons every later RK4 stage, `±∞` turns into NaN at the next lookup) returns
    /// a non-finite `origin`, which is how callers detect a broken flow (see
    /// [`super::ink::RemapStats::non_finite_origins`]). Lanes never influence each other.
    pub(crate) fn trace_lanes<const L: usize>(&self, starts: [[f64; 2]; L]) -> [Trace; L] {
        let mut x = starts;
        let mut contact = [[None; 3]; L];
        // Per lane, ω at the current position and the current step's later snapshot, when known.
        let mut carried: [Option<f64>; L] = [None; L];
        for step in &self.steps {
            let xn = self.advect(step, x);
            for lane in 0..L {
                carried[lane] =
                    self.contact(step, x[lane], xn[lane], carried[lane], &mut contact[lane]);
            }
            x = xn;
        }
        std::array::from_fn(|lane| {
            let mut contact = contact[lane];
            for record in &mut contact {
                if record.is_some_and(|t| t < self.t_on) {
                    *record = None;
                }
            }
            Trace { origin: x[lane], contact }
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::math;
    use super::*;

    /// Deterministic pseudo-random numbers in `[0, 1)` (64-bit LCG, top 53 bits).
    pub(crate) struct Lcg(pub u64);

    impl Lcg {
        pub(crate) fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        /// Uniform in `[lo, hi)`.
        pub(crate) fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.next()
        }
    }

    /// A `nx × ny` box with spacing `dx`.
    pub(crate) fn grid(nx: usize, ny: usize, dx: f64) -> FluidGrid {
        FluidGrid { nx, ny, dx, lx: nx as f64 * dx, ly: ny as f64 * dx }
    }

    /// A snapshot at `time` whose `(u, v, ω)` at each node `(x, y)` is `field(x, y)`.
    pub(crate) fn snapshot(
        grid: &FluidGrid,
        time: f64,
        mut field: impl FnMut(f64, f64) -> [f64; 3],
    ) -> Snapshot {
        let mut out = Snapshot::zeros(grid);
        out.time = time;
        for i in 0..grid.ny {
            for j in 0..grid.nx {
                let [u, v, w] =
                    field(grid.x0() + j as f64 * grid.dx, grid.y0() + i as f64 * grid.dx);
                let n = i * grid.nx + j;
                (out.u[n], out.v[n], out.w[n]) = (u as f32, v as f32, w as f32);
            }
        }
        out
    }

    /// Rules with the gate disabled, the valve open and no pre-roll. With point bodies
    /// ([`points`]) the soak zone of every body is the disc of radius `reach` around its centre.
    fn open_rules(reach: f64) -> ContactRules {
        ContactRules {
            soak_depth: reach,
            vorticity_gate: 0.0,
            t_on: f64::NEG_INFINITY,
            t_valve: f64::INFINITY,
        }
    }

    /// Point bodies (outlines of zero size) at these centres: their soak zone is the disc of
    /// radius `soak_depth`, the disc test of the prototype.
    pub(crate) fn points(centres: &[[[f64; 2]; 3]]) -> Vec<[BodyState; 3]> {
        let point = Shape { semi: [0.0, 0.0], axis: [1.0, 0.0], spin: 0.0, strain: 0.0 };
        centres
            .iter()
            .map(|row| row.map(|position| BodyState { position, velocity: [0.0; 2], shape: point }))
            .collect()
    }

    /// Bodies parked far outside the region under test.
    const FAR: [[f64; 2]; 3] = [[50.0, 50.0], [-50.0, 50.0], [50.0, -50.0]];

    #[test]
    fn catmull_rom_weights_partition_unity_and_interpolate() {
        assert_eq!(catmull_rom_weights(0.0), [0.0, 1.0, 0.0, 0.0]);
        for k in 0..=64u32 {
            let t = f64::from(k) / 64.0;
            let w = catmull_rom_weights(t);
            assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-15);
            // Reproduces linear data f(i) = i at offset t from tap 0 (taps -1, 0, 1, 2).
            let linear = -w[0] + w[2] + 2.0 * w[3];
            assert!((linear - t).abs() < 1e-15);
        }
    }

    /// The soak frame takes the soak zone onto the unit circle, does not depend on the sign of
    /// the axis (bit for bit), and for a disc does not depend on the axis at all.
    #[test]
    fn the_soak_frame_is_symmetric_and_blind_to_the_axis_sign() {
        let soak = 0.03;
        let (sn, c) = math::sin_cos(0.83);
        let shape = Shape { semi: [0.07, 0.03], axis: [c, sn], spin: 0.0, strain: 0.0 };
        let frame = SoakFrame::new(&shape, soak);
        let norm = |v: [f64; 2]| (v[0] * v[0] + v[1] * v[1]).sqrt();
        // The zone's ends along and across the axis land on the unit circle.
        let along = frame.apply([0.10 * c, 0.10 * sn]);
        let across = frame.apply([-0.06 * sn, 0.06 * c]);
        assert!((norm(along) - 1.0).abs() < 1e-14 && (norm(across) - 1.0).abs() < 1e-14);
        // Symmetric: the image of the outline normal stays parallel to it.
        assert!((along[0] * sn - along[1] * c).abs() < 1e-14);

        let flipped = SoakFrame::new(&Shape { axis: [-c, -sn], ..shape }, soak);
        for (a, b) in [(frame.xx, flipped.xx), (frame.xy, flipped.xy), (frame.yy, flipped.yy)] {
            assert_eq!(a.to_bits(), b.to_bits());
        }

        let disc = |angle: f64| {
            let (sn, c) = math::sin_cos(angle);
            SoakFrame::new(
                &Shape { semi: [0.05, 0.05], axis: [c, sn], spin: 0.0, strain: 0.0 },
                soak,
            )
        };
        for angle in [0.0, 0.4, 1.9, -2.7] {
            let frame = disc(angle);
            assert_eq!(frame.xy, 0.0);
            assert!((frame.xx - 12.5).abs() < 1e-13 && (frame.yy - 12.5).abs() < 1e-13);
        }
    }

    /// Regression: the tidal axis is an eigenvector, whose sign can flip between two snapshots.
    /// With a frame that was odd in the axis, the parcel's path in the soak frame then ran
    /// through the body's centre and every gated parcel on the canvas recorded a contact.
    #[test]
    fn an_axis_sign_flip_between_snapshots_changes_no_trace() {
        let g = grid(24, 16, 0.125);
        let times = [1.0, 1.02, 1.04];
        let snaps: Vec<Snapshot> =
            times.iter().map(|&t| snapshot(&g, t, |_, _| [0.3, -0.2, 100.0])).collect();
        let (sn, c) = math::sin_cos(2.3);
        let body = |flip: f64| BodyState {
            position: [0.2, -0.1],
            velocity: [0.0, 0.0],
            shape: Shape {
                semi: [0.09, 0.04],
                axis: [flip * c, flip * sn],
                spin: 0.0,
                strain: 0.0,
            },
        };
        let far = BodyState { position: [1e6, 1e6], velocity: [0.0; 2], shape: Shape::disc(0.05) };
        let bodies = |flips: [f64; 3]| -> Vec<[BodyState; 3]> {
            flips.iter().map(|&flip| [body(flip), far, far]).collect()
        };
        let rules =
            ContactRules { soak_depth: 0.03, vorticity_gate: 40.0, t_on: 0.0, t_valve: 9.0 };
        let (steady, flipping) = (bodies([1.0, 1.0, 1.0]), bodies([1.0, -1.0, 1.0]));
        let trace = |bodies: &[[BodyState; 3]], start| {
            let window = FlowWindow { grid: g, snapshots: &snaps, bodies };
            let fast = Tracer::new(&window, &rules).trace(start);
            let reference = reference_trace(&window, &rules, start);
            assert_eq!(
                fast.contact.map(|t| t.map(f64::to_bits)),
                reference.contact.map(|t| t.map(f64::to_bits))
            );
            fast
        };
        let mut contacts = 0;
        for i in 0..40 {
            for j in 0..30 {
                let start = [-1.2 + 0.06 * f64::from(i), -0.9 + 0.06 * f64::from(j)];
                let (a, b) = (trace(&steady, start), trace(&flipping, start));
                assert_eq!(a.origin.map(f64::to_bits), b.origin.map(f64::to_bits));
                assert_eq!(
                    a.contact.map(|t| t.map(f64::to_bits)),
                    b.contact.map(|t| t.map(f64::to_bits))
                );
                contacts += usize::from(a.contact[0].is_some());
            }
        }
        // Only parcels that pass the body's soak zone are inked: a small part of the canvas.
        assert!(contacts > 0 && contacts < 120, "{contacts} of 1200 parcels in contact");
    }

    #[test]
    fn segment_interval_solves_the_disc_crossing() {
        // Straight through the centre: enters at s = 0.4, leaves at s = 0.6.
        let (s0, s1) = segment_interval([-0.5, 0.0], [0.5, 0.0], 0.01).unwrap();
        assert!((s0 - 0.4).abs() < 1e-15 && (s1 - 0.6).abs() < 1e-15);
        // Starts inside: clamped at 0.
        let (s0, s1) = segment_interval([0.05, 0.0], [1.05, 0.0], 0.01).unwrap();
        assert!(s0 == 0.0 && (s1 - 0.05).abs() < 1e-15);
        // Misses.
        assert!(segment_interval([-0.5, 0.2], [0.5, 0.2], 0.01).is_none());
        // Line hits the disc beyond the segment.
        assert!(segment_interval([0.5, 0.0], [1.0, 0.0], 0.01).is_none());
        // Not moving: inside or outside.
        assert_eq!(segment_interval([0.01, 0.0], [0.01, 0.0], 0.01), Some((0.0, 1.0)));
        assert!(segment_interval([0.2, 0.0], [0.2, 0.0], 0.01).is_none());
    }

    #[test]
    fn gate_interval_cases() {
        assert_eq!(gate_interval(50.0, 60.0, 40.0), (0.0, 1.0));
        assert_eq!(gate_interval(-50.0, 20.0, 40.0), (0.0, (40.0 - 50.0) / (20.0 - 50.0)));
        assert_eq!(gate_interval(20.0, -60.0, 40.0), ((40.0 - 20.0) / (60.0 - 20.0), 1.0));
        let (g0, g1) = gate_interval(20.0, 30.0, 40.0);
        assert!(g1 <= g0);
        let (g0, g1) = gate_interval(40.0, 40.0, 40.0);
        assert!(g1 <= g0);
    }

    #[test]
    fn uniform_flow_translates_parcels() {
        let g = grid(40, 30, 0.1);
        let (u, v) = (0.7, -0.4);
        let times = [0.0, 0.01, 0.025, 0.03];
        let snaps: Vec<Snapshot> =
            times.iter().map(|&t| snapshot(&g, t, |_, _| [u, v, 0.0])).collect();
        let bodies = vec![FAR; times.len()];
        let outlines = points(&bodies);
        let window = FlowWindow { grid: g, snapshots: &snaps, bodies: &outlines };
        for start in [[0.3, 0.2], [-1.9, 1.4], [1.95, -1.45], [7.3, -5.1]] {
            let trace = trace_back(&window, &open_rules(0.1), start);
            // RK4 is exact for a steady uniform flow; the f32 samples round u and v.
            let (uf, vf) = (f64::from(u as f32), f64::from(v as f32));
            assert!((trace.origin[0] - (start[0] - uf * 0.03)).abs() < 1e-14, "{trace:?}");
            assert!((trace.origin[1] - (start[1] - vf * 0.03)).abs() < 1e-14, "{trace:?}");
            assert_eq!(trace.contact, [None; 3]);
        }
    }

    #[test]
    fn solid_body_rotation_rotates_parcels_back() {
        let g = grid(64, 48, 0.05);
        let omega = 2.0;
        let times: Vec<f64> = (0..=4u32).map(|s| 0.005 * f64::from(s)).collect();
        let snaps: Vec<Snapshot> =
            times.iter().map(|&t| snapshot(&g, t, |x, y| [-omega * y, omega * x, 0.0])).collect();
        let bodies = vec![FAR; times.len()];
        let outlines = points(&bodies);
        let window = FlowWindow { grid: g, snapshots: &snaps, bodies: &outlines };
        let (s, c) = math::sin_cos(-omega * 0.02);
        let mut rng = Lcg(3);
        for _ in 0..200 {
            let p = [rng.range(-1.0, 1.0), rng.range(-0.8, 0.8)];
            let trace = trace_back(&window, &open_rules(0.1), p);
            let expected = [c * p[0] - s * p[1], s * p[0] + c * p[1]];
            // Linear fields are interpolated exactly; RK4's local error is O((Ωh)⁵); the f32
            // node values carry ~6e-8 relative error.
            for (got, want) in trace.origin.iter().zip(expected) {
                assert!((got - want).abs() < 1e-8, "{trace:?} {p:?}");
            }
        }
    }

    /// Still water, a disc sweeping from `(-0.5, 0)` to `(0.5, 0)` over `[0, 0.04]` in 4 steps.
    fn sweeping_disc(grid: &FluidGrid) -> (Vec<Snapshot>, Vec<[[f64; 2]; 3]>) {
        let times: Vec<f64> = (0..=4u32).map(|s| 0.01 * f64::from(s)).collect();
        let snaps = times.iter().map(|&t| snapshot(grid, t, |_, _| [0.0; 3])).collect();
        let bodies = times.iter().map(|&t| [[-0.5 + 25.0 * t, 0.0], FAR[1], FAR[2]]).collect();
        (snaps, bodies)
    }

    /// Latest time in `[0, 0.04]` at which the sweeping disc of radius `r` covers `p`.
    fn sweep_exit(p: [f64; 2], r: f64) -> Option<(f64, f64)> {
        if p[1].abs() >= r {
            return None;
        }
        let half = (r * r - p[1] * p[1]).sqrt();
        let (enter, exit) = ((p[0] - half + 0.5) / 25.0, (p[0] + half + 0.5) / 25.0);
        (enter <= 0.04 && exit >= 0.0).then(|| (enter.max(0.0), exit.min(0.04)))
    }

    #[test]
    fn disc_in_still_water_inks_its_swept_band_with_the_exit_time() {
        let g = grid(40, 30, 0.1);
        let (snaps, bodies) = sweeping_disc(&g);
        let outlines = points(&bodies);
        let window = FlowWindow { grid: g, snapshots: &snaps, bodies: &outlines };
        let r = 0.08;
        let tracer = Tracer::new(&window, &open_rules(r));
        let mut inked = 0;
        for i in 0..=120u32 {
            for j in 0..=40u32 {
                let p = [-0.7 + 0.0117 * f64::from(i), -0.1 + 0.005 * f64::from(j)];
                let trace = tracer.trace(p);
                assert_eq!(trace.origin, p, "still water");
                assert_eq!(trace.contact[1..], [None, None]);
                // Skip points within rounding of the band's boundary.
                let boundary = (p[1].abs() - r).abs() < 1e-9
                    || [-0.5 - r, 0.5 + r].iter().any(|e| (p[0] - e).abs() < 1e-9);
                if boundary {
                    continue;
                }
                match (sweep_exit(p, r), trace.contact[0]) {
                    (None, None) => {}
                    (Some((_, exit)), Some(t)) => {
                        inked += 1;
                        assert!((t - exit).abs() < 1e-12, "{p:?}: {t} vs {exit}");
                    }
                    (expected, got) => panic!("{p:?}: expected {expected:?}, got {got:?}"),
                }
            }
        }
        assert!(inked > 500, "{inked}");
    }

    #[test]
    fn valve_clips_and_pre_roll_drops_records() {
        let g = grid(40, 30, 0.1);
        let (snaps, bodies) = sweeping_disc(&g);
        let outlines = points(&bodies);
        let window = FlowWindow { grid: g, snapshots: &snaps, bodies: &outlines };
        let r = 0.08;
        let rules = ContactRules { t_valve: 0.025, t_on: 0.012, ..open_rules(r) };
        let tracer = Tracer::new(&window, &rules);
        let mut seen = [0; 4];
        for i in 0..=200u32 {
            let p = [-0.65 + 0.0065 * f64::from(i), 0.03];
            let got = tracer.trace(p).contact[0];
            let Some((enter, exit)) = sweep_exit(p, r) else {
                assert_eq!(got, None);
                continue;
            };
            if (enter - 0.025).abs() < 1e-9 || (exit - 0.012).abs() < 1e-9 {
                continue;
            }
            if enter > 0.025 {
                // Contact only after the valve closed: nothing counts.
                assert_eq!(got, None, "{p:?}");
                seen[0] += 1;
            } else if exit < 0.012 {
                // Contact only during the pre-roll: dropped.
                assert_eq!(got, None, "{p:?}");
                seen[1] += 1;
            } else if exit > 0.025 {
                // Straddles the valve: clipped to it.
                assert_eq!(got, Some(0.025), "{p:?}");
                seen[2] += 1;
            } else {
                assert!((got.unwrap() - exit).abs() < 1e-12, "{p:?}");
                seen[3] += 1;
            }
        }
        assert!(seen.iter().all(|&n| n > 5), "{seen:?}");
    }

    /// Still water with spatially uniform vorticity `ω(τ)` and a disc centred on the origin that
    /// moves from `from` to `to` over `[0, 1]` in four steps.
    fn gated(
        omega: impl Fn(f64) -> f64,
        from: f64,
        to: f64,
    ) -> (Vec<Snapshot>, Vec<[[f64; 2]; 3]>) {
        let g = grid(40, 30, 0.1);
        let times = [0.0, 0.25, 0.5, 0.75, 1.0];
        let snaps = times.iter().map(|&t| snapshot(&g, t, |_, _| [0.0, 0.0, omega(t)])).collect();
        let bodies =
            times.iter().map(|&t| [[from + (to - from) * t, 0.0], FAR[1], FAR[2]]).collect();
        (snaps, bodies)
    }

    #[test]
    fn gated_contact_starts_at_the_analytic_vorticity_crossing() {
        let g = grid(40, 30, 0.1);
        let rules = ContactRules { vorticity_gate: 40.0, ..open_rules(0.1) };
        let latest = |omega: &dyn Fn(f64) -> f64, from: f64, to: f64, rules: &ContactRules| {
            let (snaps, bodies) = gated(omega, from, to);
            let outlines = points(&bodies);
            let window = FlowWindow { grid: g, snapshots: &snaps, bodies: &outlines };
            trace_back(&window, rules, [0.0, 0.0]).contact[0]
        };
        // |ω| = 100(1 - τ) > 40 for τ < 0.6, either sign; the body sits on the parcel.
        let decaying = |t: f64| 100.0 * (1.0 - t);
        let t = latest(&decaying, 0.0, 0.0, &rules).unwrap();
        assert!((t - 0.6).abs() < 1e-12, "{t}");
        let t = latest(&|t| -decaying(t), 0.0, 0.0, &rules).unwrap();
        assert!((t - 0.6).abs() < 1e-12, "{t}");
        // The disc (reach 0.1, moving from -0.4 to 0.4) covers the parcel for τ ∈ [0.375, 0.625]:
        // the gated contact is the intersection with τ < 0.6, latest 0.6.
        let t = latest(&decaying, -0.4, 0.4, &rules).unwrap();
        assert!((t - 0.6).abs() < 1e-12, "{t}");
        // Disc contact τ ∈ [0, 0.1/0.24] ends before the gate closes: latest 0.41666….
        let t = latest(&decaying, 0.0, 0.24, &rules).unwrap();
        assert!((t - 0.1 / 0.24).abs() < 1e-12, "{t}");
        // |ω| = 100τ: open for τ > 0.4, so the latest contact is the window end or the valve.
        let growing = |t: f64| 100.0 * t;
        assert_eq!(latest(&growing, 0.0, 0.0, &rules), Some(1.0));
        let valve = ContactRules { t_valve: 0.9, ..rules };
        assert_eq!(latest(&growing, 0.0, 0.0, &valve), Some(0.9));
        let early_valve = ContactRules { t_valve: 0.3, ..rules };
        assert_eq!(latest(&growing, 0.0, 0.0, &early_valve), None);
        // Below the gate everywhere: no contact; gate disabled: contact at the end.
        assert_eq!(latest(&|_| 39.0, 0.0, 0.0, &rules), None);
        assert_eq!(latest(&|_| 39.0, 0.0, 0.0, &open_rules(0.1)), Some(1.0));
    }

    /// The prototype's analytic moving-disc oracle (`test_trace_torch.py`, gated latest contact),
    /// with every piece of the contact rule active at once: parcels drift with a uniform flow
    /// `U` past a disc moving with velocity `V` (so `x ≠ xn` and both disc ends move in every
    /// step), the gate is a spatial half-plane, and the valve and the pre-roll cut the window.
    ///
    /// With `τ = t - 1 ∈ [-1, 0]`, the parcel that sits at `p` at `t = 1` is at `p + Uτ` and the
    /// disc centre at `b + Vτ`, so the parcel is inside the soak zone on the open root interval of
    /// `|p - b + (U - V)τ|² = r²`. The vorticity `ω = 20(x - x_g ± 2)` is linear in `x`, hence
    /// exact under Catmull-Rom (the pathlines stay far from the periodic seam), linear in `τ`
    /// along a pathline, and exact in `f32` at the nodes; `|ω| > 40` is the half-plane `x > x_g`
    /// (`+`) or `x < x_g` (`-`), i.e. `τ > (x_g - p_x)/U_x` or `τ < (x_g - p_x)/U_x`. RK4 is exact
    /// for the steady uniform flow, so the tracer must reproduce the analytic latest contact
    /// `min(1 + hi, t_valve)` of the interval `[lo, hi]` (disc ∩ gate ∩ window) whenever
    /// `1 + lo ≤ t_valve`, dropped when it is earlier than `t_on`.
    #[test]
    fn moving_disc_in_a_drift_with_a_spatial_gate_is_analytic() {
        let g = grid(96, 64, 0.0625); // [-3, 3) × [-2, 2)
        let flow = [0.5, 0.25];
        let times: Vec<f64> = (0..=32u32).map(|s| f64::from(s) / 32.0).collect();
        let (reach, t_on, t_valve) = (0.3, 0.15, 0.8);
        let rules = ContactRules { soak_depth: reach, vorticity_gate: 40.0, t_on, t_valve };
        let mut rng = Lcg(77);
        // (gate open for x > x_g?, x_g, body slot, disc position at t = 1, disc velocity). The
        // gate lines cut through the swept regions, so the gate decides many parcels.
        let cases = [
            (true, 0.25, 0, [0.1, -0.05], [-0.75, 0.5]),
            (false, 0.75, 1, [0.1, -0.05], [-0.75, 0.5]),
            (true, -0.875, 2, [-0.2, 0.3], [0.9, -0.35]),
            (false, -0.375, 0, [-0.2, 0.3], [0.9, -0.35]),
        ];
        let mut total = [0usize; 5];
        let mut worst = 0.0f64;
        for (case, &(open_right, x_g, slot, disc, velocity)) in cases.iter().enumerate() {
            let offset = if open_right { 2.0 } else { -2.0 };
            let snaps: Vec<Snapshot> = times
                .iter()
                .map(|&t| snapshot(&g, t, |x, _| [flow[0], flow[1], 20.0 * (x - x_g + offset)]))
                .collect();
            let bodies: Vec<[[f64; 2]; 3]> = times
                .iter()
                .map(|&t| {
                    let mut row = FAR;
                    row[slot] =
                        [disc[0] + velocity[0] * (t - 1.0), disc[1] + velocity[1] * (t - 1.0)];
                    row
                })
                .collect();
            let outlines = points(&bodies);
            let window = FlowWindow { grid: g, snapshots: &snaps, bodies: &outlines };
            let tracer = Tracer::new(&window, &rules);
            let w = [flow[0] - velocity[0], flow[1] - velocity[1]];
            // [miss, plain record, clipped by the valve, dropped by the pre-roll, gate decided]
            let mut seen = [0usize; 5];
            for _ in 0..5000 {
                let p = [rng.range(-1.2, 1.8), rng.range(-1.0, 1.3)];
                // Disc: A τ² + 2B τ + C < 0 (empty unless the discriminant is positive).
                let r0 = [p[0] - disc[0], p[1] - disc[1]];
                let a = w[0] * w[0] + w[1] * w[1];
                let b = r0[0] * w[0] + r0[1] * w[1];
                let c = r0[0] * r0[0] + r0[1] * r0[1] - reach * reach;
                let discriminant = b * b - a * c;
                let (enter, exit) = if discriminant > 0.0 {
                    let root = discriminant.sqrt();
                    ((-b - root) / a, (-b + root) / a)
                } else {
                    (1.0, -1.0)
                };
                let crossing = (x_g - p[0]) / flow[0];
                let (gate_lo, gate_hi) =
                    if open_right { (crossing, 0.0) } else { (-1.0, crossing) };
                let latest = |lo: f64, hi: f64| {
                    (hi > lo && 1.0 + lo <= t_valve).then(|| (1.0 + hi).min(t_valve))
                };
                let (lo, hi) = (enter.max(gate_lo).max(-1.0), exit.min(gate_hi).min(0.0));
                let record = latest(lo, hi);
                // Skip configurations within rounding of a decision boundary.
                if (hi - lo).abs() < 1e-9
                    || (1.0 + lo - t_valve).abs() < 1e-9
                    || record.is_some_and(|t| (t - t_on).abs() < 1e-9)
                {
                    continue;
                }
                let got = tracer.trace(p);
                for (axis, origin) in got.origin.iter().enumerate() {
                    assert!((origin - (p[axis] - flow[axis])).abs() < 1e-13, "{p:?}: {got:?}");
                }
                for (body, contact) in got.contact.iter().enumerate() {
                    assert!(body == slot || contact.is_none(), "{p:?}: {got:?}");
                }
                let expected = record.filter(|&t| t >= t_on);
                match (expected, got.contact[slot]) {
                    (None, None) => seen[if record.is_some() { 3 } else { 0 }] += 1,
                    (Some(want), Some(t)) => {
                        worst = worst.max((t - want).abs());
                        seen[if want == t_valve { 2 } else { 1 }] += 1;
                    }
                    (want, got) => panic!("case {case} {p:?}: expected {want:?}, got {got:?}"),
                }
                // Without the gate the answer would differ: the gate decided this parcel.
                let ungated = latest(enter.max(-1.0), exit.min(0.0)).filter(|&t| t >= t_on);
                seen[4] += usize::from(ungated != expected);
            }
            assert!(seen[1] >= 100 && seen[4] >= 50, "case {case}: {seen:?}");
            for (sum, n) in total.iter_mut().zip(seen) {
                *sum += n;
            }
        }
        // Every branch of the rule is exercised, and the latest contacts are exact.
        assert!(total.iter().all(|&n| n >= 50), "{total:?}");
        assert!(worst < 1e-12, "{worst:e}");
    }

    #[test]
    fn runaway_and_non_finite_positions_never_panic() {
        let g = grid(8, 6, 0.25);
        let snaps: Vec<Snapshot> =
            [0.0, 0.1, 0.2].iter().map(|&t| snapshot(&g, t, |_, _| [0.0, 0.0, 50.0])).collect();
        let bodies = vec![[[0.0, 0.0]; 3]; 3];
        let outlines = points(&bodies);
        let window = FlowWindow { grid: g, snapshots: &snaps, bodies: &outlines };
        let rules = ContactRules { vorticity_gate: 40.0, ..open_rules(0.5) };
        let tracer = Tracer::new(&window, &rules);
        // Far beyond i64 in grid units, in still water: the lookups wrap to some node, the parcel
        // stays where it is, and no body is anywhere near.
        for start in [[1e300, 0.0], [-1e300, 0.3], [0.1, 1e19], [-9.3e18, -4.2e18], [0.0, -1e300]] {
            let trace = tracer.trace(start);
            assert_eq!(trace, Trace { origin: start, contact: [None; 3] }, "{start:?}");
        }
        for start in [[f64::NAN, 0.0], [f64::INFINITY, 0.0], [0.0, f64::NEG_INFINITY]] {
            let [trace, healthy] = tracer.trace_lanes([start, [0.1, 0.2]]);
            assert!(!trace.origin.iter().all(|v| v.is_finite()), "{start:?}: {trace:?}");
            // The neighbouring lane is unaffected (and contacts the bodies parked at the origin).
            assert_eq!(healthy, tracer.trace([0.1, 0.2]));
            assert_eq!(healthy.contact, [Some(0.2); 3]);
        }
        // A non-finite velocity sample poisons exactly the pathlines that read it.
        let mut snaps = snaps;
        snaps[1].u[2 * g.nx + 5] = f32::INFINITY;
        let outlines = points(&bodies);
        let window = FlowWindow { grid: g, snapshots: &snaps, bodies: &outlines };
        let tracer = Tracer::new(&window, &rules);
        let node = |i: usize, j: usize| [g.x0() + j as f64 * g.dx, g.y0() + i as f64 * g.dx];
        assert!(!tracer.trace(node(2, 5)).origin.iter().all(|v| v.is_finite()));
        assert_eq!(tracer.trace(node(4, 1)).origin, node(4, 1));
    }

    #[test]
    fn lookups_wrap_periodically() {
        let g = grid(24, 16, 0.125);
        let lattice = Lattice::new(&g);
        let tau = 2.0 * std::f64::consts::PI;
        let snap = snapshot(&g, 0.0, |x, y| {
            let (sx, cx) = math::sin_cos(tau * x / g.lx);
            let (sy, cy) = math::sin_cos(tau * 2.0 * y / g.ly);
            [sx + cy, cx * sy, cx + sx * cy]
        });
        let mut rng = Lcg(5);
        for _ in 0..500 {
            let p = [rng.range(-2.0 * g.lx, 2.0 * g.lx), rng.range(-2.0 * g.ly, 2.0 * g.ly)];
            let (kx, ky) = ((rng.next() * 5.0) as i32 - 2, (rng.next() * 5.0) as i32 - 2);
            let q = [p[0] + f64::from(kx) * g.lx, p[1] + f64::from(ky) * g.ly];
            let (a, b) = (lattice.bilinear(p), lattice.bilinear(q));
            assert!((a.sample(&snap.u) - b.sample(&snap.u)).abs() < 1e-12);
            assert!((a.sample(&snap.v) - b.sample(&snap.v)).abs() < 1e-12);
            let (wa, wb) = (lattice.catmull_rom(&snap.w, p), lattice.catmull_rom(&snap.w, q));
            assert!((wa - wb).abs() < 1e-12);
        }
        // Halfway between the last column and column 0 (across the seam).
        let row = 5;
        let p = [g.x0() + (g.nx as f64 - 0.5) * g.dx, g.y0() + row as f64 * g.dx];
        let expected =
            0.5 * (f64::from(snap.u[row * g.nx + g.nx - 1]) + f64::from(snap.u[row * g.nx]));
        assert!((lattice.bilinear(p).sample(&snap.u) - expected).abs() < 1e-15);
        // Catmull-Rom at a node reproduces the node value (taps wrap on both sides).
        for (i, j) in [(0, 0), (g.ny - 1, g.nx - 1), (0, g.nx - 2), (1, 1)] {
            let p = [g.x0() + j as f64 * g.dx, g.y0() + i as f64 * g.dx];
            let w = lattice.catmull_rom(&snap.w, p);
            assert!((w - f64::from(snap.w[i * g.nx + j])).abs() < 1e-14, "{i} {j}");
        }
    }

    /// The eager definition of docs/ember-design.md §5.2, written independently and without any
    /// shortcut: plain modular indexing on every tap, the gate evaluated at every step, every body
    /// tested at every step with the literal `_seg_iv` formula.
    pub(crate) fn reference_trace(
        window: &FlowWindow<'_>,
        rules: &ContactRules,
        start: [f64; 2],
    ) -> Trace {
        let g = window.grid;
        let (nx, ny) = (g.nx as i64, g.ny as i64);
        let inv_dx = 1.0 / g.dx;
        let tap = |f: &[f32], i: i64, j: i64| {
            f64::from(f[(i.rem_euclid(ny) * nx + j.rem_euclid(nx)) as usize])
        };
        let bilinear = |f: &[f32], p: [f64; 2]| {
            let gx = (p[0] - g.x0()) * inv_dx;
            let gy = (p[1] - g.y0()) * inv_dx;
            let (fx, fy) = (gx.floor(), gy.floor());
            let (wx, wy) = (gx - fx, gy - fy);
            let (j, i) = (fx as i64, fy as i64);
            (tap(f, i, j) * (1.0 - wx) + tap(f, i, j + 1) * wx) * (1.0 - wy)
                + (tap(f, i + 1, j) * (1.0 - wx) + tap(f, i + 1, j + 1) * wx) * wy
        };
        let cubic = |f: &[f32], p: [f64; 2]| {
            let gx = (p[0] - g.x0()) * inv_dx;
            let gy = (p[1] - g.y0()) * inv_dx;
            let (fx, fy) = (gx.floor(), gy.floor());
            let (a, b) = (catmull_rom_weights(gx - fx), catmull_rom_weights(gy - fy));
            let (j, i) = (fx as i64, fy as i64);
            let row = |r: i64| {
                tap(f, r, j - 1) * a[0]
                    + tap(f, r, j) * a[1]
                    + tap(f, r, j + 1) * a[2]
                    + tap(f, r, j + 2) * a[3]
            };
            row(i - 1) * b[0] + row(i) * b[1] + row(i + 1) * b[2] + row(i + 2) * b[3]
        };
        let clamp = |x: f64| if x > 0.0 { if x < 1.0 { x } else { 1.0 } } else { 0.0 };
        let snaps = window.snapshots;
        let last = snaps.len() - 1;
        let mut x = start;
        let mut rec: [Option<f64>; 3] = [None; 3];
        let mut o1 = cubic(&snaps[last].w, start);
        for s in (1..=last).rev() {
            let (t1, t0) = (snaps[s].time, snaps[s - 1].time);
            let (s1, s0) = (&snaps[s], &snaps[s - 1]);
            let h = -(t1 - t0);
            let u1 = |p| [bilinear(&s1.u, p), bilinear(&s1.v, p)];
            let u0 = |p| [bilinear(&s0.u, p), bilinear(&s0.v, p)];
            let k1 = u1(x);
            let x2 = [x[0] + h / 2.0 * k1[0], x[1] + h / 2.0 * k1[1]];
            let (a, b) = (u1(x2), u0(x2));
            let k2 = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
            let x3 = [x[0] + h / 2.0 * k2[0], x[1] + h / 2.0 * k2[1]];
            let (a, b) = (u1(x3), u0(x3));
            let k3 = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
            let k4 = u0([x[0] + h * k3[0], x[1] + h * k3[1]]);
            let xn = [
                x[0] + h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
                x[1] + h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
            ];
            let o0 = cubic(&s0.w, xn);
            let (g0, g1) = if rules.vorticity_gate > 0.0 {
                let wc = rules.vorticity_gate;
                let (a1, a0) = (o1.abs(), o0.abs());
                let sc = clamp((wc - a1) / if a0 == a1 { 1e-12 } else { a0 - a1 });
                let g0 = if a1 > wc {
                    0.0
                } else if a0 > wc {
                    sc
                } else {
                    1.0
                };
                let g1 = if a1 > wc {
                    if a0 > wc { 1.0 } else { sc }
                } else if a0 > wc {
                    1.0
                } else {
                    0.0
                };
                (g0, g1)
            } else {
                (0.0, 1.0)
            };
            let dt = t1 - t0;
            // The soak frame of a body state: the symmetric map R·diag(1/(semi + soak))·Rᵀ
            // (docs/ember-design.md §5.2 step 2), the same for an axis and its negative.
            let frame = |body: &BodyState, d: [f64; 2]| {
                let [c, sn] = body.shape.axis;
                let [i0, i1] = body.shape.semi.map(|semi| 1.0 / (semi + rules.soak_depth));
                let (xx, xy, yy) = (
                    (c * c) * i0 + (sn * sn) * i1,
                    (c * sn) * (i0 - i1),
                    (sn * sn) * i0 + (c * c) * i1,
                );
                [xx * d[0] + xy * d[1], xy * d[0] + yy * d[1]]
            };
            for (i, record) in rec.iter_mut().enumerate() {
                let (b1, b0) = (&window.bodies[s][i], &window.bodies[s - 1][i]);
                let a = frame(b1, [x[0] - b1.position[0], x[1] - b1.position[1]]);
                let b = frame(b0, [xn[0] - b0.position[0], xn[1] - b0.position[1]]);
                let r = 1.0;
                let d = [b[0] - a[0], b[1] - a[1]];
                let big_a = d[0] * d[0] + d[1] * d[1];
                let big_b = a[0] * d[0] + a[1] * d[1];
                let big_c = a[0] * a[0] + a[1] * a[1] - r * r;
                let disc = big_b * big_b - big_a * big_c;
                let sq = (if disc > 0.0 { disc } else { 0.0 }).sqrt();
                let big_as = if big_a > 1e-24 { big_a } else { 1e-24 };
                let mut seg0 = clamp((-big_b - sq) / big_as);
                let mut seg1 = clamp((-big_b + sq) / big_as);
                if disc <= 0.0 {
                    seg1 = seg0;
                }
                if big_a < 1e-20 {
                    seg0 = 0.0;
                    seg1 = if big_c < 0.0 { 1.0 } else { 0.0 };
                }
                let s_lo = if seg0 > g0 { seg0 } else { g0 };
                let s_hi = if seg1 < g1 { seg1 } else { g1 };
                let hit = s_hi > s_lo;
                let t_lo = t1 - s_hi * dt;
                let t_hi = t1 - s_lo * dt;
                if record.is_none() && hit && t_lo <= rules.t_valve {
                    *record = Some(if t_hi < rules.t_valve { t_hi } else { rules.t_valve });
                }
            }
            x = xn;
            o1 = o0;
        }
        for record in &mut rec {
            if record.is_some_and(|t| t < rules.t_on) {
                *record = None;
            }
        }
        Trace { origin: x, contact: rec }
    }

    /// A random smooth periodic flow (a few Fourier modes of a stream function) plus node noise,
    /// with a vorticity field of magnitude around `scale`; `S` snapshots with random spacing.
    pub(crate) fn random_window(
        rng: &mut Lcg,
        g: &FluidGrid,
        intervals: usize,
        speed: f64,
        scale: f64,
    ) -> (Vec<Snapshot>, Vec<[BodyState; 3]>) {
        let tau = 2.0 * std::f64::consts::PI;
        let modes: Vec<[f64; 5]> = (0..6)
            .map(|_| {
                let kx = tau * (rng.next() * 4.0).floor() / g.lx;
                let ky = tau * (rng.next() * 4.0 - 2.0).floor() / g.ly;
                [kx, ky, rng.range(0.0, tau), rng.range(-1.0, 1.0), rng.range(-3.0, 3.0)]
            })
            .collect();
        let mut t = rng.range(0.0, 2.0);
        let mut times = vec![t];
        for _ in 0..intervals {
            t += rng.range(0.001, 0.02);
            times.push(t);
        }
        let noise = rng.range(0.0, 0.3);
        let seed = rng.0;
        let snaps = times
            .iter()
            .map(|&time| {
                let mut local = Lcg(seed ^ time.to_bits());
                snapshot(g, time, |x, y| {
                    let (mut u, mut v, mut w) = (0.0, 0.0, 0.0);
                    for &[kx, ky, phase, amp, drift] in &modes {
                        let k = (kx * kx + ky * ky).sqrt().max(1.0);
                        let (s, c) = math::sin_cos(kx * x + ky * y + phase + drift * time);
                        u += speed * amp * ky / k * c;
                        v -= speed * amp * kx / k * c;
                        w += scale * amp * s;
                    }
                    let mut jitter = || noise * (local.next() - 0.5);
                    [u + speed * jitter(), v + speed * jitter(), w + scale * jitter()]
                })
            })
            .collect();
        // Straight paths; every outline an ellipse (a disc for one body in four) whose axes turn
        // and whose aspect changes from snapshot to snapshot.
        let mut body = || {
            let p = [rng.range(-1.0, 1.0), rng.range(-0.8, 0.8)];
            let vel = [rng.range(-3.0, 3.0), rng.range(-3.0, 3.0)];
            let radius = rng.range(0.0, 0.15);
            let disc = rng.next() < 0.25;
            let motion = [rng.range(0.0, 7.0), rng.range(-40.0, 40.0), rng.range(-9.0, 9.0)];
            (p, vel, radius, disc, motion)
        };
        let paths = [body(), body(), body()];
        let bodies = times
            .iter()
            .map(|&time| {
                let dt = time - times[0];
                paths.map(|(p, vel, radius, disc, [angle, turn, stretch])| {
                    let aspect = if disc { 1.0 } else { 2.0 + math::tanh(stretch * dt) };
                    let (s, c) = math::sin_cos(angle + turn * dt);
                    let k = aspect.sqrt();
                    BodyState {
                        position: [p[0] + vel[0] * dt, p[1] + vel[1] * dt],
                        velocity: vel,
                        shape: Shape {
                            semi: [radius * k, radius / k],
                            axis: [c, s],
                            spin: turn,
                            strain: 0.0,
                        },
                    }
                })
            })
            .collect();
        (snaps, bodies)
    }

    #[test]
    fn fast_tracer_equals_the_eager_reference_bit_for_bit() {
        let mut rng = Lcg(2024);
        let mut contacts = 0usize;
        let mut traces = 0usize;
        for case in 0..60 {
            let g = if case % 3 == 0 { grid(24, 16, 0.2) } else { grid(48, 32, 0.08) };
            let intervals = 1 + case % 4;
            let speed = if case % 5 == 0 { 40.0 } else { 2.0 };
            let (snaps, bodies) = random_window(&mut rng, &g, intervals, speed, 60.0);
            let window = FlowWindow { grid: g, snapshots: &snaps, bodies: &bodies };
            let (t_first, t_last) = (snaps[0].time, snaps[intervals].time);
            let pick = |rng: &mut Lcg| match (rng.next() * 4.0) as u32 {
                0 => t_first + (t_last - t_first) * rng.next(),
                1 => snaps[(rng.next() * (intervals + 1) as f64) as usize].time,
                2 => t_first - 1.0,
                _ => t_last + 1.0,
            };
            let rules = ContactRules {
                soak_depth: rng.range(0.05, 0.3),
                vorticity_gate: if case % 4 == 1 { 0.0 } else { rng.range(-5.0, 80.0) },
                t_on: pick(&mut rng),
                t_valve: pick(&mut rng),
            };
            let tracer = Tracer::new(&window, &rules);
            let bits =
                |t: &Trace| (t.origin.map(f64::to_bits), t.contact.map(|c| c.map(f64::to_bits)));
            for k in 0..75 {
                let starts: [[f64; 2]; 4] = std::array::from_fn(|lane| {
                    if (k + lane) % 3 == 0 {
                        [rng.range(-0.6 * g.lx, 0.6 * g.lx), rng.range(-0.6 * g.ly, 0.6 * g.ly)]
                    } else {
                        let b = bodies[intervals][(k + lane) % 3];
                        let reach = b.shape.extent() + rules.soak_depth;
                        let r = reach * rng.range(0.0, 1.5);
                        let (s, c) = math::sin_cos(rng.range(0.0, 7.0));
                        [b.position[0] + r * c, b.position[1] + r * s]
                    }
                });
                let lanes = tracer.trace_lanes(starts);
                for (start, fast) in starts.iter().zip(&lanes) {
                    let reference = reference_trace(&window, &rules, *start);
                    assert_eq!(bits(fast), bits(&reference), "case {case} start {start:?}");
                    assert_eq!(bits(&tracer.trace(*start)), bits(fast), "single vs lanes");
                    contacts += fast.contact.iter().flatten().count();
                    traces += 1;
                }
            }
        }
        // The comparison must exercise contacts (thousands of them), not only misses.
        assert!(contacts > traces / 10, "{contacts} contacts in {traces} traces");
    }

    #[test]
    fn a_single_snapshot_window_is_the_identity() {
        let g = grid(8, 8, 0.25);
        let snaps = vec![snapshot(&g, 1.0, |_, _| [1.0, 1.0, 100.0])];
        let bodies = vec![[[0.0, 0.0]; 3]];
        let outlines = points(&bodies);
        let window = FlowWindow { grid: g, snapshots: &snaps, bodies: &outlines };
        let trace = trace_back(&window, &open_rules(1.0), [0.1, 0.2]);
        assert_eq!(trace, Trace { origin: [0.1, 0.2], contact: [None; 3] });
    }

    #[test]
    #[should_panic(expected = "strictly")]
    fn non_increasing_times_are_rejected() {
        let g = grid(8, 8, 0.25);
        let snaps = vec![snapshot(&g, 1.0, |_, _| [0.0; 3]), snapshot(&g, 1.0, |_, _| [0.0; 3])];
        let bodies = vec![[[0.0, 0.0]; 3]; 2];
        let outlines = points(&bodies);
        let window = FlowWindow { grid: g, snapshots: &snaps, bodies: &outlines };
        let _ = Tracer::new(&window, &open_rules(1.0));
    }
}
