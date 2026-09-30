//! Ink fields on the supersampled node grid and their per-frame remap.
//!
//! Ink lives on a raster of *nodes* (`q×q` per output pixel plus a margin beyond the canvas). Each
//! node carries four `f32` fields (structure of arrays):
//!
//! * presence `P ∈ [0, 1]`: the node's water was inked by some body (1 for unmixed inked water;
//!   fractional where inked and clear water have been mixed at the node scale);
//! * freshness `E_i ∈ [0, 1]`, one per body: `E_i = exp(-(t - t*_i)/τ)` for the latest contact
//!   `t*_i` of the water with body `i` (0 if it never met body `i`), `τ` the look's fade time.
//!
//! Both are linear in the pigment concentration, so interpolating them models physical dilution.
//!
//! # Remap (characteristic mapping, docs/ember-design.md §5.3)
//!
//! Once per frame interval `(t_prev, t_frame]`, every node `n` at world position `x_n` is traced
//! back through the interval's velocity snapshots ([`Tracer`], exact backward RK4 characteristics
//! with gated soak-zone contacts) to its origin `X_n` at `t_prev`, recording per body the latest
//! contact `t*_i` in the interval. Then
//!
//! ```text
//! P'(n)   = 1                                      if any body recorded a contact
//!         = P(X_n) · floor_fade                    otherwise
//! E'_i(n) = exp(-(t_frame - t*_i)/τ)               if body i recorded
//!         = E_i(X_n) · fresh_fade                  otherwise
//! ```
//!
//! with `fresh_fade = exp(-(t_frame - t_prev)/τ)` (ageing by the interval) and `floor_fade` the
//! optional slow fade of the floor wash. (The stored `E` is the exact freshness up to its `f32`
//! rounding and the per-frame ageing by `fresh_fade`.)
//!
//! **The bodies are solid.** A node inside a body's outline at the frame time holds no water and
//! stores `P' = E'_i = 0`: a body's interior is never ink, so the flow cannot carry ink into it
//! and leak it back into the wakes, and a body always reads as bare paper.
//!
//! `P(X_n)`, `E_i(X_n)` are read from the previous fields by **clamped Catmull-Rom**
//! interpolation on the node grid: 4×4 taps, taps outside the grid read 0 (never-inked water
//! beyond the margin), and the result is clamped to the range of the inner 2×2 taps (monotone: no
//! overshoot, no negative ink). Fields that a contact overwrites are not sampled. Values below
//! [`FLUSH`] `= 10⁻¹²` are flushed to exactly 0 (no subnormals, no signed zeros) and the result is
//! rounded to `f32`.
//!
//! # Determinism
//!
//! Every node is computed independently by the same fixed sequence of `f64` operations; rows are
//! processed in parallel with results written to disjoint row slices, and the only reductions are
//! integer counts. The fields are therefore bit-identical for any rayon thread count.
//!
//! # Broken flow
//!
//! A NaN or infinite velocity in the window cannot panic the remap or leak into the fields: the
//! pathlines that read it come out with a non-finite origin, those nodes inherit clear water
//! (nothing is sampled at a non-finite point), and [`RemapStats::non_finite_origins`] counts them
//! for the caller to reject the frame.

use rayon::prelude::*;

use super::error::{EmberError, EmberResult};
use super::math::{self, max, min};
use super::orbit::BodyState;
use super::trace::{ContactRules, FlowWindow, Trace, Tracer, catmull_rom_weights};

/// Presence and freshness below this are flushed to exactly zero (no subnormals, and old ink does
/// not linger as noise). The look's hold amplifies freshness by `g = exp(hold / τ)`, so the flush
/// cuts up to `g·FLUSH` of strength at once; config validation bounds `g` accordingly
/// ([`crate::ember::EmberConfig::validate`]).
pub(crate) const FLUSH: f64 = 1e-12;

/// Nodes traced together in lock step (independent dependency chains for the out-of-order core).
const LANES: usize = 4;

/// The ink raster: `q×q` nodes per output pixel plus a margin of nodes around the canvas.
///
/// Node `(r, c)` sits at `x = -aspect + (c - margin + 0.5)·spacing`,
/// `y = 1 - (r - margin + 0.5)·spacing`; row 0 is the top. Output pixel `(row, col)` averages the
/// nodes `r = margin + q·row + i`, `c = margin + q·col + j` for `i, j ∈ 0..q`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NodeGrid {
    /// Output width in pixels.
    pub width: usize,
    /// Output height in pixels.
    pub height: usize,
    /// Nodes per pixel along each axis.
    pub supersample: usize,
    /// Margin nodes on every side.
    pub margin: usize,
    /// Node columns (`q·width + 2·margin`).
    pub cols: usize,
    /// Node rows (`q·height + 2·margin`).
    pub rows: usize,
    /// Node spacing in world units (`2 / (q·height)`).
    pub spacing: f64,
    /// Canvas half-width (`width / height`).
    pub aspect: f64,
}

impl NodeGrid {
    /// The node grid for a `width × height` output with `supersample²` nodes per pixel and
    /// `margin_world` world units of carried ink beyond the canvas.
    pub(crate) fn new(
        width: u32,
        height: u32,
        supersample: u32,
        margin_world: f64,
    ) -> EmberResult<Self> {
        if width == 0 || height == 0 || supersample == 0 {
            return Err(EmberError::InvalidConfig {
                parameter: "raster".into(),
                reason: format!("invalid raster {width}x{height} with supersample {supersample}"),
            });
        }
        let (width, height, q) = (width as usize, height as usize, supersample as usize);
        let spacing = 2.0 / (q * height) as f64;
        let margin = (margin_world / spacing).ceil() as usize;
        Ok(Self {
            width,
            height,
            supersample: q,
            margin,
            cols: q * width + 2 * margin,
            rows: q * height + 2 * margin,
            spacing,
            aspect: width as f64 / height as f64,
        })
    }

    /// Number of nodes.
    pub(crate) fn len(&self) -> usize {
        self.rows * self.cols
    }

    /// World position of node `(r, c)`.
    pub(crate) fn world(&self, r: usize, c: usize) -> [f64; 2] {
        let m = self.margin as f64;
        [
            -self.aspect + (c as f64 - m + 0.5) * self.spacing,
            1.0 - (r as f64 - m + 0.5) * self.spacing,
        ]
    }

    /// Continuous `(row, col)` node index of a world point (node centres at integers).
    pub(crate) fn index_of(&self, p: [f64; 2]) -> [f64; 2] {
        let m = self.margin as f64;
        [(1.0 - p[1]) / self.spacing - 0.5 + m, (p[0] + self.aspect) / self.spacing - 0.5 + m]
    }
}

/// Presence and per-body freshness of the ink carried by every node (structure of arrays, `f32`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct InkFields {
    /// `P ∈ [0, 1]`: the node's water has been inked by some body (diluted by mixing).
    pub presence: Vec<f32>,
    /// `E_i ∈ [0, 1]`: `exp(-(t - t*_i)/τ)` for the latest contact `t*_i` with body `i`.
    pub freshness: [Vec<f32>; 3],
}

impl InkFields {
    /// Fields of never-inked water.
    pub(crate) fn zeros(grid: &NodeGrid) -> Self {
        let n = grid.len();
        Self { presence: vec![0.0; n], freshness: [vec![0.0; n], vec![0.0; n], vec![0.0; n]] }
    }

    /// Whether all four fields hold exactly `len` nodes.
    fn has_len(&self, len: usize) -> bool {
        self.presence.len() == len && self.freshness.iter().all(|field| field.len() == len)
    }
}

/// Per-frame decay factors of the remap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct InkDecay {
    /// Fluid time of the frame (window end).
    pub frame_time: f64,
    /// The look's fade time `τ` (fluid units).
    pub fade_tau: f64,
    /// `exp(-(frame_time - window_start)/τ)`.
    pub fresh_fade: f64,
    /// `exp(-(frame_time - window_start)/floor_tau)`, or 1 when the floor never fades.
    pub floor_fade: f64,
}

/// Deterministic statistics of one remap (integer counts: exact under any reduction order).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RemapStats {
    /// Nodes that recorded at least one contact in this window.
    pub contacted_nodes: u64,
    /// Nodes whose traced origin is not finite: the window's flow held a NaN or infinite
    /// velocity (or time) on their pathline. Such a node inherits clear water, so the fields stay
    /// finite, but the frame is wrong: the pipeline must treat any non-zero count as
    /// [`super::error::EmberError::NonFinite`] (docs/ember-design.md §0, "check `is_finite` at
    /// stage boundaries").
    pub non_finite_origins: u64,
}

impl RemapStats {
    /// Element-wise sum.
    fn merge(self, other: Self) -> Self {
        Self {
            contacted_nodes: self.contacted_nodes + other.contacted_nodes,
            non_finite_origins: self.non_finite_origins + other.non_finite_origins,
        }
    }
}

/// Advances the ink fields from the window start to the window end: every node of `next` is traced
/// back through `window`, records fresh contacts, and otherwise inherits the (clamped Catmull-Rom
/// interpolated, decayed) value of `prev` at its origin. Nodes inside a body at the window end
/// hold no water and store zeros.
///
/// Rows are processed in parallel; the result does not depend on the number of threads.
///
/// A non-finite velocity in the window never panics or leaks NaN into the fields: the affected
/// nodes are counted in [`RemapStats::non_finite_origins`], which the caller must check.
///
/// # Panics
///
/// If `prev` or `next` does not hold `grid.len()` nodes per field, or the window is malformed
/// (see [`Tracer::new`]).
pub(crate) fn remap(
    prev: &InkFields,
    next: &mut InkFields,
    grid: &NodeGrid,
    window: &FlowWindow<'_>,
    rules: &ContactRules,
    decay: &InkDecay,
) -> RemapStats {
    let n = grid.len();
    assert!(prev.has_len(n) && next.has_len(n), "ink fields must match the node grid");
    let solid = *window.bodies.last().expect("a flow window has at least one snapshot");
    let remapper =
        Remapper { grid, tracer: Tracer::new(window, rules), prev, decay: *decay, solid };
    let cols = grid.cols;
    let [e0, e1, e2] = &mut next.freshness;
    (
        next.presence.par_chunks_mut(cols),
        e0.par_chunks_mut(cols),
        e1.par_chunks_mut(cols),
        e2.par_chunks_mut(cols),
    )
        .into_par_iter()
        .enumerate()
        .map(|(r, (presence, e0, e1, e2))| remapper.row(r, presence, [e0, e1, e2]))
        .reduce(RemapStats::default, RemapStats::merge)
}

/// Whether the world point `p` lies inside any of the `bodies`' outlines.
#[inline(always)]
fn inside_a_body(bodies: &[BodyState; 3], p: [f64; 2]) -> bool {
    bodies
        .iter()
        .any(|body| body.shape.contains([p[0] - body.position[0], p[1] - body.position[1]]))
}

/// Everything one remap needs, shared by all rows.
struct Remapper<'a> {
    /// The node grid.
    grid: &'a NodeGrid,
    /// The window, prepared for tracing.
    tracer: Tracer<'a>,
    /// Fields at the window start.
    prev: &'a InkFields,
    /// Decay factors of this frame.
    decay: InkDecay,
    /// The bodies at the window end: their interiors hold no water.
    solid: [BodyState; 3],
}

impl Remapper<'_> {
    /// Remaps node row `r` into the given output row slices and returns the row's statistics.
    ///
    /// Nodes are traced [`LANES`] at a time (lanes are independent, so this only changes the
    /// instruction schedule, not a single bit); the `cols mod LANES` nodes left at the end of the
    /// row are traced one by one.
    fn row(&self, r: usize, presence: &mut [f32], freshness: [&mut [f32]; 3]) -> RemapStats {
        let [e0, e1, e2] = freshness;
        let cols = presence.len();
        let mut stats = RemapStats::default();
        let mut store = |c: usize, trace: &Trace| {
            stats.non_finite_origins += u64::from(!trace.origin.iter().all(|v| v.is_finite()));
            if inside_a_body(&self.solid, self.grid.world(r, c)) {
                (presence[c], e0[c], e1[c], e2[c]) = (0.0, 0.0, 0.0, 0.0);
                return;
            }
            let (values, inked) = self.node(trace);
            (presence[c], e0[c], e1[c], e2[c]) = (values[0], values[1], values[2], values[3]);
            stats.contacted_nodes += u64::from(inked);
        };
        let grouped = cols - cols % LANES;
        for c0 in (0..grouped).step_by(LANES) {
            let starts = std::array::from_fn(|lane| self.grid.world(r, c0 + lane));
            for (lane, trace) in self.tracer.trace_lanes::<LANES>(starts).iter().enumerate() {
                store(c0 + lane, trace);
            }
        }
        for c in grouped..cols {
            store(c, &self.tracer.trace(self.grid.world(r, c)));
        }
        stats
    }

    /// `([P', E'_0, E'_1, E'_2], any contact)` of a water node from its backward trace.
    #[inline(always)]
    fn node(&self, trace: &Trace) -> ([f32; 4], bool) {
        let d = &self.decay;
        let inked = trace.contact.iter().any(Option::is_some);
        let stencil = if !inked || trace.contact.iter().any(Option::is_none) {
            NodeStencil::at(self.grid, trace.origin)
        } else {
            NodeStencil::Outside // every field is overwritten; nothing is sampled
        };
        let presence = if inked {
            1.0
        } else {
            flush(stencil.sample(&self.prev.presence, self.grid.cols) * d.floor_fade)
        };
        let fresh: [f64; 3] = std::array::from_fn(|i| match trace.contact[i] {
            Some(t) => flush(math::exp(-(d.frame_time - t) / d.fade_tau)),
            None => flush(stencil.sample(&self.prev.freshness[i], self.grid.cols) * d.fresh_fade),
        });
        ([presence as f32, fresh[0] as f32, fresh[1] as f32, fresh[2] as f32], inked)
    }
}

/// `v`, or exactly `+0` when `v < 10⁻¹²` (also maps `-0` to `+0`).
#[inline(always)]
fn flush(v: f64) -> f64 {
    if v < FLUSH { 0.0 } else { v }
}

/// Clamped Catmull-Rom stencil on the (non-periodic) node grid.
#[derive(Clone, Copy, Debug)]
enum NodeStencil {
    /// All sixteen taps lie outside the grid: every field reads 0.
    Outside,
    /// All sixteen taps lie inside: `corner` is the index of tap `(r0-1, c0-1)`.
    Inside {
        /// Index of the top-left tap.
        corner: usize,
        /// Column weights (taps `c0-1 … c0+2`).
        a: [f64; 4],
        /// Row weights (taps `r0-1 … r0+2`).
        b: [f64; 4],
    },
    /// Some taps lie outside: per tap row and column, its offset (`row·cols` or column) if inside.
    Edge {
        /// Row offsets `(r0-1+j)·cols` of the tap rows inside the grid.
        rows: [Option<usize>; 4],
        /// Tap columns inside the grid.
        cols: [Option<usize>; 4],
        /// Column weights.
        a: [f64; 4],
        /// Row weights.
        b: [f64; 4],
    },
}

impl NodeStencil {
    /// The stencil around world point `p` (the origin of a traced node). A non-finite point is
    /// treated as lying outside.
    #[inline(always)]
    fn at(grid: &NodeGrid, p: [f64; 2]) -> Self {
        let [fr, fc] = grid.index_of(p);
        let (rows, cols) = (grid.rows as f64, grid.cols as f64);
        // Taps r0-1 ..= r0+2 all miss [0, rows) iff r0 < -2 or r0 > rows, i.e. fr ∉ [-2, rows+1).
        if !(fr >= -2.0 && fr < rows + 1.0 && fc >= -2.0 && fc < cols + 1.0) {
            return Self::Outside;
        }
        let (r0, c0) = (fr.floor(), fc.floor());
        let b = catmull_rom_weights(fr - r0);
        let a = catmull_rom_weights(fc - c0);
        let (r0, c0) = (r0 as i64, c0 as i64);
        let (nr, nc) = (grid.rows as i64, grid.cols as i64);
        if r0 >= 1 && r0 + 2 < nr && c0 >= 1 && c0 + 2 < nc {
            let corner = (r0 - 1) as usize * grid.cols + (c0 - 1) as usize;
            return Self::Inside { corner, a, b };
        }
        let inside = |i: i64, n: i64| (0..n).contains(&i).then_some(i as usize);
        Self::Edge {
            rows: [-1, 0, 1, 2].map(|j| inside(r0 + j, nr).map(|r| r * grid.cols)),
            cols: [-1, 0, 1, 2].map(|i| inside(c0 + i, nc)),
            a,
            b,
        }
    }

    /// Clamped Catmull-Rom sample of `field` (row length `cols`).
    #[inline(always)]
    fn sample(&self, field: &[f32], cols: usize) -> f64 {
        match *self {
            Self::Outside => 0.0,
            Self::Inside { corner, a, b } => {
                let row = |j: usize| -> [f64; 4] {
                    let start = corner + j * cols;
                    let taps = &field[start..start + 4];
                    [taps[0], taps[1], taps[2], taps[3]].map(f64::from)
                };
                clamped_catmull_rom(row, a, b)
            }
            Self::Edge { rows, cols: columns, a, b } => {
                let row = |j: usize| -> [f64; 4] {
                    std::array::from_fn(|i| match (rows[j], columns[i]) {
                        (Some(r), Some(c)) => f64::from(field[r + c]),
                        _ => 0.0,
                    })
                };
                clamped_catmull_rom(row, a, b)
            }
        }
    }
}

/// `Σ_j b_j (Σ_i a_i t_ji)` over the 4×4 taps `t_ji = row(j)[i]` (left to right), clamped to the
/// range of the inner taps `t_11, t_12, t_21, t_22`. When the inner taps are all equal the clamp
/// alone fixes the result, so the outer taps are not read.
#[inline(always)]
fn clamped_catmull_rom(row: impl Fn(usize) -> [f64; 4], a: [f64; 4], b: [f64; 4]) -> f64 {
    let (t1, t2) = (row(1), row(2));
    let lo = min(min(t1[1], t1[2]), min(t2[1], t2[2]));
    let hi = max(max(t1[1], t1[2]), max(t2[1], t2[2]));
    if lo == hi {
        return lo;
    }
    let (t0, t3) = (row(0), row(3));
    let combine = |t: [f64; 4]| t[0] * a[0] + t[1] * a[1] + t[2] * a[2] + t[3] * a[3];
    let v = combine(t0) * b[0] + combine(t1) * b[1] + combine(t2) * b[2] + combine(t3) * b[3];
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::super::fluid::{FluidGrid, Snapshot};
    use super::super::math;
    use super::super::orbit::Shape;
    use super::super::trace::tests::{Lcg, grid, points, random_window, reference_trace, snapshot};
    use super::*;

    /// Bodies parked far outside the region under test.
    const FAR: [[f64; 2]; 3] = [[50.0, 50.0], [-50.0, 50.0], [50.0, -50.0]];

    /// Rules under which nothing can record (the valve closed before the window).
    fn no_contact() -> ContactRules {
        ContactRules { soak_depth: 0.1, vorticity_gate: 0.0, t_on: 0.0, t_valve: -1.0 }
    }

    fn decay(frame_time: f64, fresh_fade: f64, floor_fade: f64) -> InkDecay {
        InkDecay { frame_time, fade_tau: 0.12, fresh_fade, floor_fade }
    }

    /// Fields `P = p(x, y)`, `E_i = e_i(x, y)` sampled at the node positions (as `f32`).
    fn fields(nodes: &NodeGrid, f: impl Fn(f64, f64) -> [f64; 4]) -> InkFields {
        let mut out = InkFields::zeros(nodes);
        for r in 0..nodes.rows {
            for c in 0..nodes.cols {
                let [x, y] = nodes.world(r, c);
                let v = f(x, y);
                let n = r * nodes.cols + c;
                out.presence[n] = v[0] as f32;
                for i in 0..3 {
                    out.freshness[i][n] = v[i + 1] as f32;
                }
            }
        }
        out
    }

    /// A fluid box comfortably containing a node grid of the given aspect.
    fn fluid_for(aspect: f64) -> FluidGrid {
        let dx = 0.05;
        let nx = (2.0 * (aspect + 0.5) / dx).ceil() as usize;
        grid(nx, (3.0 / dx) as usize, dx)
    }

    /// A steady flow `field(x, y)` over `[0, dt]` in `steps` equal intervals.
    fn steady(
        fluid: &FluidGrid,
        dt: f64,
        steps: usize,
        field: impl Fn(f64, f64) -> [f64; 2] + Copy,
    ) -> (Vec<Snapshot>, Vec<[BodyState; 3]>) {
        let snaps = (0..=steps)
            .map(|s| {
                snapshot(fluid, dt * s as f64 / steps as f64, |x, y| {
                    let [u, v] = field(x, y);
                    [u, v, 0.0]
                })
            })
            .collect();
        (snaps, points(&vec![FAR; steps + 1]))
    }

    #[test]
    fn node_grid_geometry() {
        for q in 1..=3u32 {
            let nodes = NodeGrid::new(12, 8, q, 0.15).unwrap();
            let qs = q as usize;
            assert_eq!(nodes.spacing, 2.0 / (qs * 8) as f64);
            assert_eq!(nodes.margin, (0.15 / nodes.spacing).ceil() as usize);
            assert_eq!(nodes.cols, qs * 12 + 2 * nodes.margin);
            assert_eq!(nodes.rows, qs * 8 + 2 * nodes.margin);
            assert_eq!(nodes.aspect, 1.5);
            let s = 2.0 / 8.0;
            for row in 0..8 {
                for col in 0..12 {
                    // The pixel centre is the mean of its q×q nodes.
                    let mut mean = [0.0; 2];
                    for i in 0..qs {
                        for j in 0..qs {
                            let [x, y] = nodes
                                .world(nodes.margin + qs * row + i, nodes.margin + qs * col + j);
                            mean[0] += x / (qs * qs) as f64;
                            mean[1] += y / (qs * qs) as f64;
                        }
                    }
                    let centre = [-1.5 + (col as f64 + 0.5) * s, 1.0 - (row as f64 + 0.5) * s];
                    assert!(
                        (mean[0] - centre[0]).abs() < 1e-14 && (mean[1] - centre[1]).abs() < 1e-14
                    );
                }
            }
            // Node centres round-trip through the continuous index.
            for (r, c) in [(0, 0), (3, 7), (nodes.rows - 1, nodes.cols - 1)] {
                let [fr, fc] = nodes.index_of(nodes.world(r, c));
                assert!((fr - r as f64).abs() < 1e-12 && (fc - c as f64).abs() < 1e-12);
            }
            // The first canvas node is half a spacing inside the top-left corner.
            let [x, y] = nodes.world(nodes.margin, nodes.margin);
            assert!((x + 1.5 - 0.5 * nodes.spacing).abs() < 1e-15);
            assert!((y - 1.0 + 0.5 * nodes.spacing).abs() < 1e-15);
        }
        assert!(NodeGrid::new(0, 8, 2, 0.1).is_err());
        assert!(NodeGrid::new(8, 8, 0, 0.1).is_err());
        assert_eq!(NodeGrid::new(8, 4, 2, 0.0).unwrap().margin, 0);
    }

    #[test]
    fn uniform_flow_translates_a_blob_by_whole_nodes() {
        let nodes = NodeGrid::new(24, 16, 2, 0.1).unwrap();
        let fluid = fluid_for(nodes.aspect);
        let h = nodes.spacing;
        // Three node spacings right and two down over dt = 0.02, in three snapshot intervals.
        let dt = 0.02;
        let (u, v) = (3.0 * h / dt, -2.0 * h / dt);
        let (snaps, bodies) = steady(&fluid, dt, 3, |_, _| [u, v]);
        let window = FlowWindow { grid: fluid, snapshots: &snaps, bodies: &bodies };
        let blob = |x: f64, y: f64| {
            let r2 = (x - 0.1) * (x - 0.1) + (y + 0.05) * (y + 0.05);
            let b = if r2 < 0.25 { (1.0 - r2 / 0.25) * (1.0 - r2 / 0.25) } else { 0.0 };
            [b, 0.8 * b, 0.5 * b, 0.0]
        };
        let prev = fields(&nodes, blob);
        let mut next = InkFields::zeros(&nodes);
        let stats = remap(&prev, &mut next, &nodes, &window, &no_contact(), &decay(dt, 0.9, 0.95));
        assert_eq!(stats.contacted_nodes, 0);
        for r in 2..nodes.rows {
            for c in 3..nodes.cols {
                let (n, m) = (r * nodes.cols + c, (r - 2) * nodes.cols + c - 3);
                let expect = |field: &[f32], fade: f64| f64::from(field[m]) * fade;
                assert!((f64::from(next.presence[n]) - expect(&prev.presence, 0.95)).abs() < 1e-5);
                for i in 0..3 {
                    let got = f64::from(next.freshness[i][n]);
                    assert!((got - expect(&prev.freshness[i], 0.9)).abs() < 1e-5, "{r} {c} {i}");
                }
            }
        }
        assert!(next.freshness[2].iter().all(|&e| e == 0.0));
    }

    #[test]
    fn rotation_flow_rotates_the_fields() {
        let nodes = NodeGrid::new(40, 30, 2, 0.1).unwrap();
        let fluid = fluid_for(nodes.aspect);
        let omega = 3.0;
        let dt = 0.03;
        let (snaps, bodies) = steady(&fluid, dt, 3, |x, y| [-omega * y, omega * x]);
        let window = FlowWindow { grid: fluid, snapshots: &snaps, bodies: &bodies };
        let pattern = |x: f64, y: f64| {
            let (s, _) = math::sin_cos(2.0 * x);
            let (_, c) = math::sin_cos(1.5 * y);
            let p = 0.5 + 0.4 * s * c;
            [p, 0.9 * p, p * p, 0.3]
        };
        let prev = fields(&nodes, pattern);
        let mut next = InkFields::zeros(&nodes);
        remap(&prev, &mut next, &nodes, &window, &no_contact(), &decay(dt, 1.0, 1.0));
        let (s, c) = math::sin_cos(-omega * dt);
        let mut worst = 0.0f64;
        for r in 0..nodes.rows {
            for col in 0..nodes.cols {
                let [x, y] = nodes.world(r, col);
                if x.abs() > 1.2 || y.abs() > 0.9 {
                    continue; // origins must stay inside the grid
                }
                let expected = pattern(c * x - s * y, s * x + c * y);
                let n = r * nodes.cols + col;
                let got = [
                    next.presence[n],
                    next.freshness[0][n],
                    next.freshness[1][n],
                    next.freshness[2][n],
                ];
                for (g, e) in got.iter().zip(expected) {
                    worst = worst.max((f64::from(*g) - e).abs());
                }
            }
        }
        // Catmull-Rom error O(h³) for this smooth pattern at h = 1/30.
        assert!(worst < 2e-4, "{worst}");
    }

    #[test]
    fn a_disc_in_still_water_inks_its_swept_band() {
        let nodes = NodeGrid::new(30, 20, 2, 0.1).unwrap();
        let fluid = fluid_for(nodes.aspect);
        let times: Vec<f64> = (0..=4u32).map(|s| 0.6 + 0.01 * f64::from(s)).collect();
        let snaps: Vec<Snapshot> =
            times.iter().map(|&t| snapshot(&fluid, t, |_, _| [0.0; 3])).collect();
        // Body 0 sweeps from (-0.5, 0) to (0.5, 0); the others stay far away.
        let centres: Vec<[[f64; 2]; 3]> =
            times.iter().map(|&t| [[-0.5 + 25.0 * (t - 0.6), 0.0], FAR[1], FAR[2]]).collect();
        let bodies = points(&centres);
        let window = FlowWindow { grid: fluid, snapshots: &snaps, bodies: &bodies };
        let r = 0.08;
        let rules =
            ContactRules { soak_depth: r, vorticity_gate: 0.0, t_on: 0.5, t_valve: f64::INFINITY };
        let prev = fields(&nodes, |_, _| [0.0, 0.0, 0.3, 0.0]);
        let mut next = InkFields::zeros(&nodes);
        let t_frame = 0.64;
        let stats = remap(&prev, &mut next, &nodes, &window, &rules, &decay(t_frame, 0.5, 1.0));
        let mut band = 0;
        for row in 0..nodes.rows {
            for col in 0..nodes.cols {
                let [x, y] = nodes.world(row, col);
                let n = row * nodes.cols + col;
                let half = (r * r - y * y).sqrt();
                let inside = y.abs() < r && x + half > -0.5 && x - half < 0.5;
                let near_edge = (y.abs() - r).abs() < 1e-9
                    || (x + half + 0.5).abs() < 1e-9
                    || (x - half - 0.5).abs() < 1e-9;
                if near_edge {
                    continue;
                }
                assert_eq!(f64::from(next.freshness[1][n]), f64::from(0.3f32) * 0.5);
                assert_eq!(next.freshness[2][n], 0.0);
                if inside {
                    band += 1;
                    let exit = ((x + half + 0.5) / 25.0 + 0.6).min(t_frame);
                    let fresh = math::exp(-(t_frame - exit) / 0.12);
                    assert_eq!(next.presence[n], 1.0);
                    assert!((f64::from(next.freshness[0][n]) - fresh).abs() < 1e-6, "{x} {y}");
                } else {
                    assert_eq!(next.presence[n], 0.0);
                    assert_eq!(next.freshness[0][n], 0.0);
                }
            }
        }
        assert!(band > 80, "{band}");
        assert_eq!(stats.contacted_nodes, band);
    }

    #[test]
    fn water_from_beyond_the_grid_is_clear() {
        // Spacing 1/8 and dt = 2⁻⁷ make the velocity exact in f32.
        let nodes = NodeGrid::new(20, 16, 1, 0.0).unwrap();
        let fluid = fluid_for(nodes.aspect);
        let h = nodes.spacing;
        let dt = 0.0078125;
        // Everything moves ten nodes to the right.
        let (snaps, bodies) = steady(&fluid, dt, 2, |_, _| [10.0 * h / dt, 0.0]);
        let window = FlowWindow { grid: fluid, snapshots: &snaps, bodies: &bodies };
        let prev = fields(&nodes, |_, _| [1.0; 4]);
        let mut next = InkFields::zeros(&nodes);
        remap(&prev, &mut next, &nodes, &window, &no_contact(), &decay(dt, 0.5, 0.25));
        for r in 0..nodes.rows {
            for c in 0..nodes.cols {
                let n = r * nodes.cols + c;
                let (p, e) = (next.presence[n], next.freshness[0][n]);
                if c < 10 {
                    assert_eq!((p, e), (0.0, 0.0), "{r} {c}");
                } else {
                    assert_eq!((p, e), (0.25, 0.5), "{r} {c}");
                }
            }
        }
    }

    #[test]
    fn interpolation_is_clamped_to_the_inner_taps() {
        let nodes = NodeGrid::new(20, 16, 1, 0.0).unwrap();
        let fluid = fluid_for(nodes.aspect);
        let h = nodes.spacing;
        let dt = 0.0078125;
        // Half a node to the right: Catmull-Rom overshoots a step, the clamp removes it.
        let (snaps, bodies) = steady(&fluid, dt, 1, |_, _| [0.5 * h / dt, 0.0]);
        let window = FlowWindow { grid: fluid, snapshots: &snaps, bodies: &bodies };
        let prev = fields(&nodes, |x, _| if x < 0.0 { [1.0, 1.0, 0.0, 0.0] } else { [0.0; 4] });
        let mut next = InkFields::zeros(&nodes);
        remap(&prev, &mut next, &nodes, &window, &no_contact(), &decay(dt, 1.0, 1.0));
        let step = nodes.cols / 2;
        let row = &next.presence[5 * nodes.cols..6 * nodes.cols];
        assert!(row.iter().all(|&p| (0.0..=1.0).contains(&p)));
        assert_eq!(row[step - 1], 1.0, "overshoot clamped");
        assert!((row[step] - 0.5).abs() < 1e-6, "{}", row[step]);
        assert_eq!(row[step + 1], 0.0, "undershoot clamped");
    }

    #[test]
    fn fading_flushes_tiny_values_to_zero() {
        let nodes = NodeGrid::new(8, 6, 1, 0.0).unwrap();
        let fluid = fluid_for(nodes.aspect);
        let (snaps, bodies) = steady(&fluid, 0.01, 1, |_, _| [0.0, 0.0]);
        let window = FlowWindow { grid: fluid, snapshots: &snaps, bodies: &bodies };
        let prev = fields(&nodes, |_, _| [1.5e-12, 1.5e-12, 3e-12, 1.0]);
        let mut next = InkFields::zeros(&nodes);
        remap(&prev, &mut next, &nodes, &window, &no_contact(), &decay(0.01, 0.5, 0.5));
        assert!(next.presence.iter().all(|&p| p.to_bits() == 0));
        assert!(next.freshness[0].iter().all(|&e| e.to_bits() == 0));
        assert!(next.freshness[1].iter().all(|&e| e == (f64::from(3e-12f32) * 0.5) as f32));
        assert!(next.freshness[2].iter().all(|&e| e == 0.5));
    }

    /// Clamped Catmull-Rom sample of a node field at a continuous index, literally: sixteen taps
    /// with an explicit bounds check each, no shortcut.
    fn reference_sample(nodes: &NodeGrid, field: &[f32], p: [f64; 2]) -> f64 {
        let [fr, fc] = nodes.index_of(p);
        if !(fr.is_finite() && fc.is_finite()) {
            return 0.0;
        }
        let (r0, c0) = (fr.floor(), fc.floor());
        let (b, a) = (catmull_rom_weights(fr - r0), catmull_rom_weights(fc - c0));
        let tap = |j: i64, i: i64| {
            let (r, c) = (r0 as i64 - 1 + j, c0 as i64 - 1 + i);
            if r < 0 || c < 0 || r >= nodes.rows as i64 || c >= nodes.cols as i64 {
                0.0
            } else {
                f64::from(field[r as usize * nodes.cols + c as usize])
            }
        };
        let row =
            |j: i64| tap(j, 0) * a[0] + tap(j, 1) * a[1] + tap(j, 2) * a[2] + tap(j, 3) * a[3];
        let v = row(0) * b[0] + row(1) * b[1] + row(2) * b[2] + row(3) * b[3];
        let inner = [tap(1, 1), tap(1, 2), tap(2, 1), tap(2, 2)];
        let lo = inner.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = inner.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if v < lo {
            lo
        } else if v > hi {
            hi
        } else {
            v
        }
    }

    /// docs/ember-design.md §5.3 node by node with the eager reference tracer, the literal
    /// sampler and the literal inside-a-body test.
    fn reference_remap(
        prev: &InkFields,
        nodes: &NodeGrid,
        window: &FlowWindow<'_>,
        rules: &ContactRules,
        d: &InkDecay,
    ) -> (InkFields, u64) {
        let mut out = InkFields::zeros(nodes);
        let mut contacted = 0;
        let fl = |v: f64| if v < 1e-12 { 0.0 } else { v };
        for r in 0..nodes.rows {
            for c in 0..nodes.cols {
                let n = r * nodes.cols + c;
                let p = nodes.world(r, c);
                let trace = reference_trace(window, rules, p);
                let solid = window.bodies[window.bodies.len() - 1].iter().any(|body| {
                    let [a, b] = body.shape.semi;
                    let [cs, sn] = body.shape.axis;
                    let d = [p[0] - body.position[0], p[1] - body.position[1]];
                    let (x, y) = (cs * d[0] + sn * d[1], -sn * d[0] + cs * d[1]);
                    a > 0.0 && b > 0.0 && (x / a) * (x / a) + (y / b) * (y / b) < 1.0
                });
                if solid {
                    continue; // no water: the fields stay zero, and no contact counts
                }
                let any = trace.contact.iter().any(Option::is_some);
                contacted += u64::from(any);
                let p = reference_sample(nodes, &prev.presence, trace.origin);
                out.presence[n] = if any { 1.0 } else { fl(p * d.floor_fade) as f32 };
                for i in 0..3 {
                    let e = reference_sample(nodes, &prev.freshness[i], trace.origin);
                    out.freshness[i][n] = match trace.contact[i] {
                        Some(t) => fl(math::exp(-(d.frame_time - t) / d.fade_tau)) as f32,
                        None => fl(e * d.fresh_fade) as f32,
                    };
                }
            }
        }
        (out, contacted)
    }

    /// Previous fields with clear, saturated, smooth and noisy regions.
    fn patchy_fields(nodes: &NodeGrid, rng: &mut Lcg) -> InkFields {
        let mut out = InkFields::zeros(nodes);
        for n in 0..nodes.len() {
            let (r, c) = (n / nodes.cols, n % nodes.cols);
            let region = (r * 4 / nodes.rows) * 4 + c * 4 / nodes.cols;
            let mut value = |k: usize| -> f32 {
                match (region + k) % 4 {
                    0 => 0.0,
                    1 => 1.0,
                    2 => (0.5 + 0.4 * math::sin(0.3 * c as f64 + 0.2 * r as f64 + k as f64)) as f32,
                    _ => rng.next() as f32,
                }
            };
            out.presence[n] = value(0);
            for i in 0..3 {
                out.freshness[i][n] = value(i + 1);
            }
        }
        out
    }

    /// Bits of all four fields: presence, then freshness.
    fn field_bits(f: &InkFields) -> Vec<u32> {
        let mut all: Vec<u32> = f.presence.iter().map(|v| v.to_bits()).collect();
        for field in &f.freshness {
            all.extend(field.iter().map(|v| v.to_bits()));
        }
        all
    }

    #[test]
    fn remap_equals_the_node_by_node_reference_bit_for_bit() {
        let mut rng = Lcg(99);
        let mut total_contacted = 0;
        // (width, supersample): row lengths 52, 25, 26, 27, 75, 69 cover every remainder mod
        // LANES, so both the lane groups and the one-by-one row tail are compared.
        let rasters = [(24, 2), (23, 1), (24, 1), (25, 1), (23, 3), (21, 3)];
        let mut tails = [false; LANES];
        for (case, (width, q)) in rasters.into_iter().enumerate() {
            let nodes = NodeGrid::new(width, 16, q, 0.12).unwrap();
            tails[nodes.cols % LANES] = true;
            let fluid = grid(48, 32, 0.08);
            let speed = if case % 2 == 0 { 2.0 } else { 25.0 };
            let (snaps, bodies) = random_window(&mut rng, &fluid, 1 + case % 3, speed, 60.0);
            let window = FlowWindow { grid: fluid, snapshots: &snaps, bodies: &bodies };
            let last = snaps.len() - 1;
            let rules = ContactRules {
                soak_depth: rng.range(0.02, 0.3),
                vorticity_gate: if case == 3 { 0.0 } else { 40.0 },
                t_on: snaps[0].time + (snaps[last].time - snaps[0].time) * rng.range(-0.5, 0.5),
                t_valve: snaps[0].time + (snaps[last].time - snaps[0].time) * rng.range(0.5, 1.5),
            };
            let d = decay(snaps[last].time, 0.8, if case % 2 == 0 { 1.0 } else { 0.97 });
            let prev = patchy_fields(&nodes, &mut rng);
            let mut next = InkFields::zeros(&nodes);
            let stats = remap(&prev, &mut next, &nodes, &window, &rules, &d);
            let (expected, contacted) = reference_remap(&prev, &nodes, &window, &rules, &d);
            assert!(field_bits(&next) == field_bits(&expected), "case {case}");
            assert_eq!(stats, RemapStats { contacted_nodes: contacted, non_finite_origins: 0 });
            total_contacted += contacted;
        }
        assert!(total_contacted > 100, "{total_contacted}");
        assert_eq!(tails, [true; LANES]);
    }

    #[test]
    fn a_broken_flow_is_counted_and_never_leaks_into_the_fields() {
        // 12×8 nodes at x = ±0.125, ±0.375, …, ±1.375; fluid columns at x = -2 + 0.05 j.
        let nodes = NodeGrid::new(12, 8, 1, 0.0).unwrap();
        let fluid = fluid_for(nodes.aspect);
        assert_eq!((fluid.nx, fluid.x0()), (80, -2.0));
        let (mut snaps, bodies) = steady(&fluid, 0.01, 2, |_, _| [0.3, 0.0]);
        for n in 0..fluid.len() {
            let j = n % fluid.nx;
            if j < 40 {
                snaps[1].u[n] = f32::NAN; // x < 0 at the middle snapshot
            }
            if j >= 60 {
                snaps[0].v[n] = f32::INFINITY; // x ≥ 1 at the first snapshot
            }
        }
        let window = FlowWindow { grid: fluid, snapshots: &snaps, bodies: &bodies };
        let prev = fields(&nodes, |_, _| [0.5; 4]);
        let mut next = InkFields::zeros(&nodes);
        let stats = remap(&prev, &mut next, &nodes, &window, &no_contact(), &decay(0.01, 0.8, 0.9));
        // The parcels drift left by 0.003: every node at x < 0 reads the NaN column block, every
        // node at x > 1 the infinite one; the four columns in between are healthy.
        assert_eq!(stats, RemapStats { contacted_nodes: 0, non_finite_origins: 8 * (6 + 2) });
        for r in 0..nodes.rows {
            for c in 0..nodes.cols {
                let n = r * nodes.cols + c;
                let healthy = (6..10).contains(&c);
                let (p, e) = (next.presence[n], next.freshness[0][n]);
                if healthy {
                    assert_eq!((p, e), ((0.5f64 * 0.9) as f32, (0.5f64 * 0.8) as f32), "{r} {c}");
                } else {
                    assert_eq!((p.to_bits(), e.to_bits()), (0, 0), "{r} {c}");
                }
            }
        }
    }

    /// Cross-architecture canary: SHA-256 of the output bits of one remap on a fixed random window
    /// (plus its contact count). The fixture goes through `math::sin_cos` and `math::tanh`
    /// (libm) and `f32` rounding; the remap exercises RK4 with bilinear lookups, Catmull-Rom
    /// vorticity, the gate, the elliptical soak-zone test, the valve and the pre-roll, `math::exp`,
    /// the clamped Catmull-Rom sampler, the flush, the solid bodies and the row tail. One changed
    /// bit anywhere, on any CPU, changes the hash. Re-bless only for an intended change of the
    /// remap or of the shared test fixtures, and say why. Last re-blessed for `ember-v2`: the
    /// ember field is gone, the bodies are ellipses and their interiors are solid.
    #[test]
    fn remap_output_matches_the_golden_hash() {
        use sha2::{Digest, Sha256};
        const GOLDEN: &str = "3ed4290997b79fb2453ebc7276d528cd6f3e9f0545b3231699c52bfd32725afe";
        let mut rng = Lcg(4242);
        let nodes = NodeGrid::new(25, 16, 2, 0.12).unwrap();
        assert_eq!(nodes.cols % LANES, 2);
        let fluid = grid(48, 32, 0.08);
        let (snaps, bodies) = random_window(&mut rng, &fluid, 3, 6.0, 60.0);
        let window = FlowWindow { grid: fluid, snapshots: &snaps, bodies: &bodies };
        let (first, last) = (snaps[0].time, snaps[3].time);
        let rules = ContactRules {
            soak_depth: 0.3,
            vorticity_gate: 40.0,
            t_on: first + 0.3 * (last - first),
            t_valve: first + 0.8 * (last - first),
        };
        let prev = patchy_fields(&nodes, &mut rng);
        let mut next = InkFields::zeros(&nodes);
        let stats = remap(&prev, &mut next, &nodes, &window, &rules, &decay(last, 0.8, 0.97));
        assert_eq!(stats.non_finite_origins, 0);
        assert!(stats.contacted_nodes > 50, "{stats:?}");
        let mut hasher = Sha256::new();
        hasher.update(stats.contacted_nodes.to_le_bytes());
        for bits in field_bits(&next) {
            hasher.update(bits.to_le_bytes());
        }
        assert_eq!(hex::encode(hasher.finalize()), GOLDEN);
    }

    #[test]
    fn remap_is_bit_identical_for_any_thread_count() {
        let mut rng = Lcg(17);
        let nodes = NodeGrid::new(40, 26, 2, 0.1).unwrap();
        let fluid = grid(64, 40, 0.07);
        let (snaps, bodies) = random_window(&mut rng, &fluid, 3, 2.0, 60.0);
        let window = FlowWindow { grid: fluid, snapshots: &snaps, bodies: &bodies };
        let rules = ContactRules {
            soak_depth: 0.3,
            vorticity_gate: 40.0,
            t_on: f64::NEG_INFINITY,
            t_valve: f64::INFINITY,
        };
        let d = decay(snaps[3].time, 0.8, 0.99);
        let prev = patchy_fields(&nodes, &mut rng);
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                let mut next = InkFields::zeros(&nodes);
                let stats = remap(&prev, &mut next, &nodes, &window, &rules, &d);
                (next, stats)
            })
        };
        let (one, one_stats) = run(1);
        let (three, three_stats) = run(3);
        assert!(one_stats.contacted_nodes > 0);
        assert_eq!(one_stats, three_stats);
        assert!(one.presence.iter().zip(&three.presence).all(|(a, b)| a.to_bits() == b.to_bits()));
        for i in 0..3 {
            let (a, b) = (&one.freshness[i], &three.freshness[i]);
            assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    /// Throughput of the remap on a production-like frame: 3×3 nodes per pixel of a 960×620 view
    /// (plus the ink margin) over the default fluid grid, a smooth divergence-free flow with
    /// |u| ~ 1–3 and three snapshot intervals of 0.0026. Run with
    /// `cargo test --release --lib ember::ink::tests::remap_throughput -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark"]
    fn remap_throughput() {
        use std::time::Instant;
        let (width, height) = (960u32, 620u32);
        let defaults = crate::ember::EmberConfig::default();
        let nodes = NodeGrid::new(width, height, defaults.raster.supersample, 0.15).unwrap();
        let aspect = f64::from(width) / f64::from(height);
        let fluid =
            FluidGrid::for_canvas(aspect, defaults.fluid.rows, defaults.fluid.box_margin).unwrap();
        let tau = 2.0 * std::f64::consts::PI;
        let mut rng = Lcg(1);
        // Stream function ψ = Σ A sin(k·x + φ + ct) over periodic modes; u = ∂ψ/∂y, v = -∂ψ/∂x.
        let modes: Vec<[f64; 5]> = (0..24)
            .map(|_| {
                let kx = tau * (1.0 + (rng.next() * 8.0).floor()) / fluid.lx;
                let ky = tau * ((rng.next() * 12.0).floor() - 6.0) / fluid.ly;
                let k = (kx * kx + ky * ky).sqrt();
                [kx, ky, rng.range(0.0, tau), 0.45 / k, rng.range(-5.0, 5.0)]
            })
            .collect();
        let times: Vec<f64> = (0..=3u32).map(|s| 3.0 + 0.0026 * f64::from(s)).collect();
        let started = Instant::now();
        let snaps: Vec<Snapshot> = times
            .par_iter()
            .map(|&t| {
                snapshot(&fluid, t, |x, y| {
                    let (mut u, mut v, mut w) = (0.0, 0.0, 0.0);
                    for &[kx, ky, phase, amp, drift] in &modes {
                        let (s, c) = math::sin_cos(kx * x + ky * y + phase + drift * t);
                        u += amp * ky * c;
                        v -= amp * kx * c;
                        w += amp * (kx * kx + ky * ky) * s;
                    }
                    [u, v, w]
                })
            })
            .collect();
        let speeds: Vec<f64> = snaps[0]
            .u
            .iter()
            .zip(&snaps[0].v)
            .map(|(u, v)| f64::from(u * u + v * v).sqrt())
            .collect();
        let mean = speeds.iter().sum::<f64>() / speeds.len() as f64;
        let peak = speeds.iter().copied().fold(0.0, f64::max);
        println!(
            "flow built in {:.2}s: mean |u| {mean:.2}, max {peak:.2}",
            started.elapsed().as_secs_f64()
        );
        let bodies: Vec<[BodyState; 3]> =
            times
                .iter()
                .map(|&t| {
                    let s = (t - 3.0) * 2.0;
                    [[-0.6 + s, 0.2], [0.5, -0.3 + s], [0.1 - s, 0.5 - s]].map(|position| {
                        BodyState { position, velocity: [2.0, 0.0], shape: Shape::disc(0.05) }
                    })
                })
                .collect();
        let window = FlowWindow { grid: fluid, snapshots: &snaps, bodies: &bodies };
        let rules =
            ContactRules { soak_depth: 0.03, vorticity_gate: 40.0, t_on: 0.5, t_valve: 10.0 };
        let d =
            InkDecay { frame_time: times[3], fade_tau: 0.12, fresh_fade: 0.94, floor_fade: 1.0 };
        // Blocky fields (uniform patches: the clamp shortcut applies almost everywhere) and noisy
        // fields (every sample needs all sixteen taps: the worst case).
        let mut blocky = InkFields::zeros(&nodes);
        for (n, p) in blocky.presence.iter_mut().enumerate() {
            let clear = (n / nodes.cols / 64 + n % nodes.cols / 64).is_multiple_of(3);
            *p = if clear { 0.0 } else { 0.7 };
        }
        blocky.freshness[0].clone_from(&blocky.presence);
        let mut noisy = InkFields::zeros(&nodes);
        for field in std::iter::once(&mut noisy.presence).chain(noisy.freshness.iter_mut()) {
            for v in field.iter_mut() {
                *v = 0.1 + 0.8 * rng.next() as f32;
            }
        }
        let node_steps = (nodes.len() * 3) as f64;
        println!(
            "nodes {}x{} ({:.2} M), fluid {}x{}",
            nodes.cols,
            nodes.rows,
            nodes.len() as f64 * 1e-6,
            fluid.nx,
            fluid.ny
        );
        let all = std::thread::available_parallelism().map_or(1, usize::from);
        for (name, prev) in [("blocky", &blocky), ("noisy", &noisy)] {
            for threads in [1, all] {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                let mut next = InkFields::zeros(&nodes);
                let mut best = f64::INFINITY;
                let mut stats = RemapStats::default();
                for _ in 0..3 {
                    let clock = Instant::now();
                    stats = pool.install(|| remap(prev, &mut next, &nodes, &window, &rules, &d));
                    best = best.min(clock.elapsed().as_secs_f64());
                }
                println!(
                    "{name} fields, {threads:2} threads: {best:.3} s/frame, {:.3e} node-steps/s \
                     ({:.3e} per thread), {} contacted",
                    node_steps / best,
                    node_steps / best / threads as f64,
                    stats.contacted_nodes
                );
            }
        }
        // Tracing alone (no field sampling), single thread, with and without contact work.
        let closed = ContactRules { t_valve: 0.0, ..rules };
        for (name, rules) in [("with contacts", &rules), ("pathlines only", &closed)] {
            let tracer = Tracer::new(&window, rules);
            let clock = Instant::now();
            let mut checksum = 0.0;
            for r in 0..nodes.rows {
                for c0 in (0..nodes.cols - LANES).step_by(LANES) {
                    let starts = std::array::from_fn(|lane| nodes.world(r, c0 + lane));
                    let traces = tracer.trace_lanes::<LANES>(starts);
                    checksum += traces[0].origin[0];
                }
            }
            let rate = node_steps / clock.elapsed().as_secs_f64();
            println!("trace {name}, 1 thread: {rate:.3e} node-steps/s ({checksum:.1})");
        }
    }
}
