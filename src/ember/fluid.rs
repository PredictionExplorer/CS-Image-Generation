//! Two-dimensional Navier–Stokes wake solver: the water the three bodies stir.
//!
//! A port of the museum-lab `wake/ns.py` solver (same equations and operator structure) in `f64`
//! on the deterministic [`Rfft2d`].
//!
//! # Equations
//!
//! Incompressible flow in vorticity–streamfunction form on the doubly periodic box
//! `[-lx/2, lx/2) × [-ly/2, ly/2)` (world units; the canvas `[-aspect, aspect] × [-1, 1]` sits in
//! the middle):
//!
//! ```text
//! ∂ω/∂t + u·∇ω = ν∇²ω,    ω = ∂v/∂x - ∂u/∂y,    ψ̂ = ω̂/k²,    u = ∂ψ/∂y,    v = -∂ψ/∂x
//! ```
//!
//! with `ν = u_ref·(2R)/Re` (reference speed `u_ref`, body radius `R`, Reynolds number `Re`).
//! The bodies enter through Brinkman volume penalisation and a sponge absorbs wakes that leave
//! the canvas, both applied as a split step after every advection–diffusion step.
//!
//! # Discretisation
//!
//! * **Space.** Pseudo-spectral on the `ny × nx` node grid of [`FluidGrid`]; the state is the half
//!   spectrum `ω̂[ky][kx]` ([`Spectrum`], numpy `rfft2` layout) with `kx_j = 2πj/lx`
//!   (`j ≤ nx/2`) and `ky_i = 2π·(i < ny/2 ? i : i - ny)/ly` (the Nyquist row is negative),
//!   `k² = kx² + ky²`. The 2/3 rule keeps only `|kx|, |ky| < (2/3)·π/dx` (strict; evaluated
//!   exactly in integers as `3|m| < n`); every other mode of the state is exactly zero.
//! * **Velocity.** `û = i·ky·ω̂/k²`, `v̂ = -i·kx·ω̂/k²` (zero at `k = 0`).
//! * **Advection.** `N(ω̂) = -mask·FFT(u·∂ₓω + v·∂ᵧω)` with `∂ₓω = IFFT(i·kx·ω̂)`,
//!   `∂ᵧω = IFFT(i·ky·ω̂)` (five real transforms).
//! * **Dissipation.** `D = ν·k² + (c_h/dx)·(k²/k_c²)^12` with `k_c = (2/3)·π/dx` and the
//!   hyperviscosity coefficient `c_h`; the 24th power of `k/k_c` is formed by repeated squaring
//!   (`x²`, `x⁴`, `x⁸`, `x¹⁶`, `x²⁴ = x¹⁶·x⁸`). It only acts near the grid scale.
//! * **Time.** Lawson integrating-factor RK4 (diffusion exact) with `e₂ = exp(-D·h/2)` (flushed to
//!   zero when `D·h/2 > 700`) and `e₁ = e₂²`:
//!
//!   ```text
//!   k1 = N(ω̂)                   k2 = N(e₂·(ω̂ + h/2·k1))
//!   k3 = N(e₂·ω̂ + h/2·k2)       k4 = N(e₁·ω̂ + h·e₂·k3)
//!   ω̂ ← e₁·ω̂ + h/6·(e₁·k1 + 2·e₂·(k2 + k3) + k4)
//!   ```
//!
//!   (`e₁` is also flushed to zero when `D·h > 700`, where `e₂²` would be subnormal.)
//! * **Bodies** (after each RK4 step, at the new time `t + h`). With `(u, v)` the velocity of
//!   `ω̂`, each body `b` (centre `p_b`, velocity `V_b`, elliptical outline with signed distance
//!   `δ_b(x)` and deformation velocity `D_b(x)`, see `orbit::Shape`; edge width `W`) has the mask
//!   `c_b = ½·(1 - tanh(δ_b(x - p_b)/W))` (no periodic wrap). With `χ = max_b c_b` and the
//!   mask-weighted material velocity `ū = Σ c_b·(V_b + D_b) / max(Σ c_b, 10⁻¹²)`, the implicit
//!   Brinkman step with `a = χ/η` (`η` = permeability as a fraction of the step) is
//!   `u ← (u + a·ū)/(1 + a)`. For a disc `δ = |x - p| - R` and `D = 0`. The mask is evaluated only
//!   where `δ/W < 20`, inside a node box around each body that reaches `extent + 20·W` from its
//!   centre. The `δ/W` cut changes no bit: beyond it the values would be below `5·10⁻¹⁸`, and the
//!   pinned libm's `tanh` already rounds to exactly `1` from `δ/W ≈ 19.07` on (a unit test checks
//!   this). A disc's box holds every node with `δ/W < 20`; away from a stretched body's outline
//!   Taubin's distance falls below the Euclidean one, so its box also drops mask values, all below
//!   `2·10⁻¹³` at the defaults. The box is part of the definition (docs/ember-design.md §4.2).
//! * **Projection and sponge.** `ω = IFFT(i·kx·FFT(v) - i·ky·FFT(u))·exp(-σ·h)` with
//!   `σ(x, y) = σ₀·max(ramp(x; aspect + pad, lx/2 - 4dx), ramp(y; 1 + pad, ly/2 - 4dx))`,
//!   `ramp(c; in, out) = s²(3 - 2s)`, `s = clamp((|c| - in)/(out - in), 0, 1)`, then
//!   `ω̂ = mask·FFT(ω)`. Taking the curl and rebuilding `u` from `ψ` is the projection onto
//!   divergence-free fields.
//! * **Step size.** `h = min(cfl·dx/max(u_max, s_bodies, 10⁻⁶), h_max)`, where `u_max` is the
//!   largest penalised flow speed of the previous step and `s_bodies` the bodies' look-ahead speed
//!   (`BodyMotion::speed_bound`: the largest sampled material speed of any body, centre plus
//!   deformation, over a short window ahead, the prototype's rule); the final steps before a target time are adjusted so that the target is
//!   hit exactly (see [`WakeSolver::advance_to`]).
//!
//! 26 real two-dimensional transforms per step: 20 for the four advection evaluations (four
//! inverses and one forward each) and 6 for the forcing (`u`, `v` of `ω̂`; `FFT(u)`, `FFT(v)`;
//! the curl; the dealiasing forward). The 23 whose input or output is dealiased skip the
//! wavenumber columns `kx ≥ (2/3)·π/dx`, which are exactly zero ([`Rfft2d::inverse_with`],
//! [`Rfft2d::forward_with`]); only `FFT(u)`, `FFT(v)` and the curl run over the full width.
//!
//! Units: lengths in world units (the canvas is 2 high), times in fluid time units (the median
//! body speed is `u_ref`), `ν` in length²/time, `D` and `σ` in 1/time.
//!
//! # Conventions worth knowing
//!
//! * As in numpy and the prototype, the Nyquist row has `ky = -π/dx`. The dealiased state never
//!   holds Nyquist modes, but the curl of the penalised velocity does, and there the Nyquist
//!   part of `∂u/∂y` is mirror-symmetric in `y` rather than antisymmetric. The sponge multiplies
//!   that `y`-uniform checkerboard and folds a little of it into kept modes, so a mirror-symmetric
//!   set-up is not exactly mirror-antisymmetric in `ω` while the sponge is on. Measured for one
//!   disc moving along the axis (largest `|ω(x, y) + ω(x, -y)|` over the largest `|ω|`; with the
//!   Nyquist `ky` zeroed it stays below `3·10⁻¹²`, accumulated round-off): `10⁻³` on the 48×32
//!   test grid, whose sponge ramp is 1.4 cells wide; `3·10⁻⁸` on a 720×512 grid (ramp ~51 cells)
//!   and `10⁻¹⁰` on a 1440×1024 grid (ramp ~106 cells), both with the default parameters after
//!   0.1 time units. The default 2160×1536 grid's ramp is ~160 cells wide.
//! * The sponge is a physical-space multiplication, so it changes the mean vorticity (the `k = 0`
//!   mode) slightly; the velocity ignores that mode (`1/k² := 0`), the vorticity snapshots
//!   include it — as in the prototype.
//!
//! # Determinism
//!
//! Every value is computed by a fixed sequence of IEEE-754 operations; transcendental functions
//! go through [`math`]; parallel loops only split independent rows (or independent FFT blocks);
//! the only reductions are the exact maximum of `u² + v²`, logical ANDs of finiteness flags and
//! integer counts. Results are bit-identical for every thread count and CPU architecture.

use std::f64::consts::{PI, TAU};
use std::fmt;

use rayon::prelude::*;

use super::config::FluidConfig;
use super::error::{EmberError, EmberResult};
use super::fft::{Rfft2d, Spectrum, next_smooth_even};
use super::math;
use super::orbit::{BodyMotion, BodyState};

/// Value of `(d - R)/W` at and beyond which the body mask is taken to be exactly zero.
const MASK_CUTOFF: f64 = 20.0;
/// Integrating-factor exponents above this are flushed to exactly zero (avoids subnormals).
const FLUSH_EXPONENT: f64 = 700.0;
/// Relative slack within which the remaining time to a target is taken in one step.
const LANDING_SLACK: f64 = 1e-9;
/// Floor of the speed scale of the CFL condition.
const MIN_SPEED: f64 = 1e-6;
/// Floor of the summed mask in the mask-weighted body velocity.
const CHI_SUM_FLOOR: f64 = 1e-12;

/// The doubly periodic fluid box and its grid.
///
/// Node `(i, j)` (row `i`, column `j`) sits at `x = -lx/2 + j·dx`, `y = -ly/2 + i·dx`; row 0 is
/// the bottom of the box (`y` minimum) and storage is row-major (`index = i·nx + j`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FluidGrid {
    /// Columns (even, 5-smooth).
    pub nx: usize,
    /// Rows (even, 5-smooth).
    pub ny: usize,
    /// Grid spacing (same along both axes).
    pub dx: f64,
    /// Box width `nx·dx`.
    pub lx: f64,
    /// Box height `ny·dx`.
    pub ly: f64,
}

impl FluidGrid {
    /// The smallest box of `rows` (rounded up to even 5-smooth) rows that contains the canvas
    /// `[-aspect, aspect] × [-1, 1]` plus `margin` on every side.
    pub(crate) fn for_canvas(aspect: f64, rows: usize, margin: f64) -> EmberResult<Self> {
        if !(aspect.is_finite() && aspect > 0.0 && margin.is_finite() && margin > 0.0) {
            return Err(EmberError::InvalidConfig {
                parameter: "fluid.box_margin".into(),
                reason: format!("invalid canvas aspect {aspect} or margin {margin}"),
            });
        }
        let ny = next_smooth_even(rows);
        let dx = 2.0 * (1.0 + margin) / ny as f64;
        let nx = next_smooth_even((2.0 * (aspect + margin) / dx).ceil() as usize);
        Ok(Self { nx, ny, dx, lx: nx as f64 * dx, ly: ny as f64 * dx })
    }

    /// Number of nodes.
    pub(crate) fn len(&self) -> usize {
        self.nx * self.ny
    }

    /// World `x` of column 0.
    pub(crate) fn x0(&self) -> f64 {
        -0.5 * self.lx
    }

    /// World `y` of row 0.
    pub(crate) fn y0(&self) -> f64 {
        -0.5 * self.ly
    }
}

/// Velocity and vorticity at one instant on the fluid grid (row-major, row 0 = bottom), rounded
/// to `f32` for compact tracing.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Snapshot {
    /// Fluid time of the snapshot.
    pub time: f64,
    /// Velocity `x` component.
    pub u: Vec<f32>,
    /// Velocity `y` component.
    pub v: Vec<f32>,
    /// Dealiased vorticity `ω = ∂v/∂x - ∂u/∂y`.
    pub w: Vec<f32>,
}

impl Snapshot {
    /// A zeroed snapshot for `grid`.
    pub(crate) fn zeros(grid: &FluidGrid) -> Self {
        let n = grid.len();
        Self { time: 0.0, u: vec![0.0; n], v: vec![0.0; n], w: vec![0.0; n] }
    }
}

/// Deterministic run statistics of the solver.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct FluidStats {
    /// Time steps taken.
    pub steps: u64,
    /// Smallest step size (0 before the first step).
    pub min_dt: f64,
    /// Largest step size.
    pub max_dt: f64,
    /// Largest penalised flow speed seen.
    pub max_speed: f64,
    /// Body-mask evaluations (nodes × bodies with `(d - R)/W < 20`), summed over all steps: the
    /// work of the Brinkman forcing.
    pub mask_evaluations: u64,
}

/// Physical and numerical parameters of the solver, derived from [`FluidConfig`].
#[derive(Clone, Copy, Debug)]
struct Params {
    /// Courant number.
    cfl: f64,
    /// Largest step.
    max_dt: f64,
    /// Width `W` of the tanh edge of the body mask.
    mask_width: f64,
    /// Brinkman permeability `η` as a fraction of the step (`a = χ/η`).
    eta_ratio: f64,
}

/// Wavenumber-space operators, all in the half-spectrum layout.
struct Operators {
    /// `kx_j = 2πj/lx`, `j ≤ nx/2`.
    kx: Vec<f64>,
    /// `ky_i = 2π·(i < ny/2 ? i : i - ny)/ly`.
    ky: Vec<f64>,
    /// `1/k²` inside the dealiasing mask (0 at `k = 0` and outside the mask).
    inv_k2: Vec<f64>,
    /// `D = ν·k² + (c_h/dx)·(k²/k_c²)^12` inside the mask (0 outside).
    damping: Vec<f64>,
    /// Columns `kx < band` are inside the mask (`band = ⌈nx/3⌉`).
    band: usize,
    /// Whether row `ky` is inside the mask (`3·|m| < ny`).
    row_in_mask: Vec<bool>,
    /// Modes per spectrum row (`nx/2 + 1`).
    kx_len: usize,
}

impl Operators {
    /// Operators of `grid` with viscosity `nu` and hyperviscosity coefficient `hyper`.
    fn new(grid: &FluidGrid, nu: f64, hyper: f64) -> Self {
        let (nx, ny, dx) = (grid.nx, grid.ny, grid.dx);
        let kx_len = nx / 2 + 1;
        let kx: Vec<f64> = (0..kx_len).map(|j| TAU * j as f64 / grid.lx).collect();
        let signed = |i: usize| if i < ny / 2 { i as f64 } else { i as f64 - ny as f64 };
        let ky: Vec<f64> = (0..ny).map(|i| TAU * signed(i) / grid.ly).collect();
        // 3j < nx  ⇔  kx_j < (2/3)·π/dx, exactly.
        let band = nx.div_ceil(3);
        let row_in_mask: Vec<bool> = (0..ny)
            .map(|i| {
                let m = if i < ny / 2 { i } else { ny - i };
                3 * m < ny
            })
            .collect();
        let kc = (2.0 / 3.0) * (PI / dx);
        let kc2 = kc * kc;
        let hyper_rate = hyper / dx;
        let mut inv_k2 = vec![0.0; ny * kx_len];
        let mut damping = vec![0.0; ny * kx_len];
        for i in (0..ny).filter(|&i| row_in_mask[i]) {
            for j in 0..band {
                let k2 = kx[j] * kx[j] + ky[i] * ky[i];
                let x2 = k2 / kc2;
                let x4 = x2 * x2;
                let x8 = x4 * x4;
                let x16 = x8 * x8;
                let x24 = x16 * x8;
                inv_k2[i * kx_len + j] = if k2 > 0.0 { 1.0 / k2 } else { 0.0 };
                damping[i * kx_len + j] = nu * k2 + hyper_rate * x24;
            }
        }
        Self { kx, ky, inv_k2, damping, band, row_in_mask, kx_len }
    }

    /// `û = i·ky·ψ̂` with `ψ̂ = ω̂/k²`, in place on the segment `kx0..` of row `row`.
    fn to_u(&self, row: usize, kx0: usize, re: &mut [f64], im: &mut [f64]) {
        let ky = self.ky[row];
        let inv = &self.inv_k2[row * self.kx_len + kx0..][..re.len()];
        for ((r, i), &g) in re.iter_mut().zip(im.iter_mut()).zip(inv) {
            let (pr, pi) = (*r * g, *i * g);
            (*r, *i) = (-(ky * pi), ky * pr);
        }
    }

    /// `v̂ = -i·kx·ψ̂`, in place on the segment `kx0..` of row `row`.
    fn to_v(&self, row: usize, kx0: usize, re: &mut [f64], im: &mut [f64]) {
        let inv = &self.inv_k2[row * self.kx_len + kx0..][..re.len()];
        let kx = &self.kx[kx0..][..re.len()];
        for (((r, i), &g), &k) in re.iter_mut().zip(im.iter_mut()).zip(inv).zip(kx) {
            let (pr, pi) = (*r * g, *i * g);
            (*r, *i) = (k * pi, -(k * pr));
        }
    }

    /// `i·kx·ω̂` (the spectrum of `∂ₓω`), in place.
    fn to_dx(&self, kx0: usize, re: &mut [f64], im: &mut [f64]) {
        let kx = &self.kx[kx0..][..re.len()];
        for ((r, i), &k) in re.iter_mut().zip(im.iter_mut()).zip(kx) {
            (*r, *i) = (-(k * *i), k * *r);
        }
    }

    /// `i·ky·ω̂` (the spectrum of `∂ᵧω`), in place.
    fn to_dy(&self, row: usize, re: &mut [f64], im: &mut [f64]) {
        let ky = self.ky[row];
        for (r, i) in re.iter_mut().zip(im.iter_mut()) {
            (*r, *i) = (-(ky * *i), ky * *r);
        }
    }

    /// `i·kx·V̂ - i·ky·Û` in place, where the segment holds `Û` and `v_hat` supplies `V̂`.
    fn curl(&self, v_hat: &Spectrum, row: usize, kx0: usize, re: &mut [f64], im: &mut [f64]) {
        let ky = self.ky[row];
        let at = row * self.kx_len + kx0;
        let n = re.len();
        let (vr, vi) = (&v_hat.re[at..at + n], &v_hat.im[at..at + n]);
        let kx = &self.kx[kx0..kx0 + n];
        for l in 0..n {
            let (ur, ui) = (re[l], im[l]);
            // (i·kx)·V - (i·ky)·U = (-kx·Vi + ky·Ui) + i·(kx·Vr - ky·Ur).
            re[l] = -(kx[l] * vi[l]) - -(ky * ui);
            im[l] = kx[l] * vr[l] - ky * ur;
        }
    }

    /// Multiplies a finished spectrum row by the dealiasing mask (`sign = 1`) or by `-mask`
    /// (`sign = -1`). Columns `kx ≥ band` are already zero (band-limited transform).
    fn mask_row(&self, row: usize, sign: f64, re: &mut [f64], im: &mut [f64]) {
        if !self.row_in_mask[row] {
            re.fill(0.0);
            im.fill(0.0);
        } else if sign < 0.0 {
            for (r, i) in re[..self.band].iter_mut().zip(&mut im[..self.band]) {
                (*r, *i) = (-*r, -*i);
            }
        }
    }
}

/// `s²(3 - 2s)` with `s = clamp((|c| - inner)/(outer - inner), 0, 1)`; a step at `inner` if the
/// ramp is degenerate (`outer ≤ inner`, only on very coarse grids).
fn ramp(c: f64, inner: f64, outer: f64) -> f64 {
    let width = outer - inner;
    let s = if width > 0.0 {
        ((c.abs() - inner) / width).clamp(0.0, 1.0)
    } else if c.abs() > inner {
        1.0
    } else {
        0.0
    };
    s * s * (3.0 - 2.0 * s)
}

/// Vorticity sponge `σ(x, y) = σ₀·max(ramp_x(x), ramp_y(y))`, stored separably.
///
/// Because `σ₀·max(a, b) = max(σ₀·a, σ₀·b)` exactly (rounding is monotone) and `exp(-σ·h)` only
/// depends on which factor wins, `exp(-σ_ij·h)` equals the per-column factor `exp(-σx_j·h)` when
/// `σx_j > σy_i` and the per-row factor otherwise, bit for bit, at `nx + ny` exponentials per
/// step.
struct Sponge {
    /// `σ₀·ramp_x` per column.
    sigma_x: Vec<f64>,
    /// `σ₀·ramp_y` per row.
    sigma_y: Vec<f64>,
    /// `exp(-σx·h)` per column for the cached step.
    factor_x: Vec<f64>,
    /// `exp(-σy·h)` per row for the cached step.
    factor_y: Vec<f64>,
}

/// Grid-node coordinates, with `x_j = (j - nx/2)·dx` and `y_i = (i - ny/2)·dx`: the same values as
/// `-lx/2 + j·dx` in real arithmetic, with a single rounding, so the node set is exactly symmetric
/// about the origin.
struct Nodes {
    /// `x` of each column.
    x: Vec<f64>,
    /// `y` of each row.
    y: Vec<f64>,
}

impl Nodes {
    /// Node coordinates of `grid`.
    fn new(grid: &FluidGrid) -> Self {
        let axis = |n: usize| -> Vec<f64> {
            (0..n).map(|j| (j as f64 - (n / 2) as f64) * grid.dx).collect()
        };
        Self { x: axis(grid.nx), y: axis(grid.ny) }
    }
}

/// Integrating factors `e₂ = exp(-D·h/2)`, `e₁ = e₂²` for the cached step `h`.
struct Factors {
    /// Step the factors belong to (`None` before the first step).
    step: Option<f64>,
    /// `e₂` per mode (0 outside the mask).
    e2: Vec<f64>,
    /// `e₁` per mode (0 outside the mask).
    e1: Vec<f64>,
}

/// Physical-space work fields (`ny × nx`).
struct Fields {
    /// `u` (advection velocity, then the penalised velocity).
    u: Vec<f64>,
    /// `v`.
    v: Vec<f64>,
    /// `∂ₓω`, later the projected vorticity.
    a: Vec<f64>,
    /// `∂ᵧω`.
    b: Vec<f64>,
}

/// Rows and columns (half-open) of the nodes one body's mask can reach, plus its state.
#[derive(Clone, Copy, Debug)]
struct BodyBox {
    /// First row.
    row_lo: usize,
    /// One past the last row.
    row_hi: usize,
    /// First column.
    col_lo: usize,
    /// One past the last column.
    col_hi: usize,
    /// The body.
    state: BodyState,
}

impl BodyBox {
    /// The node box of `state`: every node within `reach` (`extent + 20·W`) of its centre along
    /// each axis, plus a cell of margin (the per-node test decides), or `None` if it misses the
    /// grid. For a disc it holds every node with `δ/W < 20`.
    fn new(state: BodyState, nodes: &Nodes, dx: f64, reach: f64) -> Option<Self> {
        let range = |centre: f64, len: usize| -> Option<(usize, usize)> {
            let half = (len / 2) as f64;
            let lo = ((centre - reach) / dx).floor() + half - 1.0;
            let hi = ((centre + reach) / dx).ceil() + half + 2.0;
            if hi <= 0.0 || lo >= len as f64 {
                return None;
            }
            Some((lo.max(0.0) as usize, (hi as usize).min(len)))
        };
        let (col_lo, col_hi) = range(state.position[0], nodes.x.len())?;
        let (row_lo, row_hi) = range(state.position[1], nodes.y.len())?;
        Some(Self { row_lo, row_hi, col_lo, col_hi, state })
    }
}

/// Pseudo-spectral vorticity solver with Brinkman-penalised moving bodies.
pub(crate) struct WakeSolver {
    /// The grid.
    grid: FluidGrid,
    /// Parameters.
    params: Params,
    /// Transform plan.
    fft: Rfft2d,
    /// Spectral operators.
    ops: Operators,
    /// Sponge.
    sponge: Sponge,
    /// Node coordinates.
    nodes: Nodes,
    /// Integrating factors.
    factors: Factors,
    /// The state `ω̂` (dealiased half spectrum).
    w_hat: Spectrum,
    /// RK4 stage derivative 1 (also `FFT(u)` during the forcing).
    k1: Spectrum,
    /// RK4 stage derivative 2 (also `FFT(v)` during the forcing).
    k2: Spectrum,
    /// RK4 stage derivative 3.
    k3: Spectrum,
    /// RK4 stage derivative 4.
    k4: Spectrum,
    /// RK4 stage input (zero outside the mask).
    stage: Spectrum,
    /// Physical work fields.
    fields: Fields,
    /// Fluid time.
    t: f64,
    /// Largest penalised flow speed of the last step.
    umax: f64,
    /// Statistics.
    stats: FluidStats,
}

impl fmt::Debug for WakeSolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WakeSolver")
            .field("grid", &self.grid)
            .field("t", &self.t)
            .field("umax", &self.umax)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

/// Rejects `value` unless finite and `> 0` (or `≥ 0` if `allow_zero`).
fn require(value: f64, allow_zero: bool, parameter: &str) -> EmberResult<()> {
    if value.is_finite() && (value > 0.0 || (allow_zero && value == 0.0)) {
        Ok(())
    } else {
        Err(EmberError::InvalidConfig {
            parameter: parameter.to_string(),
            reason: format!(
                "must be finite and {} 0, got {value}",
                if allow_zero { ">=" } else { ">" }
            ),
        })
    }
}

impl WakeSolver {
    /// A fluid at rest at `t = 0` on `grid` with the parameters of `config`; the sponge frames
    /// the canvas `[-aspect, aspect] × [-1, 1]`.
    pub(crate) fn new(grid: FluidGrid, config: &FluidConfig, aspect: f64) -> EmberResult<Self> {
        require(aspect, false, "aspect")?;
        require(grid.dx, false, "fluid grid spacing")?;
        require(grid.lx, false, "fluid box width")?;
        require(grid.ly, false, "fluid box height")?;
        require(config.reynolds, false, "fluid.reynolds")?;
        require(config.reference_speed, false, "fluid.reference_speed")?;
        require(config.body_radius, false, "fluid.body_radius")?;
        require(config.cfl, false, "fluid.cfl")?;
        require(config.max_dt, false, "fluid.max_dt")?;
        require(config.brinkman_eta_ratio, false, "fluid.brinkman_eta_ratio")?;
        require(config.mask_width, false, "fluid.mask_width")?;
        require(config.sponge_rate, true, "fluid.sponge_rate")?;
        require(config.sponge_pad, true, "fluid.sponge_pad")?;
        require(config.hyperviscosity, true, "fluid.hyperviscosity")?;
        if grid.lx != grid.nx as f64 * grid.dx || grid.ly != grid.ny as f64 * grid.dx {
            return Err(EmberError::InvalidConfig {
                parameter: "fluid grid".into(),
                reason: format!(
                    "box {}x{} is not {}x{} cells of {}",
                    grid.lx, grid.ly, grid.nx, grid.ny, grid.dx
                ),
            });
        }
        let fft = Rfft2d::new(grid.nx, grid.ny)?;
        debug_assert_eq!((fft.nx(), fft.ny()), (grid.nx, grid.ny));

        let nu = config.reference_speed * (2.0 * config.body_radius) / config.reynolds;
        let ops = Operators::new(&grid, nu, config.hyperviscosity);
        let nodes = Nodes::new(&grid);
        let edge = 4.0 * grid.dx;
        let sponge_x: Vec<f64> = nodes
            .x
            .iter()
            .map(|&x| {
                config.sponge_rate * ramp(x, aspect + config.sponge_pad, 0.5 * grid.lx - edge)
            })
            .collect();
        let sponge_y: Vec<f64> = nodes
            .y
            .iter()
            .map(|&y| config.sponge_rate * ramp(y, 1.0 + config.sponge_pad, 0.5 * grid.ly - edge))
            .collect();
        let sponge = Sponge {
            factor_x: vec![1.0; grid.nx],
            factor_y: vec![1.0; grid.ny],
            sigma_x: sponge_x,
            sigma_y: sponge_y,
        };
        let modes = grid.ny * fft.kx_len();
        let n = grid.len();
        Ok(Self {
            grid,
            params: Params {
                cfl: config.cfl,
                max_dt: config.max_dt,
                mask_width: config.mask_width,
                eta_ratio: config.brinkman_eta_ratio,
            },
            factors: Factors { step: None, e2: vec![0.0; modes], e1: vec![0.0; modes] },
            w_hat: fft.zero_spectrum(),
            k1: fft.zero_spectrum(),
            k2: fft.zero_spectrum(),
            k3: fft.zero_spectrum(),
            k4: fft.zero_spectrum(),
            stage: fft.zero_spectrum(),
            fields: Fields { u: vec![0.0; n], v: vec![0.0; n], a: vec![0.0; n], b: vec![0.0; n] },
            fft,
            ops,
            sponge,
            nodes,
            t: 0.0,
            umax: 0.0,
            stats: FluidStats::default(),
        })
    }

    /// Current fluid time.
    pub(crate) fn time(&self) -> f64 {
        self.t
    }

    /// Statistics so far.
    pub(crate) fn stats(&self) -> FluidStats {
        self.stats
    }

    /// Integrates until the fluid time equals `target` exactly (CFL-limited steps).
    ///
    /// Each step first takes `h_cfl = min(cfl·dx/max(u_max, s_bodies(t), 10⁻⁶), h_max)`. With
    /// `rem = target - t`: if `rem ≤ h_cfl·(1 + 10⁻⁹)` the step is `rem` and the time is set to
    /// `target` exactly afterwards; if `rem < 2·h_cfl` the step is `rem/2` (so the last two steps
    /// are balanced instead of leaving a sliver); otherwise it is `h_cfl`. A `target` equal to the
    /// current time takes no step.
    ///
    /// # Errors
    ///
    /// [`EmberError::InvalidSchedule`] if `target` is not finite or lies before the current time;
    /// [`EmberError::NonFinite`] if the flow, the bodies or the step size stop being finite.
    pub(crate) fn advance_to(&mut self, target: f64, bodies: &dyn BodyMotion) -> EmberResult<()> {
        if !target.is_finite() || target < self.t {
            return Err(EmberError::InvalidSchedule {
                reason: format!("fluid target time {target} precedes the current time {}", self.t),
            });
        }
        while self.t < target {
            let bound = bodies.speed_bound(self.t);
            if !(bound.is_finite() && bound >= 0.0) {
                return Err(EmberError::NonFinite { stage: "body speed bound", time: self.t });
            }
            let vmax = if self.umax > bound { self.umax } else { bound };
            let vmax = if vmax > MIN_SPEED { vmax } else { MIN_SPEED };
            let cfl_step = self.params.cfl * self.grid.dx / vmax;
            let h_cfl = if cfl_step < self.params.max_dt { cfl_step } else { self.params.max_dt };
            let rem = target - self.t;
            let (h, t_new) = if rem <= h_cfl * (1.0 + LANDING_SLACK) {
                (rem, target)
            } else if rem < 2.0 * h_cfl {
                (rem / 2.0, self.t + rem / 2.0)
            } else {
                (h_cfl, self.t + h_cfl)
            };
            if !(h > 0.0 && t_new > self.t) {
                return Err(EmberError::NonFinite { stage: "fluid time step", time: self.t });
            }
            self.step(h, t_new, bodies)?;
            self.t = t_new;
        }
        Ok(())
    }

    /// Writes the velocity and dealiased vorticity at the current time into `out`
    /// (`u, v = velocity(ω̂)`, `w = IFFT(ω̂)`, each rounded to `f32`).
    pub(crate) fn snapshot_into(&mut self, out: &mut Snapshot) {
        let n = self.grid.len();
        let nx = self.grid.nx;
        out.u.resize(n, 0.0);
        out.v.resize(n, 0.0);
        out.w.resize(n, 0.0);
        let Self { fft, ops, w_hat, fields, .. } = self;
        let band = ops.band;
        fft.inverse_with(w_hat, band, |r, k, re, im| ops.to_u(r, k, re, im), &mut fields.u);
        fft.inverse_with(w_hat, band, |r, k, re, im| ops.to_v(r, k, re, im), &mut fields.v);
        fft.inverse_with(w_hat, band, |_, _, _, _| {}, &mut fields.a);
        for (dst, src) in
            [(&mut out.u, &fields.u), (&mut out.v, &fields.v), (&mut out.w, &fields.a)]
        {
            dst.par_chunks_mut(nx).zip(src.par_chunks(nx)).for_each(|(d, s)| {
                for (d, &s) in d.iter_mut().zip(s) {
                    *d = s as f32;
                }
            });
        }
        out.time = self.t;
    }

    /// One step of size `h` ending at `t_new`: IF-RK4 advection–diffusion, then penalisation,
    /// projection and sponge with the bodies at `t_new`.
    fn step(&mut self, h: f64, t_new: f64, bodies: &dyn BodyMotion) -> EmberResult<()> {
        self.update_factors(h);
        self.ifrk4(h);
        let states = bodies.bodies_at(t_new);
        let finite = states
            .iter()
            .all(|b| b.position.iter().chain(&b.velocity).all(|value| value.is_finite()));
        if !finite {
            return Err(EmberError::NonFinite { stage: "body motion", time: t_new });
        }
        let evaluations = self.force(t_new, &states)?;
        let stats = &mut self.stats;
        if stats.steps == 0 || h < stats.min_dt {
            stats.min_dt = h;
        }
        if h > stats.max_dt {
            stats.max_dt = h;
        }
        if self.umax > stats.max_speed {
            stats.max_speed = self.umax;
        }
        stats.steps += 1;
        stats.mask_evaluations += evaluations;
        Ok(())
    }

    /// Recomputes the integrating factors and the sponge factors for step `h` (no-op if `h` is
    /// the cached step).
    fn update_factors(&mut self, h: f64) {
        if self.factors.step == Some(h) {
            return;
        }
        let Self { ops, factors, sponge, .. } = self;
        let (band, kx_len) = (ops.band, ops.kx_len);
        let half = h / 2.0;
        factors
            .e2
            .par_chunks_mut(kx_len)
            .zip(factors.e1.par_chunks_mut(kx_len))
            .enumerate()
            .for_each(|(row, (e2, e1))| {
                if !ops.row_in_mask[row] {
                    return;
                }
                let damping = &ops.damping[row * kx_len..row * kx_len + band];
                for ((e2, e1), &d) in e2.iter_mut().zip(e1.iter_mut()).zip(damping) {
                    let exponent = d * half;
                    if exponent > FLUSH_EXPONENT {
                        (*e2, *e1) = (0.0, 0.0);
                    } else {
                        let e = math::exp(-exponent);
                        // e₁ = e₂² = e^(-2·exponent): flushed by the same rule on its own
                        // exponent, before it could reach the subnormal range (e^-708).
                        *e2 = e;
                        *e1 = if exponent + exponent > FLUSH_EXPONENT { 0.0 } else { e * e };
                    }
                }
            });
        let factor = |sigma: f64| if sigma > 0.0 { math::exp(-(sigma * h)) } else { 1.0 };
        for (f, &s) in sponge.factor_x.iter_mut().zip(&sponge.sigma_x) {
            *f = factor(s);
        }
        for (f, &s) in sponge.factor_y.iter_mut().zip(&sponge.sigma_y) {
            *f = factor(s);
        }
        factors.step = Some(h);
    }

    /// One Lawson IF-RK4 step of the advection–diffusion equation.
    fn ifrk4(&mut self, h: f64) {
        let Self { fft, ops, factors, w_hat, k1, k2, k3, k4, stage, fields, .. } = self;
        let (e1, e2) = (&factors.e1, &factors.e2);
        let half = h / 2.0;

        nonlinear(fft, ops, w_hat, k1, fields);
        // stage = e₂·(ω̂ + h/2·k1)
        masked_rows(ops, stage, |at, sr, si| {
            let n = sr.len();
            let (wr, wi, ar, ai) =
                (&w_hat.re[at..][..n], &w_hat.im[at..][..n], &k1.re[at..][..n], &k1.im[at..][..n]);
            let e2 = &e2[at..][..n];
            for j in 0..n {
                sr[j] = e2[j] * (wr[j] + half * ar[j]);
                si[j] = e2[j] * (wi[j] + half * ai[j]);
            }
        });
        nonlinear(fft, ops, stage, k2, fields);
        // stage = e₂·ω̂ + h/2·k2
        masked_rows(ops, stage, |at, sr, si| {
            let n = sr.len();
            let (wr, wi, br, bi) =
                (&w_hat.re[at..][..n], &w_hat.im[at..][..n], &k2.re[at..][..n], &k2.im[at..][..n]);
            let e2 = &e2[at..][..n];
            for j in 0..n {
                sr[j] = e2[j] * wr[j] + half * br[j];
                si[j] = e2[j] * wi[j] + half * bi[j];
            }
        });
        nonlinear(fft, ops, stage, k3, fields);
        // stage = e₁·ω̂ + h·e₂·k3
        masked_rows(ops, stage, |at, sr, si| {
            let n = sr.len();
            let (wr, wi, cr, ci) =
                (&w_hat.re[at..][..n], &w_hat.im[at..][..n], &k3.re[at..][..n], &k3.im[at..][..n]);
            let (e1, e2) = (&e1[at..][..n], &e2[at..][..n]);
            for j in 0..n {
                let g = h * e2[j];
                sr[j] = e1[j] * wr[j] + g * cr[j];
                si[j] = e1[j] * wi[j] + g * ci[j];
            }
        });
        nonlinear(fft, ops, stage, k4, fields);
        // ω̂ ← e₁·ω̂ + h/6·(e₁·k1 + 2·e₂·(k2 + k3) + k4)
        let sixth = h / 6.0;
        let (k1, k2, k3, k4) = (&*k1, &*k2, &*k3, &*k4);
        masked_rows(ops, w_hat, |at, wr, wi| {
            let n = wr.len();
            let (e1, e2) = (&e1[at..][..n], &e2[at..][..n]);
            let (ar, ai) = (&k1.re[at..][..n], &k1.im[at..][..n]);
            let (br, bi) = (&k2.re[at..][..n], &k2.im[at..][..n]);
            let (cr, ci) = (&k3.re[at..][..n], &k3.im[at..][..n]);
            let (dr, di) = (&k4.re[at..][..n], &k4.im[at..][..n]);
            for j in 0..n {
                let two_e2 = 2.0 * e2[j];
                wr[j] = e1[j] * wr[j] + sixth * (e1[j] * ar[j] + two_e2 * (br[j] + cr[j]) + dr[j]);
                wi[j] = e1[j] * wi[j] + sixth * (e1[j] * ai[j] + two_e2 * (bi[j] + ci[j]) + di[j]);
            }
        });
    }

    /// The forcing split at `t_new`: penalise the velocity of `ω̂` towards the bodies, record the
    /// peak speed, take the curl, apply the sponge and dealias. Returns the mask evaluations.
    fn force(&mut self, t_new: f64, states: &[BodyState; 3]) -> EmberResult<u64> {
        let Self { grid, params, fft, ops, sponge, nodes, w_hat, k1, k2, fields, .. } = self;
        let (nx, band, full) = (grid.nx, ops.band, ops.kx_len);

        fft.inverse_with(w_hat, band, |r, k, re, im| ops.to_u(r, k, re, im), &mut fields.u);
        fft.inverse_with(w_hat, band, |r, k, re, im| ops.to_v(r, k, re, im), &mut fields.v);
        let evaluations = penalise(grid, params, nodes, states, &mut fields.u, &mut fields.v);

        let (peak2, finite) = fields
            .u
            .par_chunks(nx)
            .zip(fields.v.par_chunks(nx))
            .map(|(u, v)| {
                let mut peak = 0.0;
                let mut finite = true;
                for (&a, &b) in u.iter().zip(v) {
                    let s = a * a + b * b;
                    finite &= s.is_finite();
                    if s > peak {
                        peak = s;
                    }
                }
                (peak, finite)
            })
            .reduce(|| (0.0, true), |(p, f), (q, g)| (if q > p { q } else { p }, f && g));
        if !finite {
            return Err(EmberError::NonFinite { stage: "fluid velocity", time: t_new });
        }
        self.umax = peak2.sqrt();

        // Project: ω = IFFT(i·kx·FFT(v) - i·ky·FFT(u)).
        fft.forward(&fields.u, k1);
        fft.forward(&fields.v, k2);
        let v_hat = &*k2;
        fft.inverse_with(k1, full, |r, k, re, im| ops.curl(v_hat, r, k, re, im), &mut fields.a);

        // Sponge: ω ← ω·exp(-σ·h) where σ > 0.
        let finite = fields
            .a
            .par_chunks_mut(nx)
            .enumerate()
            .map(|(row, w)| {
                let (sy, fy) = (sponge.sigma_y[row], sponge.factor_y[row]);
                let mut finite = true;
                for ((w, &sx), &fx) in w.iter_mut().zip(&sponge.sigma_x).zip(&sponge.factor_x) {
                    if sx > sy {
                        *w *= fx;
                    } else if sy > 0.0 {
                        *w *= fy;
                    }
                    finite &= w.is_finite();
                }
                finite
            })
            .reduce(|| true, |f, g| f && g);
        if !finite {
            return Err(EmberError::NonFinite { stage: "fluid vorticity", time: t_new });
        }

        // Dealias: ω̂ = mask·FFT(ω).
        let omega = &fields.a;
        fft.forward_with(
            |row, dst| dst.copy_from_slice(&omega[row * nx..(row + 1) * nx]),
            band,
            |row, _, re, im| ops.mask_row(row, 1.0, re, im),
            w_hat,
        );
        Ok(evaluations)
    }
}

/// `N(ω̂) = -mask·FFT(u·∂ₓω + v·∂ᵧω)` into `out` (five transforms).
fn nonlinear(
    fft: &mut Rfft2d,
    ops: &Operators,
    input: &Spectrum,
    out: &mut Spectrum,
    fields: &mut Fields,
) {
    let (nx, band) = (fft.nx(), ops.band);
    fft.inverse_with(input, band, |r, k, re, im| ops.to_u(r, k, re, im), &mut fields.u);
    fft.inverse_with(input, band, |r, k, re, im| ops.to_v(r, k, re, im), &mut fields.v);
    fft.inverse_with(input, band, |_, k, re, im| ops.to_dx(k, re, im), &mut fields.a);
    fft.inverse_with(input, band, |r, _, re, im| ops.to_dy(r, re, im), &mut fields.b);
    let (u, v, wx, wy) = (&fields.u, &fields.v, &fields.a, &fields.b);
    fft.forward_with(
        |row, dst| {
            let span = row * nx..(row + 1) * nx;
            let (u, v, wx, wy) = (&u[span.clone()], &v[span.clone()], &wx[span.clone()], &wy[span]);
            for ((((d, &u), &v), &wx), &wy) in dst.iter_mut().zip(u).zip(v).zip(wx).zip(wy) {
                *d = u * wx + v * wy;
            }
        },
        band,
        |row, _, re, im| ops.mask_row(row, -1.0, re, im),
        out,
    );
}

/// Runs `f(offset, re, im)` in parallel on the dealiased part (`kx < band`) of every masked row
/// of `target`, where `offset = row·(nx/2+1)` locates the row in other spectra. The rest of
/// `target` is left untouched.
fn masked_rows<F>(ops: &Operators, target: &mut Spectrum, f: F)
where
    F: Fn(usize, &mut [f64], &mut [f64]) + Sync,
{
    let (band, kx_len) = (ops.band, ops.kx_len);
    target.re.par_chunks_mut(kx_len).zip(target.im.par_chunks_mut(kx_len)).enumerate().for_each(
        |(row, (re, im))| {
            if ops.row_in_mask[row] {
                f(row * kx_len, &mut re[..band], &mut im[..band]);
            }
        },
    );
}

/// Implicit Brinkman penalisation of `(u, v)` towards the three bodies; returns the number of
/// mask evaluations. Rows are independent, so they run in parallel.
fn penalise(
    grid: &FluidGrid,
    params: &Params,
    nodes: &Nodes,
    states: &[BodyState; 3],
    u: &mut [f64],
    v: &mut [f64],
) -> u64 {
    let (nx, dx) = (grid.nx, grid.dx);
    let (width, eta) = (params.mask_width, params.eta_ratio);
    let boxes: [Option<BodyBox>; 3] = std::array::from_fn(|b| {
        let reach = states[b].shape.extent() + MASK_CUTOFF * width;
        BodyBox::new(states[b], nodes, dx, reach)
    });
    let Some(row_lo) = boxes.iter().flatten().map(|b| b.row_lo).min() else {
        return 0;
    };
    let row_hi = boxes.iter().flatten().map(|b| b.row_hi).max().unwrap_or(row_lo);
    u[row_lo * nx..row_hi * nx]
        .par_chunks_mut(nx)
        .zip(v[row_lo * nx..row_hi * nx].par_chunks_mut(nx))
        .enumerate()
        .map(|(offset, (u_row, v_row))| {
            let row = row_lo + offset;
            let y = nodes.y[row];
            let active: [Option<&BodyBox>; 3] = std::array::from_fn(|b| {
                boxes[b].as_ref().filter(|bx| (bx.row_lo..bx.row_hi).contains(&row))
            });
            let Some(col_lo) = active.iter().flatten().map(|b| b.col_lo).min() else {
                return 0;
            };
            let col_hi = active.iter().flatten().map(|b| b.col_hi).max().unwrap_or(col_lo);
            let mut evaluations = 0;
            for col in col_lo..col_hi {
                let x = nodes.x[col];
                let (mut sum, mut chi, mut ub, mut vb) = (0.0, 0.0, 0.0, 0.0);
                for body in active.iter().flatten() {
                    if !(body.col_lo..body.col_hi).contains(&col) {
                        continue;
                    }
                    let shape = &body.state.shape;
                    let d = [x - body.state.position[0], y - body.state.position[1]];
                    let z = shape.signed_distance(d) / width;
                    if z < MASK_CUTOFF {
                        evaluations += 1;
                        let c = 0.5 * (1.0 - math::tanh(z));
                        // The body's material velocity: its centre's plus its deformation's.
                        let [du, dv] = shape.deformation_velocity(d);
                        sum += c;
                        ub += c * (body.state.velocity[0] + du);
                        vb += c * (body.state.velocity[1] + dv);
                        if c > chi {
                            chi = c;
                        }
                    }
                }
                if chi > 0.0 {
                    let a = chi / eta;
                    let norm = if sum > CHI_SUM_FLOOR { sum } else { CHI_SUM_FLOOR };
                    let (ub, vb) = (ub / norm, vb / norm);
                    u_row[col] = (u_row[col] + a * ub) / (1.0 + a);
                    v_row[col] = (v_row[col] + a * vb) / (1.0 + a);
                }
            }
            evaluations
        })
        .sum()
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)] // index loops mirror the formulas of the oracles
mod tests {
    use super::*;
    use crate::ember::orbit::Shape;
    use sha2::{Digest, Sha256};

    /// A small test configuration on a coarse grid: fat, resolved discs at low Reynolds number.
    fn test_config() -> FluidConfig {
        FluidConfig {
            rows: 32,
            box_margin: 0.6,
            reynolds: 40.0,
            reference_speed: 1.0,
            body_radius: 0.3,
            cfl: 0.5,
            max_dt: 0.02,
            brinkman_eta_ratio: 0.01,
            mask_width: 0.1,
            sponge_rate: 25.0,
            sponge_pad: 0.06,
            hyperviscosity: 144.0,
            max_snapshot_interval: 0.01,
            max_snapshot_travel: 0.5,
        }
    }

    /// The 48×32 box (`dx = 0.1`) around a 1.5-aspect canvas.
    fn test_grid() -> FluidGrid {
        FluidGrid::for_canvas(1.5, 32, 0.6).expect("valid")
    }

    /// Bodies in uniform straight-line motion.
    struct Linear {
        /// Positions at `t = 0`.
        start: [[f64; 2]; 3],
        /// Constant velocities.
        velocity: [[f64; 2]; 3],
    }

    impl BodyMotion for Linear {
        fn bodies_at(&self, t: f64) -> [BodyState; 3] {
            std::array::from_fn(|b| BodyState {
                position: [
                    self.start[b][0] + self.velocity[b][0] * t,
                    self.start[b][1] + self.velocity[b][1] * t,
                ],
                velocity: self.velocity[b],
                shape: Shape::disc(test_config().body_radius),
            })
        }

        fn speed_bound(&self, _t: f64) -> f64 {
            self.velocity.iter().map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt()).fold(0.0, f64::max)
        }
    }

    /// Three bodies on a circle, a third of a turn apart: discs (`aspect` 1), or ellipses of the
    /// disc's area with their long axis along the path, turning with it.
    struct Circling {
        /// Circle radius.
        radius: f64,
        /// Angular speed.
        rate: f64,
        /// Axis ratio of the bodies.
        aspect: f64,
    }

    impl Circling {
        fn discs(radius: f64, rate: f64) -> Self {
            Self { radius, rate, aspect: 1.0 }
        }
    }

    impl BodyMotion for Circling {
        fn bodies_at(&self, t: f64) -> [BodyState; 3] {
            std::array::from_fn(|b| {
                let (s, c) = math::sin_cos(self.rate * t + TAU * b as f64 / 3.0);
                let speed = self.radius * self.rate;
                let r = test_config().body_radius;
                let shape = if self.aspect == 1.0 {
                    Shape::disc(r)
                } else {
                    let k = self.aspect.sqrt();
                    Shape { semi: [r * k, r / k], axis: [-s, c], spin: self.rate, strain: 0.0 }
                };
                BodyState {
                    position: [self.radius * c, self.radius * s],
                    velocity: [-speed * s, speed * c],
                    shape,
                }
            })
        }

        fn speed_bound(&self, _t: f64) -> f64 {
            let r = test_config().body_radius;
            self.radius * self.rate + self.rate * r * self.aspect.sqrt()
        }
    }

    /// One body parked at the origin with the given (fixed) outline; the others are absent.
    struct Parked(Shape);

    impl BodyMotion for Parked {
        fn bodies_at(&self, _t: f64) -> [BodyState; 3] {
            let far = |x: f64, y: f64| BodyState {
                position: [x, y],
                velocity: [0.0; 2],
                shape: Shape::disc(test_config().body_radius),
            };
            [
                BodyState { position: [0.0; 2], velocity: [0.0; 2], shape: self.0 },
                far(1e7, 1e7),
                far(-1e7, 1e7),
            ]
        }

        fn speed_bound(&self, _t: f64) -> f64 {
            self.0.deformation_speed()
        }
    }

    /// Bodies parked far outside the box: the fluid feels no body at all.
    fn absent() -> Linear {
        Linear { start: [[1e7, 1e7], [-1e7, 1e7], [1e7, -1e7]], velocity: [[0.0; 2]; 3] }
    }

    /// One disc at `(-0.6, 0)` moving along +x at unit speed; the other two are absent.
    fn one_mover() -> Linear {
        Linear {
            start: [[-0.6, 0.0], [1e7, 1e7], [-1e7, 1e7]],
            velocity: [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        }
    }

    /// Three discs moving in different directions.
    fn three_movers() -> Linear {
        Linear {
            start: [[-0.8, -0.3], [0.7, 0.4], [0.1, -0.6]],
            velocity: [[1.0, 0.3], [-0.8, -0.2], [0.2, 0.9]],
        }
    }

    /// `f64` copy of an `f32` field.
    fn widen(values: &[f32]) -> Vec<f64> {
        values.iter().map(|&x| f64::from(x)).collect()
    }

    /// Maximum absolute value.
    fn peak(values: &[f64]) -> f64 {
        values.iter().fold(0.0, |m: f64, x| m.max(x.abs()))
    }

    /// Adds the half spectrum (numpy `rfft2` layout) of the node field
    /// `amplitude·cos(θ + phase)`, `θ = 2π·(a·j/nx + b·i/ny)` at row `i`, column `j`, to `spec`.
    ///
    /// The full transform is `amplitude·(nx·ny/2)·e^{±i·phase}` at `(ky, kx) = ±(b, a)`; only the
    /// `kx ≥ 0` half is stored, so a mode with `a < 0` is stored as its conjugate and a mode with
    /// `a = 0` fills both of its entries in the `kx = 0` column. In world units
    /// `θ = k·(x - x₀)` with `k = (2πa/lx, 2πb/ly)` and the box corner `x₀`.
    fn add_cosine(
        spec: &mut Spectrum,
        grid: &FluidGrid,
        mode: [i64; 2],
        amplitude: f64,
        phase: f64,
    ) {
        let (nx, ny) = (grid.nx as i64, grid.ny as i64);
        assert!(2 * mode[0].abs() < nx && 2 * mode[1].abs() < ny, "no Nyquist or aliased modes");
        let [a, b] = mode;
        let (a, b, phase) =
            if a < 0 || (a == 0 && b < 0) { (-a, -b, -phase) } else { (a, b, phase) };
        let half = amplitude * (grid.nx * grid.ny) as f64 / 2.0;
        let (s, c) = math::sin_cos(phase);
        let kxl = grid.nx / 2 + 1;
        let mut add = |row: i64, re: f64, im: f64| {
            let at = row.rem_euclid(ny) as usize * kxl + a as usize;
            spec.re[at] += re;
            spec.im[at] += im;
        };
        add(b, half * c, half * s);
        if a == 0 {
            // The Hermitian partner e^{-i(θ + phase)} also lies in the stored kx = 0 column.
            add(-b, half * c, -(half * s));
        }
    }

    /// Largest modulus of the difference of two spectra.
    fn spectral_distance(x: &Spectrum, y: &Spectrum) -> f64 {
        let (xs, ys) = (x.re.iter().zip(&x.im), y.re.iter().zip(&y.im));
        xs.zip(ys).fold(0.0, |worst: f64, ((xr, xi), (yr, yi))| {
            let (dr, di) = (xr - yr, xi - yi);
            worst.max((dr * dr + di * di).sqrt())
        })
    }

    /// World wavevector `(2πa/lx, 2πb/ly)` of the integer mode `(a, b)`.
    fn wavevector(grid: &FluidGrid, mode: [i64; 2]) -> [f64; 2] {
        [TAU * mode[0] as f64 / grid.lx, TAU * mode[1] as f64 / grid.ly]
    }

    #[test]
    fn default_grid_is_2160_by_1536() {
        let config = crate::ember::config::EmberConfig::default().fluid;
        let grid =
            FluidGrid::for_canvas(3456.0 / 2234.0, config.rows, config.box_margin).expect("valid");
        assert_eq!((grid.nx, grid.ny), (2160, 1536));
        assert_eq!(grid.lx, 2160.0 * grid.dx);
        let small = test_grid();
        assert_eq!((small.nx, small.ny), (48, 32));
    }

    #[test]
    fn mask_truncation_is_bit_exact() {
        // Beyond (d - R)/W = 19.07 the pinned libm's tanh is exactly 1, so c = 0 already.
        let mut z = 19.07;
        while z < 40.0 {
            assert_eq!(0.5 * (1.0 - math::tanh(z)), 0.0, "z = {z}");
            z += 1e-3;
        }
        assert!(0.5 * (1.0 - math::tanh(19.0)) > 0.0);
    }

    #[test]
    fn operators_follow_the_dealiasing_rule() {
        let grid = test_grid();
        let ops = Operators::new(&grid, 0.01, 144.0);
        assert_eq!(ops.band, 16); // j < 48/3
        let rows: Vec<usize> = (0..grid.ny).filter(|&i| ops.row_in_mask[i]).collect();
        let expected: Vec<usize> = (0..11).chain(22..32).collect(); // |m| < 32/3
        assert_eq!(rows, expected);
        let kc = (2.0 / 3.0) * PI / grid.dx;
        for i in 0..grid.ny {
            for j in 0..=grid.nx / 2 {
                let inside = ops.kx[j].abs() < kc && ops.ky[i].abs() < kc;
                assert_eq!(inside, ops.row_in_mask[i] && j < ops.band, "({i}, {j})");
            }
        }
        assert!(ops.ky[grid.ny / 2] < 0.0, "the Nyquist ky is negative");
        // The hyperviscous part is tiny well inside the band and ~c_h/dx at its edge.
        let at = |i: usize, j: usize| ops.damping[i * ops.kx_len + j];
        let k2 = ops.kx[1] * ops.kx[1];
        assert!((at(0, 1) - 0.01 * k2).abs() <= 1e-12 * at(0, 1));
        let mut ratio24 = 1.0;
        for _ in 0..24 {
            ratio24 *= 15.0 / 16.0;
        }
        assert!(at(0, 15) > 0.5 * 144.0 / grid.dx * ratio24);
    }

    #[test]
    fn sponge_is_zero_on_the_canvas_and_full_at_the_box_edge() {
        let grid = test_grid();
        let solver = WakeSolver::new(grid, &test_config(), 1.5).expect("valid");
        let sponge = &solver.sponge;
        for (j, &x) in solver.nodes.x.iter().enumerate() {
            let s = sponge.sigma_x[j];
            if x.abs() <= 1.56 {
                assert_eq!(s, 0.0, "x = {x}");
            }
            if x.abs() >= 0.5 * grid.lx - 4.0 * grid.dx {
                assert_eq!(s, 25.0, "x = {x}");
            }
            assert!((0.0..=25.0).contains(&s));
        }
        for (i, &y) in solver.nodes.y.iter().enumerate() {
            if y.abs() <= 1.06 {
                assert_eq!(sponge.sigma_y[i], 0.0, "y = {y}");
            }
        }
        assert_eq!(ramp(0.5, 0.0, 1.0), 0.5);
        assert_eq!(ramp(-2.0, 0.0, 1.0), 1.0);
        assert_eq!(ramp(2.0, 1.0, 0.5), 1.0, "degenerate ramp is a step");
    }

    #[test]
    fn single_fourier_mode_decays_exactly_as_exp_minus_d_t() {
        let grid = test_grid();
        let mut config = test_config();
        config.sponge_rate = 0.0;
        let mut solver = WakeSolver::new(grid, &config, 1.5).expect("valid");
        let kxl = grid.nx / 2 + 1;
        let (i, j) = (3, 2); // ky index 3, kx index 2: a pure travelling wave cos(k·x)
        let amplitude = (grid.nx * grid.ny) as f64 / 2.0;
        solver.w_hat.re[i * kxl + j] = amplitude;
        let damping = solver.ops.damping[i * kxl + j];
        let bodies = absent();
        for target in [0.05, 0.1234, 0.3] {
            solver.advance_to(target, &bodies).expect("finite");
            let expected = amplitude * math::exp(-damping * target);
            let got = solver.w_hat.re[i * kxl + j];
            assert!(
                (got - expected).abs() <= 1e-10 * amplitude,
                "t = {target}: {got} vs {expected}"
            );
            // Every other mode stays at round-off level (a single mode is a steady Euler flow).
            let others = solver
                .w_hat
                .re
                .iter()
                .zip(&solver.w_hat.im)
                .enumerate()
                .filter(|&(index, _)| index != i * kxl + j)
                .fold(0.0, |m: f64, (_, (r, im))| m.max(r.abs()).max(im.abs()));
            assert!(others <= 1e-9 * amplitude, "leak {others}");
        }
        assert_eq!(solver.stats().mask_evaluations, 0);
    }

    #[test]
    fn advection_term_matches_the_analytic_two_mode_interaction() {
        // For ω = A·cos(θ_p + φ_p) + B·cos(θ_q + φ_q) (θ_k = k·(x - x₀)) the streamfunction is
        // ψ = A/|p|²·cos(θ_p + φ_p) + B/|q|²·cos(θ_q + φ_q), and in u·∇ω = ψ_y·ω_x - ψ_x·ω_y
        // each mode's self-interaction cancels (a single mode is a steady Euler flow), leaving
        //   u·∇ω = C·sin(θ_p + φ_p)·sin(θ_q + φ_q),   C = A·B·(p×q)·(1/|q|² - 1/|p|²),
        //        = C/2·[cos(θ_{p-q} + φ_p - φ_q) - cos(θ_{p+q} + φ_p + φ_q)]
        // with p×q = p_x·q_y - p_y·q_x. So N = -mask·FFT(u·∇ω) holds exactly two modes, and
        // only those that pass the 2/3 rule.
        let grid = test_grid(); // 48×32: kept columns |a| < 16, kept rows |b| < 32/3
        let ops = Operators::new(&grid, 0.01, 144.0);
        let mut fft = Rfft2d::new(grid.nx, grid.ny).expect("smooth");
        let n = grid.len();
        let mut fields =
            Fields { u: vec![0.0; n], v: vec![0.0; n], a: vec![0.0; n], b: vec![0.0; n] };
        let cases: [([i64; 2], [i64; 2]); 5] = [
            ([3, 1], [1, 2]),  // p ± q = (2, -1), (4, 3)
            ([2, 1], [2, -3]), // p - q = (0, 4) lies in the kx = 0 column
            ([1, -2], [5, 3]), // p - q = (-4, -5) is stored as its conjugate at (4, 5)
            ([9, 0], [8, 1]),  // p + q = (17, 1): column outside the band, masked
            ([3, 6], [2, 5]),  // p + q = (5, 11): row outside the band, masked
        ];
        let kept = |m: [i64; 2]| 3 * m[0].abs() < grid.nx as i64 && 3 * m[1].abs() < grid.ny as i64;
        for (case, &(p_mode, q_mode)) in cases.iter().enumerate() {
            let shift = case as f64;
            let (amp_p, amp_q) = (1.3 + 0.2 * shift, -0.7 - 0.1 * shift);
            let (phase_p, phase_q) = (0.4 + shift, -1.1 + 0.5 * shift);
            let mut w_hat = fft.zero_spectrum();
            add_cosine(&mut w_hat, &grid, p_mode, amp_p, phase_p);
            add_cosine(&mut w_hat, &grid, q_mode, amp_q, phase_q);
            let mut out = fft.zero_spectrum();
            nonlinear(&mut fft, &ops, &w_hat, &mut out, &mut fields);

            let (p, q) = (wavevector(&grid, p_mode), wavevector(&grid, q_mode));
            let (p2, q2) = (p[0] * p[0] + p[1] * p[1], q[0] * q[0] + q[1] * q[1]);
            let c = amp_p * amp_q * (p[0] * q[1] - p[1] * q[0]) * (1.0 / q2 - 1.0 / p2);
            let difference = [p_mode[0] - q_mode[0], p_mode[1] - q_mode[1]];
            let sum = [p_mode[0] + q_mode[0], p_mode[1] + q_mode[1]];
            assert!(kept(p_mode) && kept(q_mode) && kept(difference));
            let mut expected = fft.zero_spectrum();
            add_cosine(&mut expected, &grid, difference, -0.5 * c, phase_p - phase_q);
            if kept(sum) {
                add_cosine(&mut expected, &grid, sum, 0.5 * c, phase_p + phase_q);
            }
            // Round-off scale: |u|·|∇ω| bound times the transform gain nx·ny.
            let (kp, kq) = (p2.sqrt(), q2.sqrt());
            let scale = (amp_p.abs() / kp + amp_q.abs() / kq)
                * (amp_p.abs() * kp + amp_q.abs() * kq)
                * n as f64;
            let signal = 0.25 * c.abs() * n as f64;
            assert!(signal > 1e-3 * scale, "case {case}: the interaction is not negligible");
            let error = spectral_distance(&out, &expected);
            assert!(error <= 1e-13 * scale, "case {case}: error {error:e}, signal {signal:e}");
        }
    }

    #[test]
    fn if_rk4_converges_at_fourth_order_in_time() {
        // The pure advection–diffusion step (no bodies, no forcing split; default viscosity and
        // hyperviscosity) from a smooth multi-mode field with velocities of order 1: the error
        // against a 128-step reference must fall 2⁴ = 16-fold per halving of h (the reference's
        // own error is ~8⁴ times smaller than the finest run's). A wrong stage input or weight
        // makes the scheme lower order or inconsistent; the sign of N, which this test cannot
        // see, is pinned by the two-mode oracle above.
        let grid = test_grid();
        let config = test_config();
        let modes: [([i64; 2], f64, f64); 6] = [
            ([1, 0], 3.0, 0.3),
            ([0, 1], -2.5, 1.2),
            ([1, 1], 2.0, -0.7),
            ([2, -1], 1.5, 2.1),
            ([1, -2], -1.8, 0.9),
            ([3, 1], 1.2, -1.6),
        ];
        let duration = 0.2;
        let run = |steps: u32| -> Spectrum {
            let mut solver = WakeSolver::new(grid, &config, 1.5).expect("valid");
            for &(mode, amplitude, phase) in &modes {
                add_cosine(&mut solver.w_hat, &grid, mode, amplitude, phase);
            }
            let h = duration / f64::from(steps);
            for _ in 0..steps {
                solver.update_factors(h);
                solver.ifrk4(h);
            }
            solver.w_hat
        };
        let reference = run(128);
        let len = reference.re.len();
        let zero = Spectrum { re: vec![0.0; len], im: vec![0.0; len] };
        let size = spectral_distance(&reference, &zero);
        let errors = [4, 8, 16].map(|steps| spectral_distance(&run(steps), &reference));
        // Relative errors ~2·10⁻⁴ … 6·10⁻⁷: far above round-off, in the asymptotic range.
        assert!(errors[0] < 1e-3 * size && errors[2] > 1e-9 * size, "{errors:?} vs {size:e}");
        for pair in errors.windows(2) {
            let ratio = pair[0] / pair[1];
            assert!((16.0 / 1.5..=16.0 * 1.5).contains(&ratio), "ratio {ratio} ({errors:?})");
        }
    }

    #[test]
    fn lamb_oseen_dipole_translates_and_spreads_as_predicted() {
        // Two Lamb–Oseen vortices ω = ±Γ/(π·r²)·exp(-|x - x±|²/r²) at x± = (x_c, ±d/2). Alone,
        // each is an exact axisymmetric Navier–Stokes solution: u·∇ω ≡ 0 and the core spreads
        // as r² → r² + 4νt, which is also the growth of its vorticity-weighted second moment
        // ⟨|x - X|²⟩ = r². Together they form a dipole: the upper (counter-clockwise, ω > 0)
        // and the lower (clockwise) vortex carry each other along +x. By the mean-value property
        // of the partner's harmonic velocity, the centroid of a well-separated core moves with
        // the velocity at its centre, which in the doubly periodic box (the partner and all its
        // images; the vortex's own lattice cancels by symmetry) is the lattice sum
        //   U = Γ/(lx·ly)·Σ_{k≠0} ky·sin(ky·d)/|k|²·exp(-|k|²r²/4),
        // ~21% below the unbounded-plane Γ/(2πd) = 0.5 for this 6.4-wide box.
        let (nx, ny, dx) = (64, 64, 0.1);
        let grid = FluidGrid { nx, ny, dx, lx: nx as f64 * dx, ly: ny as f64 * dx };
        let mut config = test_config();
        config.sponge_rate = 0.0;
        let nu = config.reference_speed * (2.0 * config.body_radius) / config.reynolds;
        let mut solver = WakeSolver::new(grid, &config, 1.0).expect("valid");
        let (gamma, d, core, start) = (TAU * 0.8, 1.6, 0.3, -0.5); // centres on nodes
        let mut lattice = 0.0;
        for a in -40_i32..=40 {
            for b in -40_i32..=40 {
                let (kx, ky) = (TAU * f64::from(a) / grid.lx, TAU * f64::from(b) / grid.ly);
                let k2 = kx * kx + ky * ky;
                if k2 > 0.0 {
                    lattice += ky * math::sin(ky * d) / k2 * math::exp(-k2 * core * core / 4.0);
                }
            }
        }
        let lattice = gamma / (grid.lx * grid.ly) * lattice;

        let mut omega = vec![0.0; grid.len()];
        for i in 0..ny {
            for j in 0..nx {
                let (x, y) = (solver.nodes.x[j], solver.nodes.y[i]);
                let blob = |yc: f64| {
                    let r2 = (x - start) * (x - start) + (y - yc) * (y - yc);
                    gamma / (PI * core * core) * math::exp(-r2 / (core * core))
                };
                omega[i * nx + j] = blob(0.5 * d) - blob(-0.5 * d);
            }
        }
        let (fft, ops) = (&mut solver.fft, &solver.ops);
        let omega_ref = &omega;
        fft.forward_with(
            |row, dst| dst.copy_from_slice(&omega_ref[row * nx..(row + 1) * nx]),
            ops.band,
            |row, _, re, im| ops.mask_row(row, 1.0, re, im),
            &mut solver.w_hat,
        );
        // Circulation, centroid and second moment of the upper vortex (rows y ≥ 0), plus the
        // snapshot they were measured on.
        let measure = |solver: &mut WakeSolver| {
            let mut snap = Snapshot::zeros(&grid);
            solver.snapshot_into(&mut snap);
            let cell = |i: usize, j: usize| f64::from(snap.w[i * nx + j]) * dx * dx;
            let (mut circulation, mut mx, mut my) = (0.0, 0.0, 0.0);
            for i in ny / 2..ny {
                for j in 0..nx {
                    circulation += cell(i, j);
                    mx += cell(i, j) * solver.nodes.x[j];
                    my += cell(i, j) * solver.nodes.y[i];
                }
            }
            let (cx, cy) = (mx / circulation, my / circulation);
            let mut moment = 0.0;
            for i in ny / 2..ny {
                for j in 0..nx {
                    let (rx, ry) = (solver.nodes.x[j] - cx, solver.nodes.y[i] - cy);
                    moment += cell(i, j) * (rx * rx + ry * ry);
                }
            }
            ([circulation, cx, cy, moment / circulation], snap)
        };
        let ([gamma0, x0, y0, spread0], snap) = measure(&mut solver);
        // The spectral velocity at the upper centre (-0.5, 0.8) is the lattice sum (up to the
        // dealiased Gaussian tail, exp(-(k_c·r)²/4) ~ 5·10⁻⁵).
        let centre = (ny / 2 + 8) * nx + (nx / 2 - 5);
        let induced = f64::from(snap.u[centre]);
        assert!((induced - lattice).abs() <= 1e-4 * lattice, "{induced} vs lattice {lattice}");
        assert!((gamma0 - gamma).abs() <= 1e-3 * gamma, "circulation {gamma0}");

        let bodies = absent();
        solver.advance_to(0.1, &bodies).expect("finite");
        let ([_, x1, ..], _) = measure(&mut solver);
        let early = (x1 - x0) / 0.1;
        assert!((early - lattice).abs() <= 0.01 * lattice, "speed {early} vs {lattice}");

        let duration = 0.6;
        solver.advance_to(duration, &bodies).expect("finite");
        let ([gamma2, x2, y2, spread2], _) = measure(&mut solver);
        let mean = (x2 - x0) / duration;
        assert!((mean - lattice).abs() <= 0.03 * lattice, "mean speed {mean} vs {lattice}");
        assert!((y2 - y0).abs() <= 5e-3, "the dipole drifts sideways: {y0} → {y2}");
        assert!((gamma2 - gamma0).abs() <= 2e-3 * gamma0, "circulation {gamma0} → {gamma2}");
        // Viscous spreading 4νt = 0.036 plus ~0.004 from the partner's strain deforming the
        // cores (measured in a nearly inviscid run, Re = 4000; grid- and step-converged).
        let growth = (spread2 - spread0) / (4.0 * nu * duration);
        assert!((1.0..=1.25).contains(&growth), "core growth {growth} × 4νt");
    }

    #[test]
    fn advance_to_lands_exactly_and_handles_repeated_targets() {
        let grid = test_grid();
        let mut solver = WakeSolver::new(grid, &test_config(), 1.5).expect("valid");
        let bodies = one_mover();
        let targets = [0.0, 0.013, 0.013, 0.05, 0.1 + 0.2, 0.3000000000000001, 0.41];
        let mut steps = 0;
        for &target in &targets {
            solver.advance_to(target, &bodies).expect("finite");
            assert_eq!(solver.time(), target);
            if target == 0.013 && steps > 0 {
                assert_eq!(solver.stats().steps, steps, "a repeated target takes no step");
            }
            steps = solver.stats().steps;
        }
        let stats = solver.stats();
        assert!(stats.min_dt > 0.0 && stats.max_dt <= 0.02 && stats.min_dt <= stats.max_dt);
        assert!(stats.max_speed > 0.5, "the disc drags the fluid ({})", stats.max_speed);
        assert!(stats.mask_evaluations > 0);
        assert!(matches!(solver.advance_to(0.2, &bodies), Err(EmberError::InvalidSchedule { .. })));
        assert!(matches!(
            solver.advance_to(f64::NAN, &bodies),
            Err(EmberError::InvalidSchedule { .. })
        ));
    }

    #[test]
    fn step_sizes_follow_the_cfl_rule() {
        let grid = test_grid();
        let mut config = test_config();
        config.max_dt = 1.0;
        let mut solver = WakeSolver::new(grid, &config, 1.5).expect("valid");
        // Fluid at rest, body speed bound 2: h_cfl = 0.5·0.1/2 = 0.025 for the first step.
        let bodies = Linear { start: [[1e7; 2]; 3], velocity: [[2.0, 0.0], [0.0; 2], [0.0; 2]] };
        solver.advance_to(0.025 * (1.0 + 5e-10), &bodies).expect("finite");
        assert_eq!(solver.stats().steps, 1, "within the landing slack: one step");
        let mut solver = WakeSolver::new(grid, &config, 1.5).expect("valid");
        solver.advance_to(0.04, &bodies).expect("finite");
        let stats = solver.stats();
        assert_eq!(stats.steps, 2, "rem < 2·h_cfl: two balanced halves");
        assert_eq!((stats.min_dt, stats.max_dt), (0.02, 0.02));
    }

    #[test]
    fn snapshot_velocity_is_divergence_free_and_its_curl_is_the_vorticity() {
        let grid = test_grid();
        let mut solver = WakeSolver::new(grid, &test_config(), 1.5).expect("valid");
        solver.advance_to(0.2, &three_movers()).expect("finite");
        let mut snap = Snapshot::zeros(&grid);
        solver.snapshot_into(&mut snap);
        assert_eq!(snap.time, 0.2);
        let mut fft = Rfft2d::new(grid.nx, grid.ny).expect("smooth");
        let (mut su, mut sv, mut sw) =
            (fft.zero_spectrum(), fft.zero_spectrum(), fft.zero_spectrum());
        fft.forward(&widen(&snap.u), &mut su);
        fft.forward(&widen(&snap.v), &mut sv);
        fft.forward(&widen(&snap.w), &mut sw);
        let ops = &solver.ops;
        let kxl = grid.nx / 2 + 1;
        let (mut div, mut curl_err, mut scale): (f64, f64, f64) = (0.0, 0.0, 0.0);
        for i in 0..grid.ny {
            for j in 0..kxl {
                let at = i * kxl + j;
                let (kx, ky) = (ops.kx[j], ops.ky[i]);
                // div = i·(kx·U + ky·V); curl = i·(kx·V - ky·U).
                let (dr, di) =
                    (-(kx * su.im[at] + ky * sv.im[at]), kx * su.re[at] + ky * sv.re[at]);
                let (cr, ci) =
                    (-(kx * sv.im[at] - ky * su.im[at]), kx * sv.re[at] - ky * su.re[at]);
                let norm = |a: f64, b: f64| (a * a + b * b).sqrt();
                div = div.max(norm(dr, di));
                if at != 0 {
                    // The sponge changes the mean vorticity (the k = 0 mode), which the
                    // velocity cannot carry; every other mode must match.
                    curl_err = curl_err.max(norm(cr - sw.re[at], ci - sw.im[at]));
                }
                scale = scale.max(norm(sw.re[at], sw.im[at]));
            }
        }
        assert!(scale > 1.0, "the discs stirred the fluid");
        assert!(div <= 1e-6 * scale, "divergence {div:e} vs {scale:e}");
        assert!(curl_err <= 1e-6 * scale, "curl mismatch {curl_err:e} vs {scale:e}");
        // The snapshot vorticity is the dealiased state.
        assert!(peak(&widen(&snap.w)) > 0.0);
    }

    #[test]
    fn a_moving_disc_sheds_a_mirror_antisymmetric_wake() {
        let grid = test_grid();
        // Without the sponge the scheme is exactly mirror-equivariant up to round-off. (With it,
        // the numpy Nyquist convention ky = -π/dx makes the Nyquist-row part of ∂u/∂y
        // mirror-symmetric instead of antisymmetric; the sponge multiplies that y-uniform
        // checkerboard and aliases a little of it into kept modes. On this coarse grid with a
        // 1.4-cell sponge ramp that is a 1e-3 relative asymmetry; the ramp spans ~51 cells on the
        // 720×512 grid (3e-8) and ~106 on a 1440×1024 grid (1e-10); see the
        // module documentation.)
        let mut config = test_config();
        config.sponge_rate = 0.0;
        let mut solver = WakeSolver::new(grid, &config, 1.5).expect("valid");
        solver.advance_to(0.4, &one_mover()).expect("finite");
        let mut snap = Snapshot::zeros(&grid);
        solver.snapshot_into(&mut snap);
        let (nx, ny) = (grid.nx, grid.ny);
        let w = widen(&snap.w);
        let (u, v) = (widen(&snap.u), widen(&snap.v));
        let scale = peak(&w);
        assert!(scale > 1.0, "a wake formed ({scale})");
        // Mirror y → -y maps row i to row (ny - i) mod ny: ω and v flip sign, u does not.
        let mut asym: f64 = 0.0;
        for i in 0..ny {
            let m = (ny - i) % ny;
            for j in 0..nx {
                asym = asym
                    .max((w[i * nx + j] + w[m * nx + j]).abs() / scale)
                    .max((v[i * nx + j] + v[m * nx + j]).abs())
                    .max((u[i * nx + j] - u[m * nx + j]).abs());
            }
        }
        assert!(asym <= 1e-6, "mirror asymmetry {asym:e}");
        // The disc drags fluid along +x, so its boundary layer spins counter-clockwise above
        // (∂u/∂y < 0, ω > 0) and clockwise below: the vorticity peak lies above the axis, the
        // trough below, and the upper half carries positive circulation.
        let (mut top, mut bottom, mut upper) = ((f64::MIN, 0), (f64::MAX, 0), 0.0);
        for i in 0..ny {
            for j in 0..nx {
                let value = w[i * nx + j];
                if value > top.0 {
                    top = (value, i);
                }
                if value < bottom.0 {
                    bottom = (value, i);
                }
                if i > ny / 2 {
                    upper += value;
                }
            }
        }
        assert!(top.1 > ny / 2 && bottom.1 < ny / 2, "peak row {} trough row {}", top.1, bottom.1);
        assert!(upper > 0.0, "upper-half circulation {upper}");
        let disc_x = -0.6 + 0.4;
        let col = ((disc_x / grid.dx).round() as isize + (nx / 2) as isize) as usize;
        // Inside the disc the fluid moves with it.
        let centre = (ny / 2) * nx + col;
        assert!(
            (u[centre] - 1.0).abs() < 0.2 && v[centre].abs() < 1e-5,
            "{} {}",
            u[centre],
            v[centre]
        );
    }

    #[test]
    fn sponge_damps_vorticity_outside_the_canvas() {
        let grid = test_grid();
        let mut solver = WakeSolver::new(grid, &test_config(), 1.5).expect("valid");
        // Two identical weak Gaussian blobs: one on the canvas, one in the sponge band.
        let (nx, ny) = (grid.nx, grid.ny);
        let blob = |x0: f64, y0: f64, x: f64, y: f64| {
            let r2 = (x - x0) * (x - x0) + (y - y0) * (y - y0);
            1e-3 * math::exp(-r2 / 0.02)
        };
        let mut omega = vec![0.0; nx * ny];
        for i in 0..ny {
            for j in 0..nx {
                let (x, y) = (solver.nodes.x[j], solver.nodes.y[i]);
                omega[i * nx + j] = blob(0.0, 0.0, x, y) + blob(2.1, 0.0, x, y);
            }
        }
        let (fft, ops) = (&mut solver.fft, &solver.ops);
        let omega_ref = &omega;
        fft.forward_with(
            |row, dst| dst.copy_from_slice(&omega_ref[row * nx..(row + 1) * nx]),
            ops.band,
            |row, _, re, im| ops.mask_row(row, 1.0, re, im),
            &mut solver.w_hat,
        );
        solver.advance_to(0.2, &absent()).expect("finite");
        let mut snap = Snapshot::zeros(&grid);
        solver.snapshot_into(&mut snap);
        let w = widen(&snap.w);
        let near = |x0: f64| {
            let mut m: f64 = 0.0;
            for i in 0..ny {
                for j in 0..nx {
                    let (x, y) = (solver.nodes.x[j], solver.nodes.y[i]);
                    if (x - x0).abs() < 0.3 && y.abs() < 0.3 {
                        m = m.max(w[i * nx + j].abs());
                    }
                }
            }
            m
        };
        let (inside, outside) = (near(0.0), near(2.1));
        assert!(inside > 0.5e-3, "the canvas blob only diffuses ({inside})");
        assert!(outside < 0.1 * inside, "the sponge absorbs ({outside} vs {inside})");
    }

    #[test]
    fn non_finite_body_motion_is_reported() {
        let grid = test_grid();
        let mut solver = WakeSolver::new(grid, &test_config(), 1.5).expect("valid");
        let mut bodies = one_mover();
        bodies.velocity[0][1] = f64::NAN;
        let err = solver.advance_to(0.1, &bodies).expect_err("NaN body");
        assert!(matches!(err, EmberError::NonFinite { .. }), "{err}");
        let mut config = test_config();
        config.mask_width = 0.0;
        assert!(WakeSolver::new(grid, &config, 1.5).is_err());
        assert!(WakeSolver::new(grid, &test_config(), f64::NAN).is_err());
    }

    /// Runs 20 steps of three moving discs in a rayon pool of `threads` threads.
    fn run_in_pool(threads: usize) -> (Spectrum, Snapshot, FluidStats) {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("pool");
        pool.install(|| {
            let grid = FluidGrid::for_canvas(1.5, 40, 0.6).expect("valid");
            let mut solver = WakeSolver::new(grid, &test_config(), 1.5).expect("valid");
            let bodies = three_movers();
            let mut t = 0.0;
            while solver.stats().steps < 20 {
                t += 0.0137;
                solver.advance_to(t, &bodies).expect("finite");
            }
            let mut snap = Snapshot::zeros(&grid);
            solver.snapshot_into(&mut snap);
            (solver.w_hat.clone(), snap, solver.stats())
        })
    }

    #[test]
    fn results_do_not_depend_on_the_thread_count() {
        let (w1, s1, st1) = run_in_pool(1);
        let (w3, s3, st3) = run_in_pool(3);
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let bits32 = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&w1.re), bits(&w3.re));
        assert_eq!(bits(&w1.im), bits(&w3.im));
        assert_eq!(bits32(&s1.u), bits32(&s3.u));
        assert_eq!(bits32(&s1.v), bits32(&s3.v));
        assert_eq!(bits32(&s1.w), bits32(&s3.w));
        assert_eq!(st1, st3);
        assert!(st1.steps >= 20);
    }

    /// SHA-256 of the snapshot (time, then `u`, `v`, `w` as little-endian `f32` bits) after a
    /// fixed run of three circling discs on the 48×32 box. Identical on every architecture; a
    /// change means the solver's arithmetic changed and downstream golden hashes must be
    /// re-blessed.
    const GOLDEN_SNAPSHOT: &str =
        "7b53a0914f4d9c821535a0ccf9b160b63377e7c0725dac75ff20b0da4bfc938a";

    /// SHA-256 of the snapshot of a fixed run of `bodies` on the 48×32 box, and the stats.
    fn snapshot_digest(bodies: &Circling) -> (String, FluidStats) {
        let grid = test_grid();
        let mut solver = WakeSolver::new(grid, &test_config(), 1.5).expect("valid");
        for s in 1..=4 {
            solver.advance_to(0.1 * f64::from(s), bodies).expect("finite");
        }
        let mut snap = Snapshot::zeros(&grid);
        solver.snapshot_into(&mut snap);
        let mut hasher = Sha256::new();
        hasher.update(snap.time.to_bits().to_le_bytes());
        for field in [&snap.u, &snap.v, &snap.w] {
            for value in field {
                hasher.update(value.to_bits().to_le_bytes());
            }
        }
        (hex::encode(hasher.finalize()), solver.stats())
    }

    #[test]
    fn snapshot_golden_hash() {
        let (digest, stats) = snapshot_digest(&Circling::discs(0.6, 2.0));
        assert_eq!(digest, GOLDEN_SNAPSHOT, "stats {stats:?}");
    }

    /// Like [`GOLDEN_SNAPSHOT`], for turning ellipses of aspect 2.25: exercises the elliptical
    /// signed distance and the deformation velocity of the penalisation.
    const GOLDEN_STRETCHED_SNAPSHOT: &str =
        "b776dbb1e10227930ed33e3c5270d22c2a375476ff90bd5c2688ef7d3f4e1472";

    #[test]
    fn stretched_snapshot_golden_hash() {
        let (digest, stats) = snapshot_digest(&Circling { radius: 0.6, rate: 2.0, aspect: 2.25 });
        assert_eq!(digest, GOLDEN_STRETCHED_SNAPSHOT, "stats {stats:?}");
    }

    /// A disc turning in place does not stir still water (its material does not rotate: the
    /// deformation flow of a disc is zero), while an ellipse turning in place drives the water
    /// inside it with its irrotational deformation flow.
    #[test]
    fn a_turning_ellipse_stirs_the_water_with_its_deformation_flow() {
        let grid = test_grid();
        let turning = |semi: [f64; 2]| Shape { semi, axis: [1.0, 0.0], spin: 3.0, strain: 0.0 };
        let mut solver = WakeSolver::new(grid, &test_config(), 1.5).expect("valid");
        solver.advance_to(0.3, &Parked(turning([0.3, 0.3]))).expect("finite");
        let mut snap = Snapshot::zeros(&grid);
        solver.snapshot_into(&mut snap);
        assert!(snap.u.iter().chain(&snap.v).chain(&snap.w).all(|&x| x == 0.0), "still water");

        let ellipse = turning([0.6, 0.15]);
        let mut solver = WakeSolver::new(grid, &test_config(), 1.5).expect("valid");
        solver.advance_to(0.3, &Parked(ellipse)).expect("finite");
        solver.snapshot_into(&mut snap);
        // Deep inside (at least two mask widths from the outline) the water moves with the body.
        let (mut worst, mut peak, mut deep) = (0.0f64, 0.0f64, 0);
        for i in 0..grid.ny {
            for j in 0..grid.nx {
                let p = [grid.x0() + j as f64 * grid.dx, grid.y0() + i as f64 * grid.dx];
                if ellipse.signed_distance(p) > -0.2 {
                    continue;
                }
                let [du, dv] = ellipse.deformation_velocity(p);
                let n = i * grid.nx + j;
                let (u, v) = (f64::from(snap.u[n]), f64::from(snap.v[n]));
                let (eu, ev) = (u - du, v - dv);
                worst = worst.max((eu * eu + ev * ev).sqrt());
                peak = peak.max((du * du + dv * dv).sqrt());
                deep += 1;
            }
        }
        assert!(deep >= 4 && peak > 0.5, "{deep} deep nodes, peak {peak}");
        assert!(worst < 0.1 * peak, "worst {worst} vs peak {peak}");
    }

    /// Release-mode timing of full solver steps at production sizes (run with
    /// `cargo test --release --lib ember::fluid -- --ignored --nocapture`).
    #[test]
    #[ignore = "timing report; run explicitly in release mode"]
    fn step_timing() {
        let aspect = 3456.0 / 2234.0;
        let config = crate::ember::EmberConfig::default().fluid;
        for rows in [config.rows, 1024, 512] {
            let grid = FluidGrid::for_canvas(aspect, rows, config.box_margin).expect("valid");
            let mut solver = WakeSolver::new(grid, &config, aspect).expect("valid");
            let bodies = Circling::discs(0.7, 1.0 / 0.7);
            let mut t = 0.0;
            for _ in 0..3 {
                t += 1e-3;
                solver.advance_to(t, &bodies).expect("finite");
            }
            let before = solver.stats().steps;
            let start = std::time::Instant::now();
            t += 0.03;
            solver.advance_to(t, &bodies).expect("finite");
            let elapsed = start.elapsed().as_secs_f64();
            let steps = solver.stats().steps - before;
            let mut snap = Snapshot::zeros(&grid);
            let snap_start = std::time::Instant::now();
            solver.snapshot_into(&mut snap);
            let snap_ms = snap_start.elapsed().as_secs_f64() * 1e3;
            println!(
                "fluid {}x{}: {:.2} ms/step over {steps} steps (umax {:.2}); snapshot {snap_ms:.2} ms",
                grid.nx,
                grid.ny,
                elapsed * 1e3 / steps as f64,
                solver.stats().max_speed,
            );
        }
    }
}
