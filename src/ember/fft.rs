//! Deterministic fast Fourier transforms for the ember fluid solver.
//!
//! # Definitions
//!
//! The one-dimensional transform of length `n` is the unnormalised DFT
//!
//! ```text
//! X[k] = Σ_{j<n} x[j]·ω_n^{jk},   ω_n = e^{-2πi/n}   (forward)
//! x[j] = Σ_{k<n} X[k]·ω_n^{-jk}                        (inverse, unnormalised)
//! ```
//!
//! [`Rfft2d`] transforms a real `ny × nx` row-major field to its half spectrum with numpy's
//! `rfft2` layout and conventions:
//!
//! ```text
//! X[ky][kx] = Σ_{i<ny} Σ_{j<nx} x[i][j]·e^{-2πi(kx·j/nx + ky·i/ny)},   kx ∈ 0..=nx/2, ky ∈ 0..ny
//! ```
//!
//! stored at `ky·(nx/2+1) + kx`. The inverse has numpy `irfft2` semantics: a complex inverse
//! transform along `ky`, then a Hermitian (complex-to-real) inverse along `kx` in which the
//! imaginary parts of the self-conjugate `kx = 0` and `kx = nx/2` entries are ignored, and a single
//! multiplication by `1/(nx·ny)`.
//!
//! # Algorithm
//!
//! One-dimensional transforms use the self-sorting Stockham formulation of the mixed-radix
//! decimation-in-frequency FFT for every 5-smooth length (`n = 2^a·3^b·5^c`), with radix-4, -2,
//! -3 and -5 butterflies (radix 4 first, then at most one radix 2, then 3s, then 5s). A stage of
//! radix `p` at stride `s` with `m = n/(s·p)` groups maps
//!
//! ```text
//! y[q + s·(p·k + t)] = ω_{n/s}^{k·t} · Σ_{r<p} x[q + s·(k + m·r)]·ω_p^{r·t}
//! ```
//!
//! for `q < s`, `k < m`, `t < p`, ping-ponging between two buffers so the output ends in natural
//! order without a bit-reversal pass. The inverse transform is the forward transform with real
//! and imaginary parts exchanged on input and output (`IDFT(z) = swap(DFT(swap(z)))`), which is
//! exact: it only relabels buffers.
//!
//! Twiddle factors `ω_n^e` come from [`math::sin_cos`] on the smallest arc the symmetries of `n`
//! allow (the first octant when `8 | n`) and are extended to the full circle by the exact
//! symmetries `sin ↔ cos`, sign flips and conjugation, with the special angles (`0`, `π/6`, `π/4`,
//! `π/3`, `π/2`, …) set to their correctly rounded values, so the tables are portable bit
//! patterns with exact symmetries. The radix-3 and radix-5 butterflies use correctly rounded
//! literals for `sin(π/3)`, `cos(2π/5)`, `cos(4π/5)`, `sin(2π/5)` and `sin(4π/5)`.
//!
//! The kernels are batched: [`LANES`] independent transforms are stored side by side as
//! `[n][LANES]` structure-of-arrays blocks (real and imaginary parts in separate arrays) and every
//! butterfly loops over the lanes with identical scalar arithmetic, which LLVM vectorises. IEEE
//! vector lanes compute exactly what scalar code computes (Rust never contracts `a·b + c` into a
//! fused multiply-add and never reassociates), so a lane's result does not depend on the batch it
//! was computed in; a unit test checks this against single-lane transforms.
//!
//! [`Rfft2d`] packs two real rows into one complex row (`z = x_{2p} + i·x_{2p+1}`) and separates
//! them with the Hermitian identities `X_{2p}[k] = (Z[k] + Z*[n-k])/2`,
//! `X_{2p+1}[k] = (Z[k] - Z*[n-k])/(2i)`; the column transforms run on blocks of [`LANES`]
//! adjacent wavenumber columns.
//!
//! # Determinism
//!
//! Every output value is computed by a fixed sequence of IEEE-754 additions, subtractions and
//! multiplications that depends only on the transform sizes, never on the thread count or on
//! which thread ran a block: rayon distributes independent row blocks and column blocks, and the
//! per-task scratch buffers are fully overwritten before they are read.

use std::fmt;
use std::sync::{Mutex, PoisonError};

use rayon::prelude::*;

use super::error::{EmberError, EmberResult};
use super::math;

/// Independent transforms computed side by side by the batched kernels: 4 × f64 is two NEON or
/// one AVX2 register per operand and keeps a 1440-point batch (with its ping-pong buffer) within
/// the L1/L2 caches. Measured faster than 2 or 8 on the 1440×1024 grid. Results do not depend on
/// this value (lanes are independent), only speed does.
pub(crate) const LANES: usize = 4;

/// One element of a batch: the same element index of [`LANES`] independent transforms.
type Lane = [f64; LANES];

/// `√3/2 = sin(π/3)`, correctly rounded.
const SQRT3_2: f64 = 0.8660254037844386;
/// `cos(2π/5)`, correctly rounded.
const COS_2PI_5: f64 = 0.30901699437494745;
/// `cos(4π/5)`, correctly rounded.
const COS_4PI_5: f64 = -0.8090169943749475;
/// `sin(2π/5)`, correctly rounded.
const SIN_2PI_5: f64 = 0.9510565162951535;
/// `sin(4π/5)`, correctly rounded.
const SIN_4PI_5: f64 = 0.5877852522924731;

/// Whether `n ≥ 1` has no prime factor other than 2, 3 and 5.
pub(crate) fn is_smooth(n: usize) -> bool {
    if n == 0 {
        return false;
    }
    let mut m = n;
    for p in [2, 3, 5] {
        while m.is_multiple_of(p) {
            m /= p;
        }
    }
    m == 1
}

/// Smallest even 5-smooth integer `≥ n` (and `≥ 2`).
pub(crate) fn next_smooth_even(n: usize) -> usize {
    let mut m = n.max(2);
    while !m.is_multiple_of(2) || !is_smooth(m) {
        m += 1;
    }
    m
}

/// `(cos 2πj/n, sin 2πj/n)` for `n ≥ 1`, bit-portable.
///
/// The angle is folded into the first octant by exact symmetries (conjugation, `θ → π - θ`,
/// `θ → π/2 - θ`) before [`math::sin_cos`] is called, and the angles whose sine or cosine is a
/// simple algebraic number get their correctly rounded values.
fn unit_root(j: usize, n: usize) -> (f64, f64) {
    let j = j % n;
    if 2 * j > n {
        let (c, s) = unit_root(n - j, n);
        return (c, -s);
    }
    if n.is_multiple_of(2) && 4 * j > n {
        let (c, s) = unit_root(n / 2 - j, n);
        return (-c, s);
    }
    if n.is_multiple_of(4) && 8 * j > n {
        let (c, s) = unit_root(n / 4 - j, n);
        return (s, c);
    }
    if j == 0 {
        (1.0, 0.0)
    } else if 8 * j == n {
        (std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2)
    } else if 12 * j == n {
        (SQRT3_2, 0.5)
    } else if 6 * j == n {
        (0.5, SQRT3_2)
    } else if 4 * j == n {
        (0.0, 1.0)
    } else if 3 * j == n {
        (-0.5, SQRT3_2)
    } else {
        let (s, c) = math::sin_cos(std::f64::consts::TAU * j as f64 / n as f64);
        (c, s)
    }
}

/// Butterfly radix of one stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Radix {
    /// 2-point butterflies.
    Two,
    /// 3-point butterflies.
    Three,
    /// 4-point butterflies.
    Four,
    /// 5-point butterflies.
    Five,
}

impl Radix {
    /// Points per butterfly.
    fn points(self) -> usize {
        match self {
            Self::Two => 2,
            Self::Three => 3,
            Self::Four => 4,
            Self::Five => 5,
        }
    }
}

/// The radices of a 5-smooth `n` in stage order (4s, at most one 2, 3s, 5s); empty for `n = 1`.
fn factorize(n: usize) -> Vec<Radix> {
    let mut m = n;
    let mut radices = Vec::new();
    while m.is_multiple_of(4) {
        radices.push(Radix::Four);
        m /= 4;
    }
    if m.is_multiple_of(2) {
        radices.push(Radix::Two);
        m /= 2;
    }
    while m.is_multiple_of(3) {
        radices.push(Radix::Three);
        m /= 3;
    }
    while m.is_multiple_of(5) {
        radices.push(Radix::Five);
        m /= 5;
    }
    debug_assert_eq!(m, 1, "factorize needs a 5-smooth length");
    radices
}

/// One Stockham stage: radix `p`, stride `s` and `m = n/(s·p)` twiddle groups.
#[derive(Clone, Debug)]
struct Stage {
    /// Butterfly radix `p`.
    radix: Radix,
    /// Twiddle groups `m`.
    m: usize,
    /// Stride `s` (product of the previous stages' radices).
    s: usize,
    /// `Re ω_n^{k·t·s}` at `k·(p-1) + t - 1` for `k < m`, `1 ≤ t < p`.
    tw_re: Vec<f64>,
    /// `Im ω_n^{k·t·s}` (same layout).
    tw_im: Vec<f64>,
}

/// Plan of the batched complex transform of one length.
#[derive(Clone, Debug)]
struct Fft1d {
    /// Transform length.
    n: usize,
    /// Stages in execution order (empty for `n = 1`).
    stages: Vec<Stage>,
}

impl Fft1d {
    /// Plans transforms of length `n` (5-smooth, `≥ 1`).
    fn new(n: usize) -> EmberResult<Self> {
        if !is_smooth(n) {
            return Err(EmberError::UnsupportedGrid {
                nx: n,
                ny: 1,
                reason: "FFT lengths must be 5-smooth and >= 1".into(),
            });
        }
        let roots: Vec<(f64, f64)> = (0..n).map(|j| unit_root(j, n)).collect();
        let mut stages = Vec::new();
        let mut s = 1;
        for radix in factorize(n) {
            let p = radix.points();
            let m = n / (s * p);
            let mut tw_re = Vec::with_capacity(m * (p - 1));
            let mut tw_im = Vec::with_capacity(m * (p - 1));
            for k in 0..m {
                for t in 1..p {
                    // k·t·s < m·p·s = n, so the index needs no reduction.
                    let (c, sn) = roots[k * t * s];
                    tw_re.push(c);
                    tw_im.push(-sn);
                }
            }
            stages.push(Stage { radix, m, s, tw_re, tw_im });
            s *= p;
        }
        Ok(Self { n, stages })
    }

    /// Forward transforms of `B` independent sequences stored as `[n][B]` blocks.
    ///
    /// Reads `src` (not modified), writes the result to `dst` and uses `scratch` as the ping-pong
    /// buffer. All slices must hold at least `n` elements; only the first `n` are touched.
    /// Inverse (unnormalised) transforms are obtained by exchanging the real and imaginary
    /// slices of both `src` and `dst`.
    fn run<const B: usize>(
        &self,
        src: (&[[f64; B]], &[[f64; B]]),
        dst: (&mut [[f64; B]], &mut [[f64; B]]),
        scratch: (&mut [[f64; B]], &mut [[f64; B]]),
    ) {
        let n = self.n;
        let (src_re, src_im) = (&src.0[..n], &src.1[..n]);
        let (dst_re, dst_im) = (&mut dst.0[..n], &mut dst.1[..n]);
        let (scr_re, scr_im) = (&mut scratch.0[..n], &mut scratch.1[..n]);
        let count = self.stages.len();
        if count == 0 {
            dst_re.copy_from_slice(src_re);
            dst_im.copy_from_slice(src_im);
            return;
        }
        for (index, stage) in self.stages.iter().enumerate() {
            // The last stage writes `dst`; earlier stages alternate so that holds.
            let into_dst = (count - 1 - index).is_multiple_of(2);
            match (index == 0, into_dst) {
                (true, true) => stage.apply(src_re, src_im, dst_re, dst_im),
                (true, false) => stage.apply(src_re, src_im, scr_re, scr_im),
                (false, true) => stage.apply(scr_re, scr_im, dst_re, dst_im),
                (false, false) => stage.apply(dst_re, dst_im, scr_re, scr_im),
            }
        }
    }
}

/// `P` consecutive sub-slices of length `len` from the front of `slice`.
fn parts<T, const P: usize>(slice: &[T], len: usize) -> [&[T]; P] {
    std::array::from_fn(|index| &slice[index * len..(index + 1) * len])
}

/// `P` consecutive mutable sub-slices of length `len` from the front of `slice`.
fn parts_mut<T, const P: usize>(slice: &mut [T], len: usize) -> [&mut [T]; P] {
    let mut rest = slice;
    std::array::from_fn(|_| {
        let (head, tail) = std::mem::take(&mut rest).split_at_mut(len);
        rest = tail;
        head
    })
}

/// `(a_r + i·a_i)·(w_r + i·w_i)` with the fixed operation order of every twiddle product.
#[inline(always)]
fn cmul(ar: f64, ai: f64, wr: f64, wi: f64) -> (f64, f64) {
    (ar * wr - ai * wi, ar * wi + ai * wr)
}

impl Stage {
    /// Runs this stage from `x` into `y` (both `n` long).
    fn apply<const B: usize>(
        &self,
        xr: &[[f64; B]],
        xi: &[[f64; B]],
        yr: &mut [[f64; B]],
        yi: &mut [[f64; B]],
    ) {
        match self.radix {
            Radix::Two => {
                self.stage::<B, 2>(xr, xi, yr, yi, butterfly2::<B, false>, butterfly2::<B, true>);
            }
            Radix::Three => {
                self.stage::<B, 3>(xr, xi, yr, yi, butterfly3::<B, false>, butterfly3::<B, true>);
            }
            Radix::Four => {
                self.stage::<B, 4>(xr, xi, yr, yi, butterfly4::<B, false>, butterfly4::<B, true>);
            }
            Radix::Five => {
                self.stage::<B, 5>(xr, xi, yr, yi, butterfly5::<B, false>, butterfly5::<B, true>);
            }
        }
    }

    /// Splits the buffers of one radix-`P` stage into its `m` twiddle groups and runs `plain`
    /// (group 0, unit twiddles) or `twiddled` on each.
    fn stage<const B: usize, const P: usize>(
        &self,
        xr: &[[f64; B]],
        xi: &[[f64; B]],
        yr: &mut [[f64; B]],
        yi: &mut [[f64; B]],
        plain: fn(Group<'_, B, P>),
        twiddled: fn(Group<'_, B, P>),
    ) {
        let (m, s) = (self.m, self.s);
        let block = m * s;
        let in_re: [&[[f64; B]]; P] = parts(xr, block);
        let in_im: [&[[f64; B]]; P] = parts(xi, block);
        let groups =
            yr[..P * block].chunks_exact_mut(P * s).zip(yi[..P * block].chunks_exact_mut(P * s));
        for (k, (out_re, out_im)) in groups.enumerate() {
            let span = k * s..(k + 1) * s;
            let group = Group {
                in_re: in_re.map(|part| &part[span.clone()]),
                in_im: in_im.map(|part| &part[span.clone()]),
                out_re: parts_mut(out_re, s),
                out_im: parts_mut(out_im, s),
                tw_re: &self.tw_re[k * (P - 1)..(k + 1) * (P - 1)],
                tw_im: &self.tw_im[k * (P - 1)..(k + 1) * (P - 1)],
            };
            if k == 0 {
                plain(group);
            } else {
                twiddled(group);
            }
        }
    }
}

/// One twiddle group of a radix-`P` stage: `P` input and `P` output runs of `s` batch elements.
struct Group<'a, const B: usize, const P: usize> {
    /// Real parts of the inputs `x[q + s·(k + m·r)]`, `r < P`, `q < s`.
    in_re: [&'a [[f64; B]]; P],
    /// Imaginary parts of the inputs.
    in_im: [&'a [[f64; B]]; P],
    /// Real parts of the outputs `y[q + s·(P·k + t)]`, `t < P`, `q < s`.
    out_re: [&'a mut [[f64; B]]; P],
    /// Imaginary parts of the outputs.
    out_im: [&'a mut [[f64; B]]; P],
    /// Real parts of `ω^{k·t·s}`, `t = 1..P`.
    tw_re: &'a [f64],
    /// Imaginary parts of `ω^{k·t·s}`.
    tw_im: &'a [f64],
}

/// Radix-2 butterflies: `y0 = a0 + a1`, `y1 = (a0 - a1)·w1`.
#[allow(clippy::needless_range_loop)] // explicit lane loops are what LLVM vectorises
fn butterfly2<const B: usize, const TW: bool>(g: Group<'_, B, 2>) {
    let [y0r, y1r] = g.out_re;
    let [y0i, y1i] = g.out_im;
    let s = y0r.len();
    let (a0r, a1r) = (&g.in_re[0][..s], &g.in_re[1][..s]);
    let (a0i, a1i) = (&g.in_im[0][..s], &g.in_im[1][..s]);
    let (y0i, y1r, y1i) = (&mut y0i[..s], &mut y1r[..s], &mut y1i[..s]);
    let (w1r, w1i) = if TW { (g.tw_re[0], g.tw_im[0]) } else { (1.0, 0.0) };
    for q in 0..s {
        for l in 0..B {
            let (pr, pi) = (a0r[q][l], a0i[q][l]);
            let (qr, qi) = (a1r[q][l], a1i[q][l]);
            y0r[q][l] = pr + qr;
            y0i[q][l] = pi + qi;
            let (dr, di) = (pr - qr, pi - qi);
            let (dr, di) = if TW { cmul(dr, di, w1r, w1i) } else { (dr, di) };
            y1r[q][l] = dr;
            y1i[q][l] = di;
        }
    }
}

/// Radix-3 butterflies with `ω_3 = -1/2 - i·√3/2`:
/// `b0 = a0 + (a1 + a2)`, `b1,2 = (a0 - (a1 + a2)/2) ∓ i·(√3/2)·(a1 - a2)`.
#[allow(clippy::needless_range_loop)] // explicit lane loops are what LLVM vectorises
fn butterfly3<const B: usize, const TW: bool>(g: Group<'_, B, 3>) {
    let [y0r, y1r, y2r] = g.out_re;
    let [y0i, y1i, y2i] = g.out_im;
    let s = y0r.len();
    let (a0r, a1r, a2r) = (&g.in_re[0][..s], &g.in_re[1][..s], &g.in_re[2][..s]);
    let (a0i, a1i, a2i) = (&g.in_im[0][..s], &g.in_im[1][..s], &g.in_im[2][..s]);
    let (y0i, y1r, y1i, y2r, y2i) =
        (&mut y0i[..s], &mut y1r[..s], &mut y1i[..s], &mut y2r[..s], &mut y2i[..s]);
    let (w1r, w1i, w2r, w2i) =
        if TW { (g.tw_re[0], g.tw_im[0], g.tw_re[1], g.tw_im[1]) } else { (1.0, 0.0, 1.0, 0.0) };
    for q in 0..s {
        for l in 0..B {
            let (x0r, x0i) = (a0r[q][l], a0i[q][l]);
            let (sr, si) = (a1r[q][l] + a2r[q][l], a1i[q][l] + a2i[q][l]);
            let (dr, di) = (a1r[q][l] - a2r[q][l], a1i[q][l] - a2i[q][l]);
            y0r[q][l] = x0r + sr;
            y0i[q][l] = x0i + si;
            let (mr, mi) = (x0r - 0.5 * sr, x0i - 0.5 * si);
            let (nr, ni) = (SQRT3_2 * dr, SQRT3_2 * di);
            let (b1r, b1i) = (mr + ni, mi - nr);
            let (b2r, b2i) = (mr - ni, mi + nr);
            let (b1r, b1i) = if TW { cmul(b1r, b1i, w1r, w1i) } else { (b1r, b1i) };
            let (b2r, b2i) = if TW { cmul(b2r, b2i, w2r, w2i) } else { (b2r, b2i) };
            y1r[q][l] = b1r;
            y1i[q][l] = b1i;
            y2r[q][l] = b2r;
            y2i[q][l] = b2i;
        }
    }
}

/// Radix-4 butterflies with `ω_4 = -i`: `t0,1 = a0 ± a2`, `t2,3 = a1 ± a3`,
/// `b0,2 = t0 ± t2`, `b1,3 = t1 ∓ i·t3`.
#[allow(clippy::needless_range_loop)] // explicit lane loops are what LLVM vectorises
fn butterfly4<const B: usize, const TW: bool>(g: Group<'_, B, 4>) {
    let [y0r, y1r, y2r, y3r] = g.out_re;
    let [y0i, y1i, y2i, y3i] = g.out_im;
    let s = y0r.len();
    let [a0r, a1r, a2r, a3r] = g.in_re.map(|x| &x[..s]);
    let [a0i, a1i, a2i, a3i] = g.in_im.map(|x| &x[..s]);
    let (y0i, y1r, y1i, y2r, y2i, y3r, y3i) = (
        &mut y0i[..s],
        &mut y1r[..s],
        &mut y1i[..s],
        &mut y2r[..s],
        &mut y2i[..s],
        &mut y3r[..s],
        &mut y3i[..s],
    );
    let (w1r, w1i, w2r, w2i, w3r, w3i) = if TW {
        (g.tw_re[0], g.tw_im[0], g.tw_re[1], g.tw_im[1], g.tw_re[2], g.tw_im[2])
    } else {
        (1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
    };
    for q in 0..s {
        for l in 0..B {
            let (t0r, t0i) = (a0r[q][l] + a2r[q][l], a0i[q][l] + a2i[q][l]);
            let (t1r, t1i) = (a0r[q][l] - a2r[q][l], a0i[q][l] - a2i[q][l]);
            let (t2r, t2i) = (a1r[q][l] + a3r[q][l], a1i[q][l] + a3i[q][l]);
            let (t3r, t3i) = (a1r[q][l] - a3r[q][l], a1i[q][l] - a3i[q][l]);
            y0r[q][l] = t0r + t2r;
            y0i[q][l] = t0i + t2i;
            let (b1r, b1i) = (t1r + t3i, t1i - t3r);
            let (b2r, b2i) = (t0r - t2r, t0i - t2i);
            let (b3r, b3i) = (t1r - t3i, t1i + t3r);
            let (b1r, b1i) = if TW { cmul(b1r, b1i, w1r, w1i) } else { (b1r, b1i) };
            let (b2r, b2i) = if TW { cmul(b2r, b2i, w2r, w2i) } else { (b2r, b2i) };
            let (b3r, b3i) = if TW { cmul(b3r, b3i, w3r, w3i) } else { (b3r, b3i) };
            y1r[q][l] = b1r;
            y1i[q][l] = b1i;
            y2r[q][l] = b2r;
            y2i[q][l] = b2i;
            y3r[q][l] = b3r;
            y3i[q][l] = b3i;
        }
    }
}

/// Radix-5 butterflies with `c1,2 = cos(2π/5), cos(4π/5)` and `s1,2 = sin(2π/5), sin(4π/5)`:
/// `m1 = a0 + c1·(a1+a4) + c2·(a2+a3)`, `m2 = a0 + c2·(a1+a4) + c1·(a2+a3)`,
/// `n1 = s1·(a1-a4) + s2·(a2-a3)`, `n2 = s2·(a1-a4) - s1·(a2-a3)`,
/// `b1,4 = m1 ∓ i·n1`, `b2,3 = m2 ∓ i·n2`.
#[allow(clippy::needless_range_loop)] // explicit lane loops are what LLVM vectorises
fn butterfly5<const B: usize, const TW: bool>(g: Group<'_, B, 5>) {
    let [y0r, y1r, y2r, y3r, y4r] = g.out_re;
    let [y0i, y1i, y2i, y3i, y4i] = g.out_im;
    let s = y0r.len();
    let [a0r, a1r, a2r, a3r, a4r] = g.in_re.map(|x| &x[..s]);
    let [a0i, a1i, a2i, a3i, a4i] = g.in_im.map(|x| &x[..s]);
    let (y0i, y1r, y1i, y2r, y2i) =
        (&mut y0i[..s], &mut y1r[..s], &mut y1i[..s], &mut y2r[..s], &mut y2i[..s]);
    let (y3r, y3i, y4r, y4i) = (&mut y3r[..s], &mut y3i[..s], &mut y4r[..s], &mut y4i[..s]);
    let w: [(f64, f64); 4] =
        if TW { std::array::from_fn(|t| (g.tw_re[t], g.tw_im[t])) } else { [(1.0, 0.0); 4] };
    for q in 0..s {
        for l in 0..B {
            let (x0r, x0i) = (a0r[q][l], a0i[q][l]);
            let (s14r, s14i) = (a1r[q][l] + a4r[q][l], a1i[q][l] + a4i[q][l]);
            let (d14r, d14i) = (a1r[q][l] - a4r[q][l], a1i[q][l] - a4i[q][l]);
            let (s23r, s23i) = (a2r[q][l] + a3r[q][l], a2i[q][l] + a3i[q][l]);
            let (d23r, d23i) = (a2r[q][l] - a3r[q][l], a2i[q][l] - a3i[q][l]);
            y0r[q][l] = x0r + s14r + s23r;
            y0i[q][l] = x0i + s14i + s23i;
            let (m1r, m1i) = (
                x0r + COS_2PI_5 * s14r + COS_4PI_5 * s23r,
                x0i + COS_2PI_5 * s14i + COS_4PI_5 * s23i,
            );
            let (m2r, m2i) = (
                x0r + COS_4PI_5 * s14r + COS_2PI_5 * s23r,
                x0i + COS_4PI_5 * s14i + COS_2PI_5 * s23i,
            );
            let (n1r, n1i) =
                (SIN_2PI_5 * d14r + SIN_4PI_5 * d23r, SIN_2PI_5 * d14i + SIN_4PI_5 * d23i);
            let (n2r, n2i) =
                (SIN_4PI_5 * d14r - SIN_2PI_5 * d23r, SIN_4PI_5 * d14i - SIN_2PI_5 * d23i);
            let b = [
                (m1r + n1i, m1i - n1r),
                (m2r + n2i, m2i - n2r),
                (m2r - n2i, m2i + n2r),
                (m1r - n1i, m1i + n1r),
            ];
            let b =
                if TW { std::array::from_fn(|t| cmul(b[t].0, b[t].1, w[t].0, w[t].1)) } else { b };
            y1r[q][l] = b[0].0;
            y1i[q][l] = b[0].1;
            y2r[q][l] = b[1].0;
            y2i[q][l] = b[1].1;
            y3r[q][l] = b[2].0;
            y3i[q][l] = b[2].1;
            y4r[q][l] = b[3].0;
            y4i[q][l] = b[3].1;
        }
    }
}

/// Split-complex half spectrum of a real `ny × nx` field: `ny` rows of `nx/2 + 1` wavenumbers,
/// row-major (`index = ky_row·(nx/2+1) + kx`), numpy `rfft2` layout.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Spectrum {
    /// Real parts.
    pub re: Vec<f64>,
    /// Imaginary parts.
    pub im: Vec<f64>,
}

/// Per-task scratch of the 2-D transforms: three `[n][LANES]` complex batches and a real row.
#[derive(Default)]
struct Workspace {
    /// Gathered input batch (real parts).
    a_re: Vec<Lane>,
    /// Gathered input batch (imaginary parts).
    a_im: Vec<Lane>,
    /// Stockham ping-pong buffer (real parts).
    b_re: Vec<Lane>,
    /// Stockham ping-pong buffer (imaginary parts).
    b_im: Vec<Lane>,
    /// Transform output batch (real parts).
    c_re: Vec<Lane>,
    /// Transform output batch (imaginary parts).
    c_im: Vec<Lane>,
    /// One real row produced by a forward source.
    row: Vec<f64>,
}

impl Workspace {
    /// Scratch for batches of up to `len` elements and rows of `row_len` values.
    fn new(len: usize, row_len: usize) -> Self {
        let batch = || vec![[0.0; LANES]; len];
        Self {
            a_re: batch(),
            a_im: batch(),
            b_re: batch(),
            b_im: batch(),
            c_re: batch(),
            c_im: batch(),
            row: vec![0.0; row_len],
        }
    }
}

/// Reusable per-task scratch. Tasks check a [`Workspace`] out and return it when done, so after
/// the first transforms no call allocates. The contents never influence results: every task
/// overwrites the parts of its workspace it reads.
struct WorkspacePool {
    /// Batch length of new workspaces.
    len: usize,
    /// Row length of new workspaces.
    row_len: usize,
    /// Idle workspaces.
    free: Mutex<Vec<Workspace>>,
}

impl WorkspacePool {
    /// A pool that creates workspaces for batches of `len` and rows of `row_len`.
    fn new(len: usize, row_len: usize) -> Self {
        Self { len, row_len, free: Mutex::new(Vec::new()) }
    }

    /// An idle workspace (a new one if none is idle), returned to the pool on drop.
    fn checkout(&self) -> Checkout<'_> {
        let idle = self.free.lock().unwrap_or_else(PoisonError::into_inner).pop();
        Checkout {
            pool: self,
            workspace: idle.unwrap_or_else(|| Workspace::new(self.len, self.row_len)),
        }
    }
}

/// A checked-out [`Workspace`].
struct Checkout<'a> {
    /// Owner the workspace returns to.
    pool: &'a WorkspacePool,
    /// The workspace.
    workspace: Workspace,
}

impl Drop for Checkout<'_> {
    fn drop(&mut self) {
        let workspace = std::mem::take(&mut self.workspace);
        self.pool.free.lock().unwrap_or_else(PoisonError::into_inner).push(workspace);
    }
}

/// Real two-dimensional FFT plan for an `ny × nx` row-major field (both even and 5-smooth).
///
/// Holds the 1-D plans and all intermediate buffers, so transforms allocate nothing after the
/// first call. Methods take `&mut self` because they reuse those buffers.
pub(crate) struct Rfft2d {
    /// Columns of the real field.
    nx: usize,
    /// Rows of the real field.
    ny: usize,
    /// Wavenumbers per spectrum row (`nx/2 + 1`).
    kx_len: usize,
    /// Length-`nx` plan (packed row pairs).
    row_fft: Fft1d,
    /// Length-`ny` plan (wavenumber columns).
    col_fft: Fft1d,
    /// Column blocks of [`LANES`] wavenumbers (`⌈kx_len / LANES⌉`).
    col_blocks: usize,
    /// Row-pass output, tiled `[row block][column block][2·LANES rows]` (forward only).
    tiles_re: Vec<Lane>,
    /// Imaginary parts of `tiles_re`.
    tiles_im: Vec<Lane>,
    /// Column-pass output, `[column block][ny]`.
    cols_re: Vec<Lane>,
    /// Imaginary parts of `cols_re`.
    cols_im: Vec<Lane>,
    /// Per-task scratch.
    pool: WorkspacePool,
}

impl fmt::Debug for Rfft2d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Rfft2d").field("nx", &self.nx).field("ny", &self.ny).finish_non_exhaustive()
    }
}

impl Rfft2d {
    /// Plans transforms of `ny × nx` real fields.
    pub(crate) fn new(nx: usize, ny: usize) -> EmberResult<Self> {
        if nx < 2
            || ny < 2
            || !nx.is_multiple_of(2)
            || !ny.is_multiple_of(2)
            || !is_smooth(nx)
            || !is_smooth(ny)
        {
            return Err(EmberError::UnsupportedGrid {
                nx,
                ny,
                reason: "both sizes must be even and 5-smooth".into(),
            });
        }
        let kx_len = nx / 2 + 1;
        let col_blocks = kx_len.div_ceil(LANES);
        let row_blocks = (ny / 2).div_ceil(LANES);
        let tiles = row_blocks * col_blocks * 2 * LANES;
        let cols = col_blocks * ny;
        Ok(Self {
            nx,
            ny,
            kx_len,
            row_fft: Fft1d::new(nx)?,
            col_fft: Fft1d::new(ny)?,
            col_blocks,
            tiles_re: vec![[0.0; LANES]; tiles],
            tiles_im: vec![[0.0; LANES]; tiles],
            cols_re: vec![[0.0; LANES]; cols],
            cols_im: vec![[0.0; LANES]; cols],
            pool: WorkspacePool::new(nx.max(ny), nx),
        })
    }

    /// Columns of the real field.
    pub(crate) fn nx(&self) -> usize {
        self.nx
    }

    /// Rows of the real field.
    pub(crate) fn ny(&self) -> usize {
        self.ny
    }

    /// Wavenumbers per spectrum row (`nx/2 + 1`).
    pub(crate) fn kx_len(&self) -> usize {
        self.kx_len
    }

    /// A zeroed spectrum of the right size.
    pub(crate) fn zero_spectrum(&self) -> Spectrum {
        let len = self.ny * self.kx_len;
        Spectrum { re: vec![0.0; len], im: vec![0.0; len] }
    }

    /// Unnormalised forward transform `X[ky][kx] = Σ x[i][j]·e^{-2πi(kx·j/nx + ky·i/ny)}`.
    ///
    /// # Panics
    ///
    /// If `field` is not `ny·nx` long or `out` has the wrong size.
    pub(crate) fn forward(&mut self, field: &[f64], out: &mut Spectrum) {
        assert_eq!(field.len(), self.nx * self.ny, "field size");
        let nx = self.nx;
        self.forward_with(
            |row, dst| dst.copy_from_slice(&field[row * nx..(row + 1) * nx]),
            self.kx_len,
            |_, _, _, _| {},
            out,
        );
    }

    /// Forward transform of the field whose row `i` `source(i, row)` writes into `row`
    /// (`nx` values), band-limited and mapped:
    ///
    /// * only the wavenumbers `kx < band` are computed; the entries `kx ≥ band` of `out` are set
    ///   to exactly zero (`band ≥ nx/2 + 1` computes the whole spectrum);
    /// * then `map(ky, 0, re, im)` rewrites every finished spectrum row in place (`re`, `im` are
    ///   the row's `nx/2 + 1` values).
    ///
    /// Fusing the producer and the per-mode map into the transform saves a pass over memory for
    /// each, and skipping the column transforms that a dealiasing map would discard saves up to
    /// a third of the column work. The computed entries equal those of `forward`.
    ///
    /// # Panics
    ///
    /// If `out` does not have `ny·(nx/2+1)` modes.
    pub(crate) fn forward_with<S, M>(&mut self, source: S, band: usize, map: M, out: &mut Spectrum)
    where
        S: Fn(usize, &mut [f64]) + Sync,
        M: Fn(usize, usize, &mut [f64], &mut [f64]) + Sync,
    {
        let len = self.ny * self.kx_len;
        assert!(out.re.len() == len && out.im.len() == len, "spectrum size");
        let band = band.min(self.kx_len);
        self.forward_rows(&source, band);
        self.forward_columns(band);
        self.store_rows(band, &map, out);
    }

    /// Inverse transform normalised by `1/(nx·ny)` (numpy `irfft2` semantics); `spec` is not
    /// modified.
    ///
    /// # Panics
    ///
    /// If `spec` or `field` has the wrong size.
    pub(crate) fn inverse(&mut self, spec: &Spectrum, field: &mut [f64]) {
        self.inverse_with(spec, self.kx_len, |_, _, _, _| {}, field);
    }

    /// Inverse transform of the band-limited, mapped spectrum:
    ///
    /// * the entries `kx ≥ band` of `spec` are treated as exactly zero (not read);
    /// * the columns `kx < band` are gathered in segments of up to [`LANES`] wavenumbers, and
    ///   `map(ky, kx0, re, im)` rewrites the segment `kx0..kx0 + re.len()` of row `ky` in place
    ///   before it is transformed.
    ///
    /// `spec` is not modified. The result equals `inverse` of the band-limited mapped spectrum
    /// (up to the sign of exact zeros, which never changes a non-zero value).
    ///
    /// # Panics
    ///
    /// If `spec` or `field` has the wrong size.
    pub(crate) fn inverse_with<M>(
        &mut self,
        spec: &Spectrum,
        band: usize,
        map: M,
        field: &mut [f64],
    ) where
        M: Fn(usize, usize, &mut [f64], &mut [f64]) + Sync,
    {
        let len = self.ny * self.kx_len;
        assert!(spec.re.len() == len && spec.im.len() == len, "spectrum size");
        assert_eq!(field.len(), self.nx * self.ny, "field size");
        let band = band.min(self.kx_len);
        self.inverse_columns(spec, band, &map);
        self.inverse_rows(band, field);
    }

    /// Forward row pass: packed row pairs → half-spectrum rows (`kx < band`), stored as tiles.
    fn forward_rows<S>(&mut self, source: &S, band: usize)
    where
        S: Fn(usize, &mut [f64]) + Sync,
    {
        let Self { nx, ny, row_fft, col_blocks, tiles_re, tiles_im, pool, .. } = self;
        let (nx, pairs, col_blocks) = (*nx, *ny / 2, *col_blocks);
        let (row_fft, pool) = (&*row_fft, &*pool);
        let tile_len = col_blocks * 2 * LANES;
        tiles_re
            .par_chunks_mut(tile_len)
            .zip(tiles_im.par_chunks_mut(tile_len))
            .enumerate()
            .for_each_init(
                || pool.checkout(),
                |checkout, (block, (t_re, t_im))| {
                    let ws = &mut checkout.workspace;
                    // Gather: lane l carries the pair (2p, 2p+1) as (real, imaginary) parts.
                    for lane in 0..LANES {
                        let pair = block * LANES + lane;
                        if pair < pairs {
                            source(2 * pair, &mut ws.row);
                            for (dst, &x) in ws.a_re[..nx].iter_mut().zip(&ws.row) {
                                dst[lane] = x;
                            }
                            source(2 * pair + 1, &mut ws.row);
                            for (dst, &x) in ws.a_im[..nx].iter_mut().zip(&ws.row) {
                                dst[lane] = x;
                            }
                        } else {
                            for (re, im) in ws.a_re[..nx].iter_mut().zip(&mut ws.a_im[..nx]) {
                                re[lane] = 0.0;
                                im[lane] = 0.0;
                            }
                        }
                    }
                    row_fft.run::<LANES>(
                        (&ws.a_re, &ws.a_im),
                        (&mut ws.c_re, &mut ws.c_im),
                        (&mut ws.b_re, &mut ws.b_im),
                    );
                    // Unpack Z = X_even + i·X_odd with the Hermitian identities.
                    for k in 0..band {
                        let conj = if k == 0 { 0 } else { nx - k };
                        let (zr, zi) = (&ws.c_re[k], &ws.c_im[k]);
                        let (cr, ci) = (&ws.c_re[conj], &ws.c_im[conj]);
                        let tile = (k / LANES) * 2 * LANES;
                        let col = k % LANES;
                        for l in 0..LANES {
                            t_re[tile + 2 * l][col] = 0.5 * (zr[l] + cr[l]);
                            t_im[tile + 2 * l][col] = 0.5 * (zi[l] - ci[l]);
                            t_re[tile + 2 * l + 1][col] = 0.5 * (zi[l] + ci[l]);
                            t_im[tile + 2 * l + 1][col] = 0.5 * (cr[l] - zr[l]);
                        }
                    }
                },
            );
    }

    /// Forward column pass: tiles → transformed column blocks (those holding `kx < band`).
    fn forward_columns(&mut self, band: usize) {
        let Self { ny, col_fft, col_blocks, tiles_re, tiles_im, cols_re, cols_im, pool, .. } = self;
        let (ny, col_blocks) = (*ny, *col_blocks);
        let (col_fft, pool, tiles_re, tiles_im) = (&*col_fft, &*pool, &*tiles_re, &*tiles_im);
        let used = band.div_ceil(LANES) * ny;
        cols_re[..used]
            .par_chunks_mut(ny)
            .zip(cols_im[..used].par_chunks_mut(ny))
            .enumerate()
            .for_each_init(
                || pool.checkout(),
                |checkout, (block, (c_re, c_im))| {
                    let ws = &mut checkout.workspace;
                    // Lanes at or beyond `band` hold stale tiles; zero them so every lane computes a
                    // defined value (lanes are independent, so this does not affect the others).
                    let valid = (band - block * LANES).min(LANES);
                    for ky in 0..ny {
                        let tile = ((ky / (2 * LANES)) * col_blocks + block) * 2 * LANES;
                        let index = tile + ky % (2 * LANES);
                        ws.a_re[ky] = tiles_re[index];
                        ws.a_im[ky] = tiles_im[index];
                        ws.a_re[ky][valid..].fill(0.0);
                        ws.a_im[ky][valid..].fill(0.0);
                    }
                    col_fft.run::<LANES>(
                        (&ws.a_re, &ws.a_im),
                        (c_re, c_im),
                        (&mut ws.b_re, &mut ws.b_im),
                    );
                },
            );
    }

    /// Copies the transformed column blocks (`kx < band`) into the row-major spectrum, zeroes
    /// `kx ≥ band` and applies `map`.
    fn store_rows<M>(&self, band: usize, map: &M, out: &mut Spectrum)
    where
        M: Fn(usize, usize, &mut [f64], &mut [f64]) + Sync,
    {
        let (ny, kx_len) = (self.ny, self.kx_len);
        let (cols_re, cols_im) = (&self.cols_re, &self.cols_im);
        out.re
            .par_chunks_mut(kx_len)
            .zip(out.im.par_chunks_mut(kx_len))
            .enumerate()
            .with_min_len(4)
            .for_each(|(ky, (row_re, row_im))| {
                let (live_re, dead_re) = row_re.split_at_mut(band);
                let (live_im, dead_im) = row_im.split_at_mut(band);
                let segments = live_re.chunks_mut(LANES).zip(live_im.chunks_mut(LANES));
                for (block, (seg_re, seg_im)) in segments.enumerate() {
                    let n = seg_re.len();
                    seg_re.copy_from_slice(&cols_re[block * ny + ky][..n]);
                    seg_im.copy_from_slice(&cols_im[block * ny + ky][..n]);
                }
                dead_re.fill(0.0);
                dead_im.fill(0.0);
                map(ky, 0, row_re, row_im);
            });
    }

    /// Inverse column pass: mapped spectrum columns `kx < band` → inverse-transformed column
    /// blocks.
    fn inverse_columns<M>(&mut self, spec: &Spectrum, band: usize, map: &M)
    where
        M: Fn(usize, usize, &mut [f64], &mut [f64]) + Sync,
    {
        let Self { ny, kx_len, col_fft, cols_re, cols_im, pool, .. } = self;
        let (ny, kx_len, col_fft, pool) = (*ny, *kx_len, &*col_fft, &*pool);
        let used = band.div_ceil(LANES) * ny;
        cols_re[..used]
            .par_chunks_mut(ny)
            .zip(cols_im[..used].par_chunks_mut(ny))
            .enumerate()
            .for_each_init(
                || pool.checkout(),
                |checkout, (block, (c_re, c_im))| {
                    let ws = &mut checkout.workspace;
                    let kx0 = block * LANES;
                    let n = (band - kx0).min(LANES);
                    for ky in 0..ny {
                        let at = ky * kx_len + kx0;
                        let (g_re, g_im) = (&mut ws.a_re[ky], &mut ws.a_im[ky]);
                        g_re[..n].copy_from_slice(&spec.re[at..at + n]);
                        g_im[..n].copy_from_slice(&spec.im[at..at + n]);
                        g_re[n..].fill(0.0);
                        g_im[n..].fill(0.0);
                        map(ky, kx0, &mut g_re[..n], &mut g_im[..n]);
                    }
                    // Inverse = forward with real and imaginary parts exchanged.
                    col_fft.run::<LANES>(
                        (&ws.a_im, &ws.a_re),
                        (c_im, c_re),
                        (&mut ws.b_im, &mut ws.b_re),
                    );
                },
            );
    }

    /// Inverse row pass: column blocks → Hermitian row pairs → real rows scaled by `1/(nx·ny)`.
    fn inverse_rows(&mut self, band: usize, field: &mut [f64]) {
        let Self { nx, ny, row_fft, cols_re, cols_im, pool, .. } = self;
        let (nx, ny, row_fft, pool) = (*nx, *ny, &*row_fft, &*pool);
        let (cols_re, cols_im) = (&*cols_re, &*cols_im);
        let scale = 1.0 / (nx * ny) as f64;
        field.par_chunks_mut(2 * LANES * nx).enumerate().for_each_init(
            || pool.checkout(),
            |checkout, (block, rows)| {
                let ws = &mut checkout.workspace;
                let pairs = rows.len() / (2 * nx);
                // Z[k] = Y_even[k] + i·Y_odd[k] and Z[nx-k] = conj(Y_even[k]) + i·conj(Y_odd[k]);
                // only the real parts of the self-conjugate kx = 0 and kx = nx/2 entries count,
                // and the band-limited wavenumbers are exact zeros.
                for k in 0..=nx / 2 {
                    let column = (k / LANES) * ny;
                    let col = k % LANES;
                    for l in 0..LANES {
                        let (y0r, y0i, y1r, y1i) = if l < pairs && k < band {
                            let row = column + 2 * (block * LANES + l);
                            (
                                cols_re[row][col],
                                cols_im[row][col],
                                cols_re[row + 1][col],
                                cols_im[row + 1][col],
                            )
                        } else {
                            (0.0, 0.0, 0.0, 0.0)
                        };
                        if k == 0 || 2 * k == nx {
                            ws.a_re[k][l] = y0r;
                            ws.a_im[k][l] = y1r;
                        } else {
                            ws.a_re[k][l] = y0r - y1i;
                            ws.a_im[k][l] = y0i + y1r;
                            ws.a_re[nx - k][l] = y0r + y1i;
                            ws.a_im[nx - k][l] = y1r - y0i;
                        }
                    }
                }
                row_fft.run::<LANES>(
                    (&ws.a_im, &ws.a_re),
                    (&mut ws.c_im, &mut ws.c_re),
                    (&mut ws.b_im, &mut ws.b_re),
                );
                for (l, pair) in rows.chunks_exact_mut(2 * nx).enumerate() {
                    let (even, odd) = pair.split_at_mut(nx);
                    for ((e, o), (zr, zi)) in
                        even.iter_mut().zip(odd.iter_mut()).zip(ws.c_re.iter().zip(&ws.c_im))
                    {
                        *e = zr[l] * scale;
                        *o = zi[l] * scale;
                    }
                }
            },
        );
    }
}

/// Benchmark entry points (not a supported API; used by `benches/ember_fft.rs`).
#[doc(hidden)]
pub mod bench {
    use super::{Rfft2d, Spectrum};
    use crate::ember::EmberResult;

    /// A forward + inverse round trip of a fixed pseudo-random field.
    pub struct RoundTrip {
        /// The plan.
        plan: Rfft2d,
        /// The real field (overwritten by each round trip with the same values up to round-off).
        field: Vec<f64>,
        /// Its spectrum.
        spectrum: Spectrum,
    }

    impl RoundTrip {
        /// Plans an `ny × nx` round trip.
        pub fn new(nx: usize, ny: usize) -> EmberResult<Self> {
            let plan = Rfft2d::new(nx, ny)?;
            let mut state = 0x9e37_79b9_7f4a_7c15_u64;
            let field = (0..nx * ny)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
                })
                .collect();
            let spectrum = plan.zero_spectrum();
            Ok(Self { plan, field, spectrum })
        }

        /// One forward and one inverse transform; returns a value of the result so the work
        /// cannot be optimised away.
        pub fn run(&mut self) -> f64 {
            self.plan.forward(&self.field, &mut self.spectrum);
            self.plan.inverse(&self.spectrum, &mut self.field);
            self.field[0]
        }
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)] // index loops mirror the textbook formulas of the oracles
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    /// Deterministic pseudo-random values in `[-1, 1)` (xorshift64*).
    fn noise(len: usize, seed: u64) -> Vec<f64> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                let bits = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
                (bits >> 11) as f64 / (1u64 << 52) as f64 - 1.0
            })
            .collect()
    }

    /// Naive `O(n²)` DFT with direct (unfolded) libm twiddles; `sign = -1` forward, `+1` inverse.
    fn naive_dft(re: &[f64], im: &[f64], sign: f64) -> (Vec<f64>, Vec<f64>) {
        let n = re.len();
        let roots: Vec<(f64, f64)> =
            (0..n).map(|j| math::sin_cos(std::f64::consts::TAU * j as f64 / n as f64)).collect();
        let mut out_re = vec![0.0; n];
        let mut out_im = vec![0.0; n];
        for k in 0..n {
            let (mut sr, mut si) = (0.0, 0.0);
            for j in 0..n {
                let (s, c) = roots[(j * k) % n];
                let (wr, wi) = (c, sign * s);
                sr += re[j] * wr - im[j] * wi;
                si += re[j] * wi + im[j] * wr;
            }
            out_re[k] = sr;
            out_im[k] = si;
        }
        (out_re, out_im)
    }

    /// Single-sequence forward transform through the batched kernel with `B` lanes, lane `lane`.
    fn fft_lane<const B: usize>(
        plan: &Fft1d,
        re: &[f64],
        im: &[f64],
        lane: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = plan.n;
        let mut src_re = vec![[0.0; B]; n];
        let mut src_im = vec![[0.0; B]; n];
        for j in 0..n {
            for l in 0..B {
                // Other lanes carry unrelated data, which must not influence `lane`.
                src_re[j][l] = if l == lane { re[j] } else { (j * 7 + l) as f64 };
                src_im[j][l] = if l == lane { im[j] } else { -((j + 3 * l) as f64) };
            }
        }
        let mut dst = (vec![[0.0; B]; n], vec![[0.0; B]; n]);
        let mut scr = (vec![[0.0; B]; n], vec![[0.0; B]; n]);
        plan.run::<B>((&src_re, &src_im), (&mut dst.0, &mut dst.1), (&mut scr.0, &mut scr.1));
        (dst.0.iter().map(|x| x[lane]).collect(), dst.1.iter().map(|x| x[lane]).collect())
    }

    /// `max |a - b| / max |b|` over both components.
    fn rel_err(a: (&[f64], &[f64]), b: (&[f64], &[f64])) -> f64 {
        let mut err: f64 = 0.0;
        let mut scale: f64 = 0.0;
        for (x, y) in a.0.iter().zip(b.0).chain(a.1.iter().zip(b.1)) {
            err = err.max((x - y).abs());
            scale = scale.max(y.abs());
        }
        err / scale.max(1e-300)
    }

    /// Every 5-smooth length in `1..=64` plus some larger ones.
    fn test_lengths() -> Vec<usize> {
        (1..=64).filter(|&n| is_smooth(n)).chain([90, 96, 120, 360, 1440]).collect()
    }

    #[test]
    fn smooth_sizes_are_recognised() {
        assert!(!is_smooth(0));
        assert!(is_smooth(1));
        assert!(is_smooth(1440));
        assert!(!is_smooth(1442));
        assert!(!is_smooth(7 * 16));
        assert_eq!(next_smooth_even(1439), 1440);
        assert_eq!(next_smooth_even(1), 2);
        assert_eq!(next_smooth_even(43), 48);
        assert_eq!(next_smooth_even(1024), 1024);
        assert_eq!(
            factorize(1440),
            [Radix::Four, Radix::Four, Radix::Two, Radix::Three, Radix::Three, Radix::Five]
        );
        assert!(Fft1d::new(7).is_err());
        assert!(Rfft2d::new(46, 32).is_err());
        assert!(Rfft2d::new(45, 32).is_err());
        assert!(Rfft2d::new(48, 30).is_ok());
    }

    #[test]
    fn unit_roots_are_exactly_symmetric() {
        for n in [1, 2, 3, 4, 5, 6, 8, 12, 24, 30, 90, 96, 360, 1440] {
            for j in 0..n {
                let (c, s) = unit_root(j, n);
                let (cc, sc) = unit_root(n - j, n);
                assert_eq!((c, -s), (cc, sc), "n={n} j={j}");
                if n % 2 == 0 {
                    let (ch, sh) = unit_root(j + n / 2, n);
                    assert_eq!((-c, -s), (ch, sh), "half turn n={n} j={j}");
                }
                if n % 4 == 0 {
                    let (cq, sq) = unit_root(j + n / 4, n);
                    assert_eq!((-s, c), (cq, sq), "quarter turn n={n} j={j}");
                }
                let (sd, cd) = math::sin_cos(std::f64::consts::TAU * j as f64 / n as f64);
                assert!((c - cd).abs() <= 2e-15 && (s - sd).abs() <= 2e-15, "accuracy n={n} j={j}");
            }
        }
        assert_eq!(unit_root(1, 12), (SQRT3_2, 0.5));
        assert_eq!(unit_root(1, 8).0, unit_root(1, 8).1);
        assert_eq!(unit_root(360, 1440), (0.0, 1.0));
    }

    #[test]
    fn complex_fft_matches_naive_dft_for_many_lengths() {
        for n in test_lengths() {
            let plan = Fft1d::new(n).expect("smooth");
            let re = noise(n, 11 + n as u64);
            let im = noise(n, 97 + n as u64);
            let (xr, xi) = fft_lane::<1>(&plan, &re, &im, 0);
            let (nr, ni) = naive_dft(&re, &im, -1.0);
            let err = rel_err((&xr, &xi), (&nr, &ni));
            assert!(err < 1e-12, "n = {n}: relative error {err:e}");
            // Inverse through the re/im swap.
            let (yi, yr) = fft_lane::<1>(&plan, &im, &re, 0);
            let (mr, mi) = naive_dft(&re, &im, 1.0);
            let err = rel_err((&yr, &yi), (&mr, &mi));
            assert!(err < 1e-12, "inverse n = {n}: relative error {err:e}");
        }
    }

    #[test]
    fn batch_lanes_are_bit_identical_to_single_transforms() {
        for n in [1, 6, 30, 64, 90, 360] {
            let plan = Fft1d::new(n).expect("smooth");
            let re = noise(n, 5);
            let im = noise(n, 6);
            let single = fft_lane::<1>(&plan, &re, &im, 0);
            for lane in [0, 3, LANES - 1] {
                let batched = fft_lane::<LANES>(&plan, &re, &im, lane);
                let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&batched.0), bits(&single.0), "n={n} lane={lane}");
                assert_eq!(bits(&batched.1), bits(&single.1), "n={n} lane={lane}");
            }
        }
    }

    /// Naive `rfft2` (numpy conventions).
    fn naive_rfft2(field: &[f64], nx: usize, ny: usize) -> Spectrum {
        let kxl = nx / 2 + 1;
        let mut rows_re = vec![0.0; ny * kxl];
        let mut rows_im = vec![0.0; ny * kxl];
        for i in 0..ny {
            let (r, m) = naive_dft(&field[i * nx..(i + 1) * nx], &vec![0.0; nx], -1.0);
            rows_re[i * kxl..(i + 1) * kxl].copy_from_slice(&r[..kxl]);
            rows_im[i * kxl..(i + 1) * kxl].copy_from_slice(&m[..kxl]);
        }
        let mut out = Spectrum { re: vec![0.0; ny * kxl], im: vec![0.0; ny * kxl] };
        for kx in 0..kxl {
            let col_re: Vec<f64> = (0..ny).map(|i| rows_re[i * kxl + kx]).collect();
            let col_im: Vec<f64> = (0..ny).map(|i| rows_im[i * kxl + kx]).collect();
            let (r, m) = naive_dft(&col_re, &col_im, -1.0);
            for ky in 0..ny {
                out.re[ky * kxl + kx] = r[ky];
                out.im[ky * kxl + kx] = m[ky];
            }
        }
        out
    }

    /// Naive `irfft2` (numpy conventions: complex inverse along ky, then a Hermitian inverse
    /// along kx that ignores the imaginary parts of kx = 0 and kx = nx/2).
    fn naive_irfft2(spec: &Spectrum, nx: usize, ny: usize) -> Vec<f64> {
        let kxl = nx / 2 + 1;
        let mut cols_re = vec![0.0; ny * kxl];
        let mut cols_im = vec![0.0; ny * kxl];
        for kx in 0..kxl {
            let col_re: Vec<f64> = (0..ny).map(|i| spec.re[i * kxl + kx]).collect();
            let col_im: Vec<f64> = (0..ny).map(|i| spec.im[i * kxl + kx]).collect();
            let (r, m) = naive_dft(&col_re, &col_im, 1.0);
            for i in 0..ny {
                cols_re[i * kxl + kx] = r[i] / ny as f64;
                cols_im[i * kxl + kx] = m[i] / ny as f64;
            }
        }
        let mut out = vec![0.0; nx * ny];
        for i in 0..ny {
            for j in 0..nx {
                let mut acc = cols_re[i * kxl]
                    + cols_re[i * kxl + nx / 2] * if j % 2 == 0 { 1.0 } else { -1.0 };
                for k in 1..nx / 2 {
                    let (s, c) =
                        math::sin_cos(std::f64::consts::TAU * ((j * k) % nx) as f64 / nx as f64);
                    acc += 2.0 * (cols_re[i * kxl + k] * c - cols_im[i * kxl + k] * s);
                }
                out[i * nx + j] = acc / nx as f64;
            }
        }
        out
    }

    const SIZES_2D: [(usize, usize); 6] = [(2, 2), (12, 8), (30, 20), (48, 32), (90, 64), (20, 90)];

    #[test]
    fn rfft2_matches_naive_numpy_semantics() {
        for (nx, ny) in SIZES_2D {
            let mut plan = Rfft2d::new(nx, ny).expect("smooth");
            let field = noise(nx * ny, (nx * 1000 + ny) as u64);
            let mut spec = plan.zero_spectrum();
            plan.forward(&field, &mut spec);
            let reference = naive_rfft2(&field, nx, ny);
            let err = rel_err((&spec.re, &spec.im), (&reference.re, &reference.im));
            assert!(err < 1e-12, "{nx}x{ny}: forward relative error {err:e}");
            // The self-conjugate columns come out of the row pass exactly real.
            let back = {
                let mut f = vec![0.0; nx * ny];
                plan.inverse(&spec, &mut f);
                f
            };
            let err = rel_err((&back, &back), (&field, &field));
            assert!(err < 1e-13, "{nx}x{ny}: round trip relative error {err:e}");
        }
    }

    #[test]
    fn inverse_has_irfft2_semantics_for_non_hermitian_input() {
        for (nx, ny) in SIZES_2D {
            let mut plan = Rfft2d::new(nx, ny).expect("smooth");
            let len = ny * (nx / 2 + 1);
            let spec = Spectrum { re: noise(len, 3 + nx as u64), im: noise(len, 4 + ny as u64) };
            let mut field = vec![0.0; nx * ny];
            plan.inverse(&spec, &mut field);
            let reference = naive_irfft2(&spec, nx, ny);
            let err = rel_err((&field, &field), (&reference, &reference));
            assert!(err < 1e-12, "{nx}x{ny}: inverse relative error {err:e}");
            let copy = spec.clone();
            plan.inverse(&spec, &mut field);
            assert_eq!(copy, spec, "the input spectrum is not modified");
        }
    }

    #[test]
    fn parseval_and_linearity_hold() {
        let (nx, ny) = (48, 32);
        let mut plan = Rfft2d::new(nx, ny).expect("smooth");
        let a = noise(nx * ny, 1);
        let b = noise(nx * ny, 2);
        let mut sa = plan.zero_spectrum();
        let mut sb = plan.zero_spectrum();
        plan.forward(&a, &mut sa);
        plan.forward(&b, &mut sb);
        // Σ x² = (1/(nx·ny)) Σ_{all k} |X|², the half spectrum counting interior kx twice.
        let energy: f64 = a.iter().map(|x| x * x).sum();
        let kxl = nx / 2 + 1;
        let mut spectral = 0.0;
        for ky in 0..ny {
            for kx in 0..kxl {
                let weight = if kx == 0 || kx == nx / 2 { 1.0 } else { 2.0 };
                let i = ky * kxl + kx;
                spectral += weight * (sa.re[i] * sa.re[i] + sa.im[i] * sa.im[i]);
            }
        }
        spectral /= (nx * ny) as f64;
        assert!((energy - spectral).abs() <= 1e-12 * energy, "{energy} vs {spectral}");
        // Linearity: F(2a - 3b) = 2F(a) - 3F(b).
        let mix: Vec<f64> = a.iter().zip(&b).map(|(x, y)| 2.0 * x - 3.0 * y).collect();
        let mut sm = plan.zero_spectrum();
        plan.forward(&mix, &mut sm);
        let expected_re: Vec<f64> =
            sa.re.iter().zip(&sb.re).map(|(x, y)| 2.0 * x - 3.0 * y).collect();
        let expected_im: Vec<f64> =
            sa.im.iter().zip(&sb.im).map(|(x, y)| 2.0 * x - 3.0 * y).collect();
        let err = rel_err((&sm.re, &sm.im), (&expected_re, &expected_im));
        assert!(err < 1e-13, "linearity error {err:e}");
    }

    #[test]
    fn fused_source_and_maps_equal_separate_passes() {
        let (nx, ny) = (30, 20);
        let mut plan = Rfft2d::new(nx, ny).expect("smooth");
        let a = noise(nx * ny, 8);
        let b = noise(nx * ny, 9);
        let product: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
        let mut expected = plan.zero_spectrum();
        plan.forward(&product, &mut expected);
        for (re, im) in expected.re.iter_mut().zip(&mut expected.im) {
            (*re, *im) = (-*im, *re);
        }
        let mut fused = plan.zero_spectrum();
        plan.forward_with(
            |row, dst| {
                for (j, d) in dst.iter_mut().enumerate() {
                    *d = a[row * nx + j] * b[row * nx + j];
                }
            },
            usize::MAX,
            |_, _, re, im| {
                for (r, i) in re.iter_mut().zip(im.iter_mut()) {
                    (*r, *i) = (-*i, *r);
                }
            },
            &mut fused,
        );
        assert_eq!(fused, expected);

        let mut scaled = expected.clone();
        let kxl = nx / 2 + 1;
        for (index, (re, im)) in scaled.re.iter_mut().zip(&mut scaled.im).enumerate() {
            let factor = (index / kxl) as f64 + 0.25 * (index % kxl) as f64;
            *re *= factor;
            *im *= factor;
        }
        let mut plain = vec![0.0; nx * ny];
        plan.inverse(&scaled, &mut plain);
        let mut mapped = vec![0.0; nx * ny];
        plan.inverse_with(
            &expected,
            kxl,
            |ky, kx0, re, im| {
                for (l, (r, i)) in re.iter_mut().zip(im.iter_mut()).enumerate() {
                    let factor = ky as f64 + 0.25 * (kx0 + l) as f64;
                    *r *= factor;
                    *i *= factor;
                }
            },
            &mut mapped,
        );
        assert_eq!(plain, mapped);
    }

    #[test]
    fn band_limited_transforms_equal_full_transforms_of_band_limited_data() {
        for (nx, ny) in [(48, 32), (90, 64), (30, 20)] {
            let mut plan = Rfft2d::new(nx, ny).expect("smooth");
            let kxl = nx / 2 + 1;
            for band in [1, 7, 8, 9, nx.div_ceil(3), kxl - 1] {
                // Forward: the computed band equals the full transform, the rest is zero.
                let field = noise(nx * ny, (band * 31 + nx) as u64);
                let mut full = plan.zero_spectrum();
                plan.forward(&field, &mut full);
                let mut limited = Spectrum { re: vec![7.0; ny * kxl], im: vec![7.0; ny * kxl] };
                plan.forward_with(
                    |row, dst| dst.copy_from_slice(&field[row * nx..(row + 1) * nx]),
                    band,
                    |_, _, _, _| {},
                    &mut limited,
                );
                for index in 0..ny * kxl {
                    let (re, im) = if index % kxl < band {
                        (full.re[index], full.im[index])
                    } else {
                        (0.0, 0.0)
                    };
                    assert_eq!((limited.re[index], limited.im[index]), (re, im), "band {band}");
                }
                // Inverse: entries at kx >= band are ignored.
                let mut truncated = full.clone();
                for index in 0..ny * kxl {
                    if index % kxl >= band {
                        truncated.re[index] = 0.0;
                        truncated.im[index] = 0.0;
                    }
                }
                let mut expected = vec![0.0; nx * ny];
                plan.inverse(&truncated, &mut expected);
                let mut got = vec![0.0; nx * ny];
                plan.inverse_with(&full, band, |_, _, _, _| {}, &mut got);
                assert_eq!(got, expected, "{nx}x{ny} band {band}");
            }
        }
    }

    /// Forward then inverse transform under a rayon pool of `threads` threads.
    fn transform_in_pool(threads: usize, nx: usize, ny: usize) -> (Spectrum, Vec<f64>) {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("pool");
        pool.install(|| {
            let mut plan = Rfft2d::new(nx, ny).expect("smooth");
            let field = noise(nx * ny, 77);
            let mut spec = plan.zero_spectrum();
            plan.forward(&field, &mut spec);
            let mut back = vec![0.0; nx * ny];
            plan.inverse(&spec, &mut back);
            (spec, back)
        })
    }

    #[test]
    fn results_do_not_depend_on_the_thread_count() {
        for (nx, ny) in [(48, 32), (90, 64), (120, 50)] {
            let (s1, b1) = transform_in_pool(1, nx, ny);
            let (s3, b3) = transform_in_pool(3, nx, ny);
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&s1.re), bits(&s3.re));
            assert_eq!(bits(&s1.im), bits(&s3.im));
            assert_eq!(bits(&b1), bits(&b3));
        }
    }

    #[test]
    fn repeated_transforms_reuse_scratch_without_changing_results() {
        let (nx, ny) = (90, 64);
        let mut plan = Rfft2d::new(nx, ny).expect("smooth");
        let a = noise(nx * ny, 21);
        let b = noise(nx * ny, 22);
        let mut first = plan.zero_spectrum();
        plan.forward(&a, &mut first);
        let mut other = plan.zero_spectrum();
        plan.forward(&b, &mut other);
        let mut again = plan.zero_spectrum();
        plan.forward(&a, &mut again);
        assert_eq!(first, again);
    }

    /// SHA-256 of the forward transform of a fixed 48×32 pseudo-random field (real parts then
    /// imaginary parts, little-endian IEEE bits). Identical on every architecture; a change means
    /// the transform's arithmetic changed and every downstream golden hash must be re-blessed.
    const GOLDEN_48X32: &str = "7ed5ce9567564df27a9555b003bfa0cb560fd78c01f920ab767e0611ae5ae653";

    #[test]
    fn forward_transform_golden_hash() {
        let (nx, ny) = (48, 32);
        let mut plan = Rfft2d::new(nx, ny).expect("smooth");
        let field = noise(nx * ny, 0x5eed);
        let mut spec = plan.zero_spectrum();
        plan.forward(&field, &mut spec);
        let mut hasher = Sha256::new();
        for value in spec.re.iter().chain(&spec.im) {
            hasher.update(value.to_bits().to_le_bytes());
        }
        let digest = hex::encode(hasher.finalize());
        assert_eq!(digest, GOLDEN_48X32);
    }

    /// Release-mode timing of forward + inverse round trips (run with `--ignored --nocapture`).
    #[test]
    #[ignore = "timing report; run explicitly in release mode"]
    fn round_trip_timing() {
        for (nx, ny) in [(2160, 1536), (1440, 1024), (720, 512)] {
            let mut round_trip = bench::RoundTrip::new(nx, ny).expect("smooth");
            for _ in 0..5 {
                round_trip.run();
            }
            let reps = 50;
            let start = std::time::Instant::now();
            for _ in 0..reps {
                round_trip.run();
            }
            let ms = start.elapsed().as_secs_f64() * 1e3 / f64::from(reps);
            println!("rfft2d {nx}x{ny}: forward+inverse {ms:.3} ms");
        }
    }
}
