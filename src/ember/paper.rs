//! The kozo sheet: formation (flocs) and visible surface fibres, as per-pixel paper mottle and ink
//! gain.
//!
//! The sheet is one physical object measured in millimetres: `sheet_width_mm` spans the output
//! width, the pixels are square with pitch `p = sheet_width_mm / width`, and the sheet is
//! `p·height` mm tall. Coordinates run `x` right and `y` *down* from the top-left corner (image
//! convention); pixel `(row, col)` covers `[col·p, (col+1)·p) × [row·p, (row+1)·p)`.
//!
//! # Formation `F` (flocs)
//!
//! A stationary, nearly Gaussian field of unit variance built from `N` random Fourier modes,
//!
//! ```text
//! F(x, y) = Σ_j a·cos(kx_j·x + ky_j·y + φ_j),   a = sqrt(2/N),
//! λ_j = exp(U(ln floc_min, ln floc_max))  (log-uniform wavelength, mm),   θ_j, φ_j ~ U(0, 2π),
//! kx_j = (2π/λ_j)·cos θ_j / (1 + anisotropy),   ky_j = (2π/λ_j)·sin θ_j
//! ```
//!
//! (flocs stretched along `x`, the forming flow of nagashi-zuki). Each pixel holds the **exact**
//! average of `F` over its box, `a·sinc(kx·p/2)·sinc(ky·p/2)·cos(kx·xc + ky·yc + φ)` at the pixel
//! centre (`sinc t = sin t / t`), so a 2×2 block average of a fine render equals the coarse render
//! to rounding. By the angle-addition formula
//! `cos(A + B) = cos A·cos B - sin A·sin B` with `A = kx·xc + φ` (per column) and `B = ky·yc`
//! (per row), only `N·(width + height)` sine/cosine pairs are needed; the `N·width·height`
//! products are summed per pixel in ascending mode order (tiled for cache reuse, rows in
//! parallel, the per-pixel order never changes). The modes are processed in passes of
//! [`MODE_CHUNK`], each adding its modes to the stored partial sums (an exact `f64`
//! store/load), so the tables take `MODE_CHUNK·(width + height)·16` bytes whatever `N` is and
//! the result is bit-identical to a single pass.
//!
//! # Fibres `C` (covered area fraction)
//!
//! Long bast fibres lie in the sheet plane as slowly bending polylines, seeded uniformly over the
//! sheet padded by `margin_mm` on every side, `round(density·area_padded/100 mm²)` of them:
//!
//! * length log-normal with the configured 5th/95th percentiles
//!   (`μ = (ln p5 + ln p95)/2`, `σ = (ln p95 - ln p5)/(2·1.6448536…)`),
//! * width `U(width_min, width_max)` µm, depth visibility `U(visibility_min, visibility_max)`,
//! * direction: with probability `aligned_fraction` normal about the flow (`x`) with standard
//!   deviation `aligned_spread_deg`, else uniform in `[0, 2π)`,
//! * `n = max(1, ⌈length / vertex_spacing⌉)` equal segments of length `ℓ = length/n`; segment `s`
//!   has direction `θ_s`, with `θ_{s+1} = θ_s + bend·ℓ + wander·sqrt(ℓ)·z_s`, where the constant
//!   curvature `bend ~ N(0, bend_rad_per_mm)` and `z_s ~ N(0, 1)`.
//!
//! Each segment is sampled at the midpoints of `m = max(1, ⌈vertex_spacing/step⌉)` equal pieces,
//! `step = min(vertex_spacing, p/3)`, and at `across = max(1, ⌈width_max/step⌉)` points across
//! the fibre, the midpoints of equal pieces of its width along the segment's normal (`width_max`
//! is the configured widest fibre). `across > 1` exactly when `step < width_max`, i.e. at the
//! production vertex spacing when the pixels are narrower than three widths of the widest fibre
//! (`p < 3·width_max`). Every sample deposits the area `width·visibility·ℓ/(m·across)` into the
//! pixel containing it, and `C = min(deposited area / p², 1)`. The deposits accumulate
//! sequentially in fibre, segment, along and across order.
//!
//! # Gains
//!
//! ```text
//! ink_gain = clip(1 - ink_gain_fibre·C, 0.2, 2)                        (fibres shed ink)
//! mottle   = clip(1 + mottle_formation·F - mottle_fibre·C, 0.3, 3)     (paper absorbers)
//! ```
//!
//! Since `C ≥ 0`, `ink_gain ≤ 1`, so `ink_gain·(1 - mottle) ≤ 0.7 < 1`: the optics' absorption
//! stays positive.
//!
//! # Random draws (fixed order)
//!
//! All randomness comes from the given [`Sha3RandomByteStream`] through `next_f64` (uniform in
//! `[0, 1]`), in this order:
//!
//! 1. per formation mode `j = 0..N`: wavelength uniform, angle uniform, phase uniform;
//! 2. per fibre, in order: length normal, width uniform, visibility uniform, alignment uniform,
//!    aligned-angle normal, free-angle uniform, bend normal, start `x` uniform, start `y`
//!    uniform, then the `n - 1` wander normals of its segments.
//!
//! Standard normals come from the Box–Muller transform in pairs: a request with no cached value
//! draws `u1, u2`, returns `sqrt(-2 ln max(u1, 2⁻⁶⁴))·cos(2π u2)` and caches the matching sine
//! deviate for the next request (across fibres). Both angle draws happen for every fibre, so the
//! alignment coin never shifts the stream.

use std::f64::consts::TAU;

use rayon::prelude::*;

use super::config::{FibreConfig, PaperConfig};
use super::error::{EmberError, EmberResult};
use super::math;
use crate::sim::Sha3RandomByteStream;

/// Standard-normal 95th percentile `Φ⁻¹(0.95)`: log-normal lengths use `σ = ln(p95/p5)/(2·Z95)`.
const Z95: f64 = 1.6448536269514722;
/// Smallest positive `Sha3RandomByteStream::next_f64` value, `2⁻⁶⁴`; Box–Muller clamps its
/// radius uniform here so the logarithm never sees 0 (a deviate of at most 9.42σ).
const SMALLEST_UNIFORM: f64 = 1.0 / (u64::MAX as f64);
/// Most fibres a sheet may hold (a memory guard against absurd densities).
const MAX_FIBRES: f64 = 16_777_216.0;
/// Range of the ink gain.
const INK_GAIN_RANGE: (f64, f64) = (0.2, 2.0);
/// Range of the paper mottle.
const MOTTLE_RANGE: (f64, f64) = (0.3, 3.0);
/// Output rows per parallel task of the formation sum.
const ROW_BLOCK: usize = 8;
/// Output columns per cache tile of the formation sum.
const COL_TILE: usize = 64;
/// Formation modes per pass of the formation sum: bounds its sine/cosine tables to
/// `MODE_CHUNK·(width + height)·16` bytes (23 MB at 3456×2234, 134 MB at 16384×16384) for any
/// mode count, at the cost of one extra load and store of the partial sums per pass.
const MODE_CHUNK: usize = 256;

/// Paper properties under one output pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PaperSample {
    /// Multiplier of the paper's own absorption (formation and fibres).
    pub mottle: f64,
    /// Multiplier of the ink's absorption and scattering (fibres shed ink).
    pub ink_gain: f64,
}

impl Default for PaperSample {
    fn default() -> Self {
        Self { mottle: 1.0, ink_gain: 1.0 }
    }
}

/// Per-pixel mottle and ink gain of a kozo sheet covering the output.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct KozoSheet {
    /// Output width in pixels.
    pub width: usize,
    /// Output height in pixels.
    pub height: usize,
    /// Per-pixel paper absorption multiplier in `[0.3, 3]` (row-major, row 0 = top).
    pub mottle: Vec<f32>,
    /// Per-pixel ink multiplier in `[0.2, 1]` (row-major, row 0 = top).
    pub ink_gain: Vec<f32>,
}

impl KozoSheet {
    /// Generates the sheet for a `width × height` output from `rng` (draw order in the module
    /// docs). `config` must satisfy [`EmberConfig::validate`]; the values this function divides
    /// by are re-checked. The result is identical for every rayon thread count.
    ///
    /// [`EmberConfig::validate`]: super::config::EmberConfig::validate
    pub(crate) fn generate(
        width: u32,
        height: u32,
        config: &PaperConfig,
        rng: &mut Sha3RandomByteStream,
    ) -> EmberResult<Self> {
        let geometry = SheetGeometry::new(width, height, config.sheet_width_mm)?;
        if config.formation_modes == 0 {
            return Err(invalid("paper.formation_modes", "must be at least 1".into()));
        }
        let fibres = fibre_count(&geometry, &config.fibres)?;
        let formation = Formation::draw(config, rng);
        // The fibres (sequential, the rest of the random stream) are laid while the formation
        // (parallel, no randomness) is summed; neither depends on the other.
        let (field, cover) = rayon::join(
            || formation.render(&geometry),
            || fibre_cover(&geometry, &config.fibres, fibres, rng),
        );
        let cover = cover?;

        let pixels = geometry.width * geometry.height;
        let mut mottle = vec![0.0f32; pixels];
        let mut ink_gain = vec![0.0f32; pixels];
        mottle
            .par_chunks_mut(geometry.width)
            .zip(ink_gain.par_chunks_mut(geometry.width))
            .enumerate()
            .for_each(|(row, (mottle, ink_gain))| {
                let start = row * geometry.width;
                let field = &field[start..start + geometry.width];
                let cover = &cover[start..start + geometry.width];
                for (col, (&f, &c)) in field.iter().zip(cover).enumerate() {
                    let gain = 1.0 - config.ink_gain_fibre * c;
                    let absorb = 1.0 + config.mottle_formation * f - config.mottle_fibre * c;
                    ink_gain[col] = clip(gain, INK_GAIN_RANGE) as f32;
                    mottle[col] = clip(absorb, MOTTLE_RANGE) as f32;
                }
            });
        if !mottle.iter().chain(&ink_gain).all(|v| v.is_finite()) {
            return Err(EmberError::NonFinite { stage: "paper", time: 0.0 });
        }
        Ok(Self { width: geometry.width, height: geometry.height, mottle, ink_gain })
    }

    /// Paper properties of pixel `index` (row-major).
    pub(crate) fn sample(&self, index: usize) -> PaperSample {
        PaperSample {
            mottle: f64::from(self.mottle[index]),
            ink_gain: f64::from(self.ink_gain[index]),
        }
    }
}

/// Checks that the kozo sheet of a `width × height` output holds at most [`MAX_FIBRES`] visible
/// fibres under `config` (`round(density·(sheet width + 2·margin)·(sheet height + 2·margin) /
/// 100 mm²)`, see the module docs) and returns their number.
///
/// The count grows with the density and the output's aspect ratio, which
/// [`EmberConfig::validate`] cannot see together, so planning (`pipeline::plan_ember`) makes this
/// check: a request that plans successfully never fails in [`KozoSheet::generate`], which uses
/// the same count.
///
/// [`EmberConfig::validate`]: super::config::EmberConfig::validate
pub(crate) fn check_fibre_count(
    width: u32,
    height: u32,
    config: &PaperConfig,
) -> EmberResult<usize> {
    fibre_count(&SheetGeometry::new(width, height, config.sheet_width_mm)?, &config.fibres)
}

/// Visible fibres on the sheet `geometry` (see [`check_fibre_count`]); rejects a count above
/// [`MAX_FIBRES`] or a non-finite one.
fn fibre_count(geometry: &SheetGeometry, config: &FibreConfig) -> EmberResult<usize> {
    let (span_x, span_y) = fibre_span(geometry, config);
    let count = (config.density_per_cm2 * span_x * span_y / 100.0).round();
    if !(0.0..=MAX_FIBRES).contains(&count) {
        return Err(invalid(
            "paper.fibres.density_per_cm2",
            format!("the sheet would hold {count} fibres (at most {MAX_FIBRES})"),
        ));
    }
    Ok(count as usize)
}

/// Width and height (mm) of the area fibres are seeded over: the sheet padded by `margin_mm` on
/// every side.
fn fibre_span(geometry: &SheetGeometry, config: &FibreConfig) -> (f64, f64) {
    let margin = config.margin_mm;
    (geometry.width_mm + 2.0 * margin, geometry.height_mm + 2.0 * margin)
}

/// An [`EmberError::InvalidConfig`] for `parameter`.
fn invalid(parameter: &str, reason: String) -> EmberError {
    EmberError::InvalidConfig { parameter: parameter.to_string(), reason }
}

/// `v` clipped to `[lo, hi]` by explicit comparisons.
#[inline]
fn clip(v: f64, (lo, hi): (f64, f64)) -> f64 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// `sin t / t` (1 at 0).
#[inline]
fn sinc(t: f64) -> f64 {
    if t == 0.0 { 1.0 } else { math::sin(t) / t }
}

/// Physical layout of the sheet under the output.
#[derive(Clone, Copy, Debug, PartialEq)]
struct SheetGeometry {
    /// Output width in pixels.
    width: usize,
    /// Output height in pixels.
    height: usize,
    /// Pixel pitch (mm, square pixels): `width_mm / width`.
    pitch: f64,
    /// Sheet width (mm).
    width_mm: f64,
    /// Sheet height (mm): `pitch·height`, i.e. `width_mm·height/width`.
    height_mm: f64,
}

impl SheetGeometry {
    /// `width_mm` across `width` pixels; rejects empty outputs and non-positive widths.
    fn new(width: u32, height: u32, width_mm: f64) -> EmberResult<Self> {
        if width == 0 || height == 0 {
            return Err(invalid("resolution", format!("{width}x{height} must be non-empty")));
        }
        if !(width_mm.is_finite() && width_mm > 0.0) {
            return Err(invalid("paper.sheet_width_mm", "must be finite and > 0".into()));
        }
        let pitch = width_mm / f64::from(width);
        Ok(Self {
            width: width as usize,
            height: height as usize,
            pitch,
            width_mm,
            height_mm: pitch * f64::from(height),
        })
    }
}

/// The random cosine modes of the formation field.
#[derive(Clone, Debug, PartialEq)]
struct Formation {
    /// Wavevector `x` components (rad/mm).
    kx: Vec<f64>,
    /// Wavevector `y` components (rad/mm).
    ky: Vec<f64>,
    /// Phases (rad).
    phase: Vec<f64>,
}

impl Formation {
    /// Draws `config.formation_modes` modes (three uniforms each: wavelength, angle, phase).
    fn draw(config: &PaperConfig, rng: &mut Sha3RandomByteStream) -> Self {
        let modes = config.formation_modes;
        let ln_min = math::ln(config.floc_min_mm);
        let ln_span = math::ln(config.floc_max_mm) - ln_min;
        let stretch = 1.0 + config.anisotropy;
        let mut formation = Self {
            kx: Vec::with_capacity(modes),
            ky: Vec::with_capacity(modes),
            phase: Vec::with_capacity(modes),
        };
        for _ in 0..modes {
            let wavelength = math::exp(ln_min + rng.next_f64() * ln_span);
            let angle = TAU * rng.next_f64();
            let phase = TAU * rng.next_f64();
            let k = TAU / wavelength;
            let (sin, cos) = math::sin_cos(angle);
            formation.kx.push(k * cos / stretch);
            formation.ky.push(k * sin);
            formation.phase.push(phase);
        }
        formation
    }

    /// Number of modes.
    fn modes(&self) -> usize {
        self.kx.len()
    }

    /// Pixel-box averages of `F` (row-major, row 0 = top), in passes of [`MODE_CHUNK`] modes.
    fn render(&self, geometry: &SheetGeometry) -> Vec<f64> {
        self.render_in_chunks(geometry, MODE_CHUNK)
    }

    /// [`Formation::render`] with passes of `chunk` modes (at least 1); the result does not
    /// depend on `chunk`, bit for bit.
    ///
    /// Per pass, per-mode column factors `a_j·(cos A, sin A)` (mode-major, `A = kx·xc + φ`,
    /// amplitude `a_j = sqrt(2/N)·sinc(kx·p/2)·sinc(ky·p/2)` folded in) and row factors
    /// `(cos B, sin B)` (row-major, `B = ky·yc`) are tabulated in parallel. Each pixel's partial
    /// sum (0 before the first pass) is then loaded, the pass's
    /// `cos B·a cos A - sin B·a sin A` terms are added in ascending mode order, and it is stored
    /// back: every pixel sees the same additions in the same order as one pass over all modes.
    fn render_in_chunks(&self, geometry: &SheetGeometry, chunk: usize) -> Vec<f64> {
        let (width, height, modes) = (geometry.width, geometry.height, self.modes());
        let chunk = chunk.clamp(1, modes.max(1));
        let pitch = geometry.pitch;
        let half = 0.5 * pitch;
        let base = (2.0 / modes as f64).sqrt();

        // Tables of one pass (the last pass may use a prefix), allocated once.
        let mut col_cos = vec![0.0f64; chunk * width];
        let mut col_sin = vec![0.0f64; chunk * width];
        let mut row_cos = vec![0.0f64; height * chunk];
        let mut row_sin = vec![0.0f64; height * chunk];
        let mut field = vec![0.0f64; width * height];
        for first in (0..modes).step_by(chunk) {
            let count = (modes - first).min(chunk);
            let (col_cos, col_sin) = (&mut col_cos[..count * width], &mut col_sin[..count * width]);
            let (row_cos, row_sin) =
                (&mut row_cos[..height * count], &mut row_sin[..height * count]);
            col_cos.par_chunks_mut(width).zip(col_sin.par_chunks_mut(width)).enumerate().for_each(
                |(j, (cos_out, sin_out))| {
                    let (kx, ky, phase) =
                        (self.kx[first + j], self.ky[first + j], self.phase[first + j]);
                    let amp = base * sinc(kx * half) * sinc(ky * half);
                    for (col, (c, s)) in cos_out.iter_mut().zip(sin_out.iter_mut()).enumerate() {
                        let x = (col as f64 + 0.5) * pitch;
                        let (sin, cos) = math::sin_cos(kx * x + phase);
                        *c = amp * cos;
                        *s = amp * sin;
                    }
                },
            );
            let ky = &self.ky[first..first + count];
            row_cos.par_chunks_mut(count).zip(row_sin.par_chunks_mut(count)).enumerate().for_each(
                |(row, (cos_out, sin_out))| {
                    let y = (row as f64 + 0.5) * pitch;
                    for ((c, s), &ky) in cos_out.iter_mut().zip(sin_out.iter_mut()).zip(ky) {
                        let (sin, cos) = math::sin_cos(ky * y);
                        *c = cos;
                        *s = sin;
                    }
                },
            );
            let (col_cos, col_sin, row_cos, row_sin) = (&*col_cos, &*col_sin, &*row_cos, &*row_sin);
            field.par_chunks_mut(width * ROW_BLOCK).enumerate().for_each(|(block, out)| {
                let first_row = block * ROW_BLOCK;
                let rows = out.len() / width;
                let mut acc = [0.0f64; ROW_BLOCK * COL_TILE];
                for c0 in (0..width).step_by(COL_TILE) {
                    let tile = (width - c0).min(COL_TILE);
                    for r in 0..rows {
                        acc[r * COL_TILE..r * COL_TILE + tile]
                            .copy_from_slice(&out[r * width + c0..r * width + c0 + tile]);
                    }
                    for j in 0..count {
                        let cos_a = &col_cos[j * width + c0..j * width + c0 + tile];
                        let sin_a = &col_sin[j * width + c0..j * width + c0 + tile];
                        for r in 0..rows {
                            let cos_b = row_cos[(first_row + r) * count + j];
                            let sin_b = row_sin[(first_row + r) * count + j];
                            let acc = &mut acc[r * COL_TILE..r * COL_TILE + tile];
                            for ((acc, &ca), &sa) in acc.iter_mut().zip(cos_a).zip(sin_a) {
                                *acc += cos_b * ca - sin_b * sa;
                            }
                        }
                    }
                    for r in 0..rows {
                        out[r * width + c0..r * width + c0 + tile]
                            .copy_from_slice(&acc[r * COL_TILE..r * COL_TILE + tile]);
                    }
                }
            });
        }
        field
    }
}

/// Standard normal deviates by the Box–Muller transform, produced in pairs (module docs).
#[derive(Clone, Copy, Debug, Default)]
struct Gaussian {
    /// The sine deviate of the last pair, if not yet used.
    spare: Option<f64>,
}

impl Gaussian {
    /// The next standard normal deviate.
    fn next(&mut self, rng: &mut Sha3RandomByteStream) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let (z0, z1) = box_muller(rng.next_f64(), rng.next_f64());
        self.spare = Some(z1);
        z0
    }
}

/// Two independent standard normal deviates from two uniforms in `[0, 1]`:
/// `r·(cos 2πu2, sin 2πu2)` with `r = sqrt(-2 ln max(u1, 2⁻⁶⁴))`.
fn box_muller(u1: f64, u2: f64) -> (f64, f64) {
    let u1 = if u1 > SMALLEST_UNIFORM { u1 } else { SMALLEST_UNIFORM };
    let radius = (-2.0 * math::ln(u1)).sqrt();
    let (sin, cos) = math::sin_cos(TAU * u2);
    (radius * cos, radius * sin)
}

/// One fibre's drawn shape (the wander of its direction is drawn while it is laid).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Fibre {
    /// First vertex (mm, sheet coordinates).
    start: [f64; 2],
    /// Direction of the first segment (rad, `x` towards `y`).
    angle: f64,
    /// Constant turn per segment, `bend·ℓ` (rad).
    turn: f64,
    /// Number of segments `n ≥ 1`.
    segments: usize,
    /// Segment length `ℓ` (mm).
    segment_mm: f64,
    /// Width (mm).
    width_mm: f64,
    /// Depth visibility in `[0, 1]`.
    visibility: f64,
}

/// How a fibre is sampled into the pixel grid.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Splat {
    /// Samples along each segment.
    along: usize,
    /// Samples across the fibre.
    across: usize,
}

impl Splat {
    /// Sample counts for fibres at most `width_max_mm` wide on pixels of `pitch`.
    fn new(pitch: f64, vertex_spacing: f64, width_max_mm: f64) -> Self {
        let step = vertex_spacing.min(pitch / 3.0);
        let count = |extent: f64| ((extent / step).ceil() as usize).max(1);
        Self { along: count(vertex_spacing), across: count(width_max_mm) }
    }
}

/// Lays `fibre` into `cover` (deposited area per pixel, mm²). `wander` yields the direction
/// change beyond the constant turn after each segment but the last (it draws from the RNG).
fn lay_fibre(
    fibre: &Fibre,
    splat: Splat,
    geometry: &SheetGeometry,
    cover: &mut [f64],
    mut wander: impl FnMut() -> f64,
) {
    let area =
        fibre.width_mm * fibre.visibility * fibre.segment_mm / (splat.along * splat.across) as f64;
    let [mut x, mut y] = fibre.start;
    let mut theta = fibre.angle;
    for segment in 0..fibre.segments {
        let (sin, cos) = math::sin_cos(theta);
        let (dx, dy) = (fibre.segment_mm * cos, fibre.segment_mm * sin);
        for i in 0..splat.along {
            let f = (i as f64 + 0.5) / splat.along as f64;
            let (mx, my) = (x + f * dx, y + f * dy);
            for q in 0..splat.across {
                let offset = ((q as f64 + 0.5) / splat.across as f64 - 0.5) * fibre.width_mm;
                deposit(cover, geometry, mx - offset * sin, my + offset * cos, area);
            }
        }
        x += dx;
        y += dy;
        if segment + 1 < fibre.segments {
            theta += fibre.turn + wander();
        }
    }
}

/// Adds `area` to the pixel containing `(x, y)` (mm), if any.
#[inline]
fn deposit(cover: &mut [f64], geometry: &SheetGeometry, x: f64, y: f64, area: f64) {
    let col = (x / geometry.pitch).floor();
    let row = (y / geometry.pitch).floor();
    if col >= 0.0 && row >= 0.0 && col < geometry.width as f64 && row < geometry.height as f64 {
        cover[row as usize * geometry.width + col as usize] += area;
    }
}

/// Fibre coverage `C` per pixel (row-major, covered area fraction in `[0, 1]`) of `count` fibres
/// ([`fibre_count`]).
fn fibre_cover(
    geometry: &SheetGeometry,
    config: &FibreConfig,
    count: usize,
    rng: &mut Sha3RandomByteStream,
) -> EmberResult<Vec<f64>> {
    let margin = config.margin_mm;
    let (span_x, span_y) = fibre_span(geometry, config);
    if !(config.vertex_spacing_mm.is_finite() && config.vertex_spacing_mm > 0.0) {
        return Err(invalid("paper.fibres.vertex_spacing_mm", "must be finite and > 0".into()));
    }
    let ln_p5 = math::ln(config.length_p5_mm);
    let ln_p95 = math::ln(config.length_p95_mm);
    let (mu, sigma) = (0.5 * (ln_p5 + ln_p95), (ln_p95 - ln_p5) / (2.0 * Z95));
    let spread = config.aligned_spread_deg.to_radians();
    let splat = Splat::new(geometry.pitch, config.vertex_spacing_mm, 1e-3 * config.width_max_um);

    let mut cover = vec![0.0f64; geometry.width * geometry.height];
    let mut gaussian = Gaussian::default();
    for _ in 0..count {
        let length = math::exp(mu + sigma * gaussian.next(rng));
        let width_um =
            config.width_min_um + (config.width_max_um - config.width_min_um) * rng.next_f64();
        let visibility = config.visibility_min
            + (config.visibility_max - config.visibility_min) * rng.next_f64();
        let aligned = rng.next_f64() < config.aligned_fraction;
        let aligned_angle = spread * gaussian.next(rng);
        let free_angle = TAU * rng.next_f64();
        let bend = config.bend_rad_per_mm * gaussian.next(rng);
        let start = [-margin + span_x * rng.next_f64(), -margin + span_y * rng.next_f64()];
        let segments = ((length / config.vertex_spacing_mm).ceil() as usize).max(1);
        let segment_mm = length / segments as f64;
        let fibre = Fibre {
            start,
            angle: if aligned { aligned_angle } else { free_angle },
            turn: bend * segment_mm,
            segments,
            segment_mm,
            width_mm: 1e-3 * width_um,
            visibility,
        };
        let kick = config.wander_rad_per_sqrt_mm * segment_mm.sqrt();
        lay_fibre(&fibre, splat, geometry, &mut cover, || kick * gaussian.next(rng));
    }
    let pixel_area = geometry.pitch * geometry.pitch;
    for c in &mut cover {
        let fraction = *c / pixel_area;
        *c = if fraction < 1.0 { fraction } else { 1.0 };
    }
    Ok(cover)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ember::config::EmberConfig;
    use sha2::{Digest, Sha256};

    fn rng(seed: &[u8]) -> Sha3RandomByteStream {
        Sha3RandomByteStream::new(seed, 0.0, 1.0, 1.0, 1.0)
    }

    /// The production paper on a small sheet (`width_mm` across) with `modes` formation modes.
    fn small_paper(width_mm: f64, modes: usize) -> PaperConfig {
        let mut config = EmberConfig::default().paper;
        config.sheet_width_mm = width_mm;
        config.formation_modes = modes;
        config
    }

    fn mean_std(values: impl Iterator<Item = f64> + Clone) -> (f64, f64) {
        let n = values.clone().count() as f64;
        let mean = values.clone().sum::<f64>() / n;
        let var = values.map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
        (mean, var.sqrt())
    }

    fn sheet_digest(sheet: &KozoSheet) -> String {
        let mut hasher = Sha256::new();
        for v in sheet.mottle.iter().chain(&sheet.ink_gain) {
            hasher.update(v.to_bits().to_le_bytes());
        }
        hex::encode(hasher.finalize())
    }

    #[test]
    fn geometry_follows_the_output_aspect() {
        let wide = SheetGeometry::new(64, 32, 1490.0).expect("valid");
        assert_eq!(wide.pitch, 1490.0 / 64.0);
        assert_eq!(wide.height_mm, 745.0);
        let tall = SheetGeometry::new(32, 64, 1490.0).expect("valid");
        assert_eq!(tall.height_mm, 2980.0);
        let product = SheetGeometry::new(3456, 2234, 1490.0).expect("valid");
        assert!((product.height_mm - 1490.0 * 2234.0 / 3456.0).abs() <= 1e-12);
        assert!(SheetGeometry::new(0, 10, 1490.0).is_err());
        assert!(SheetGeometry::new(10, 10, 0.0).is_err());

        let config = small_paper(40.0, 16);
        for (w, h) in [(24u32, 8u32), (8, 24), (1, 1)] {
            let sheet = KozoSheet::generate(w, h, &config, &mut rng(b"aspect")).expect("valid");
            assert_eq!((sheet.width, sheet.height), (w as usize, h as usize));
            assert_eq!(sheet.mottle.len(), (w * h) as usize);
            assert_eq!(sheet.ink_gain.len(), (w * h) as usize);
        }
        let mut bad = config.clone();
        bad.formation_modes = 0;
        assert!(KozoSheet::generate(4, 4, &bad, &mut rng(b"x")).is_err());
        let mut dense = config;
        dense.fibres.density_per_cm2 = 1e9;
        assert!(KozoSheet::generate(4, 4, &dense, &mut rng(b"x")).is_err());
    }

    /// The fibre-count guard is the planning check: the count follows the documented formula,
    /// grows with the output's aspect ratio (a tall output depicts a taller sheet), and a sheet
    /// that would hold more than [`MAX_FIBRES`] is rejected naming the density, by the check and
    /// by the generator alike.
    #[test]
    fn the_fibre_count_is_checked_before_any_fibre_is_laid() {
        let paper = EmberConfig::default().paper;
        let fibres = &paper.fibres;
        // 3456×2234 on a 1490 mm sheet: (1490 + 30) mm × (963.2 + 30) mm at 1.5 per cm².
        let height_mm = 1490.0 * 2234.0 / 3456.0;
        let expected = (fibres.density_per_cm2
            * (1490.0 + 2.0 * fibres.margin_mm)
            * (height_mm + 2.0 * fibres.margin_mm)
            / 100.0)
            .round() as usize;
        assert_eq!(check_fibre_count(3456, 2234, &paper).expect("production"), expected);
        assert!((22_000..24_000).contains(&expected), "{expected}");
        let square = check_fibre_count(64, 64, &paper).expect("valid");
        let tall = check_fibre_count(64, 16_384, &paper).expect("valid");
        assert!(tall > 200 * square, "{tall} vs {square}");

        // Validation checks ranges only: it accepts a density whose count only the size decides.
        let mut config = EmberConfig::default();
        config.paper.fibres.density_per_cm2 = 1e4;
        config.validate().expect("a valid range");
        let dense = config.paper;
        for error in [
            check_fibre_count(96, 64, &dense).expect_err("too many fibres"),
            KozoSheet::generate(96, 64, &dense, &mut rng(b"dense")).expect_err("too many fibres"),
        ] {
            match error {
                EmberError::InvalidConfig { parameter, reason } => {
                    assert_eq!(parameter, "paper.fibres.density_per_cm2");
                    assert!(reason.contains("at most 16777216"), "{reason}");
                }
                other => panic!("expected InvalidConfig, got {other:?}"),
            }
        }
        assert!(check_fibre_count(0, 64, &paper).is_err(), "an empty output has no sheet");
    }

    /// Pixel values are exact box averages: a 2×2 block of a fine render averages to the coarse
    /// render of the same physical sheet.
    #[test]
    fn formation_is_the_exact_box_average() {
        let config = small_paper(24.0, 64);
        let formation = Formation::draw(&config, &mut rng(b"flocs"));
        let fine = formation.render(&SheetGeometry::new(48, 32, 24.0).expect("valid"));
        let coarse = formation.render(&SheetGeometry::new(24, 16, 24.0).expect("valid"));
        for row in 0..16 {
            for col in 0..24 {
                let block = [(0, 0), (0, 1), (1, 0), (1, 1)]
                    .map(|(i, j)| fine[(2 * row + i) * 48 + 2 * col + j]);
                let mean = 0.25 * ((block[0] + block[1]) + (block[2] + block[3]));
                assert!((mean - coarse[row * 24 + col]).abs() <= 1e-12, "({row}, {col})");
            }
        }
    }

    /// The tiled angle-addition sum equals the direct sum of box-averaged cosines.
    #[test]
    fn formation_matches_direct_evaluation() {
        let config = small_paper(30.0, 40);
        let formation = Formation::draw(&config, &mut rng(b"direct"));
        // Wider than one column tile and taller than one row block, with ragged remainders.
        let geometry = SheetGeometry::new(75, 11, 30.0).expect("valid");
        let field = formation.render(&geometry);
        let p = geometry.pitch;
        let amp = (2.0 / 40.0f64).sqrt();
        for row in 0..11 {
            for col in 0..75 {
                let (x, y) = ((col as f64 + 0.5) * p, (row as f64 + 0.5) * p);
                let mut direct = 0.0;
                for j in 0..40 {
                    let (kx, ky) = (formation.kx[j], formation.ky[j]);
                    direct += amp
                        * sinc(0.5 * kx * p)
                        * sinc(0.5 * ky * p)
                        * math::cos(kx * x + ky * y + formation.phase[j]);
                }
                assert!((field[row * 75 + col] - direct).abs() <= 1e-12, "({row}, {col})");
            }
        }
        // Wavelengths are log-uniform in [floc_min, floc_max], flocs stretched along x.
        for j in 0..40 {
            let kx = formation.kx[j] * (1.0 + config.anisotropy);
            let ky = formation.ky[j];
            let k = (kx * kx + ky * ky).sqrt();
            let wavelength = TAU / k;
            assert!(wavelength >= config.floc_min_mm * (1.0 - 1e-12));
            assert!(wavelength <= config.floc_max_mm * (1.0 + 1e-12));
        }
    }

    /// Passes of any size give the single-pass sums bit for bit (the production pass size is
    /// [`MODE_CHUNK`]; here several mode counts straddle smaller passes, ragged last pass
    /// included).
    #[test]
    fn formation_passes_do_not_change_a_bit() {
        let bits = |field: &[f64]| field.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let geometry = SheetGeometry::new(75, 19, 30.0).expect("valid");
        for modes in [1, 100, MODE_CHUNK + 37] {
            let formation = Formation::draw(&small_paper(30.0, modes), &mut rng(b"passes"));
            let single = bits(&formation.render_in_chunks(&geometry, modes));
            for chunk in [1, 7, 33, 99, MODE_CHUNK, 4096] {
                let passes = bits(&formation.render_in_chunks(&geometry, chunk));
                assert!(passes == single, "{modes} modes in passes of {chunk}");
            }
            assert!(bits(&formation.render(&geometry)) == single, "{modes} modes");
        }
    }

    /// `generate` combines exactly the formation and coverage its documented draws produce, by
    /// the documented gain formulas (written out here independently), clips included.
    #[test]
    fn gains_follow_the_formation_and_fibre_formulas() {
        // Distinct, strong coefficients so a swapped coefficient, a wrong sign or a missing
        // term changes the bits, and both lower clips engage somewhere.
        let mut config = small_paper(8.0, 48);
        config.mottle_formation = 1.5;
        config.mottle_fibre = 2.5;
        config.ink_gain_fibre = 12.0;
        config.fibres.density_per_cm2 = 40.0;
        let (width, height) = (96u32, 64u32);
        let sheet = KozoSheet::generate(width, height, &config, &mut rng(b"gains")).expect("valid");

        // Replay the draws in their documented order: the modes, then the fibres.
        let geometry = SheetGeometry::new(width, height, config.sheet_width_mm).expect("valid");
        let mut replay = rng(b"gains");
        let formation = Formation::draw(&config, &mut replay);
        let count = fibre_count(&geometry, &config.fibres).expect("valid");
        let cover = fibre_cover(&geometry, &config.fibres, count, &mut replay).expect("valid");
        let field = formation.render(&geometry);

        let (mut fibre_pixels, mut free_gain, mut clipped_gain, mut clipped_mottle) = (0, 0, 0, 0);
        for (i, (&f, &c)) in field.iter().zip(&cover).enumerate() {
            let ink_gain = (1.0 - config.ink_gain_fibre * c).clamp(0.2, 2.0) as f32;
            let mottle = (1.0 + config.mottle_formation * f - config.mottle_fibre * c)
                .clamp(0.3, 3.0) as f32;
            assert_eq!(sheet.ink_gain[i].to_bits(), ink_gain.to_bits(), "pixel {i}: ink gain");
            assert_eq!(sheet.mottle[i].to_bits(), mottle.to_bits(), "pixel {i}: mottle");
            fibre_pixels += usize::from(c > 0.0);
            free_gain += usize::from(c > 0.0 && ink_gain > 0.2);
            clipped_gain += usize::from(ink_gain == 0.2);
            clipped_mottle += usize::from(mottle == 0.3);
        }
        let pixels = (width * height) as usize;
        assert!(fibre_pixels > 50 && fibre_pixels < pixels / 2, "{fibre_pixels} fibre pixels");
        assert!(free_gain > 0 && clipped_gain > 0 && clipped_mottle > 0);
    }

    /// With a constant turn `δ` per segment (bend plus a constant wander) the fibre is a regular
    /// polygon inscribed in a circle: its vertices are the closed-form chords
    /// `V_s = V_0 + ℓ·sin(sδ/2)/sin(δ/2)·(cos, sin)(θ0 + (s-1)δ/2)` and segment `s` points along
    /// `θ0 + sδ`. Every sample (along and across) must land in the pixel of its analytic position.
    #[test]
    fn a_bent_fibre_follows_its_circular_arc() {
        let geometry = SheetGeometry::new(400, 200, 20.0).expect("valid"); // 0.05 mm pixels
        let splat = Splat::new(geometry.pitch, 0.25, 0.02);
        assert!(splat.along > 1 && splat.across == 2, "{splat:?}");
        // (turn, wander per step): a bend alone, and a bend plus a constant wander of either sign.
        for (turn, wander) in [(0.05, 0.0), (-0.03, 0.01), (0.02, 0.025)] {
            let fibre = Fibre {
                start: [4.0, 2.0],
                angle: 0.3,
                turn,
                segments: 32,
                segment_mm: 0.25,
                width_mm: 0.015,
                visibility: 0.7,
            };
            let mut cover = vec![0.0; geometry.width * geometry.height];
            let mut steps = 0;
            lay_fibre(&fibre, splat, &geometry, &mut cover, || {
                steps += 1;
                wander
            });
            assert_eq!(steps, fibre.segments - 1, "one wander step between segments");

            let delta = turn + wander;
            let ell = fibre.segment_mm;
            let chord = |s: f64| {
                let (sin, cos) = math::sin_cos(fibre.angle + 0.5 * (s - 1.0) * delta);
                let length = ell * math::sin(0.5 * s * delta) / math::sin(0.5 * delta);
                [fibre.start[0] + length * cos, fibre.start[1] + length * sin]
            };
            let area =
                fibre.width_mm * fibre.visibility * ell / (splat.along * splat.across) as f64;
            let mut expected = vec![0.0; cover.len()];
            let mut closest = f64::INFINITY;
            for s in 0..fibre.segments {
                let vertex = chord(s as f64);
                let (sin, cos) = math::sin_cos(fibre.angle + s as f64 * delta);
                for i in 0..splat.along {
                    let f = (i as f64 + 0.5) / splat.along as f64 * ell;
                    for q in 0..splat.across {
                        let o = ((q as f64 + 0.5) / splat.across as f64 - 0.5) * fibre.width_mm;
                        let x = vertex[0] + f * cos - o * sin;
                        let y = vertex[1] + f * sin + o * cos;
                        let (u, v) = (x / geometry.pitch, y / geometry.pitch);
                        closest = closest.min((u - u.round()).abs()).min((v - v.round()).abs());
                        let (col, row) = (u.floor() as usize, v.floor() as usize);
                        assert!(col < geometry.width && row < geometry.height, "({x}, {y})");
                        expected[row * geometry.width + col] += area;
                    }
                }
            }
            // No analytic sample is near a pixel edge, so rounding cannot move it across one.
            assert!(closest > 1e-9, "{closest}");
            assert!(cover == expected, "turn {turn}, wander {wander}");
            // The whole ribbon lies on the sheet: it deposits its full area.
            let total: f64 = cover.iter().sum();
            let full = fibre.width_mm * fibre.visibility * ell * fibre.segments as f64;
            assert!((total - full).abs() <= 1e-15, "{total} vs {full}");
            // And it really bends: the end is a chord of the circle of radius ℓ/(2 sin(δ/2)).
            let end = chord(fibre.segments as f64);
            let (dx, dy) = (end[0] - fibre.start[0], end[1] - fibre.start[1]);
            let span = (dx * dx + dy * dy).sqrt();
            assert!(span < 0.99 * ell * fibre.segments as f64, "{span}");
        }
    }

    #[test]
    fn box_muller_is_standard_normal_and_guards_zero() {
        let (z0, z1) = box_muller(0.0, 0.25);
        assert!(z0.is_finite() && z1.is_finite());
        assert!((z1 - (-2.0 * math::ln(SMALLEST_UNIFORM)).sqrt()).abs() <= 1e-12);
        assert_eq!(box_muller(1.0, 0.3), (0.0, 0.0));
        let mut source = rng(b"gauss");
        let mut gaussian = Gaussian::default();
        let draws: Vec<f64> = (0..20_000).map(|_| gaussian.next(&mut source)).collect();
        let (mean, std) = mean_std(draws.iter().copied());
        assert!(mean.abs() < 0.03 && (std - 1.0).abs() < 0.03, "{mean} {std}");
        let beyond = draws.iter().filter(|z| z.abs() > 1.959964).count() as f64 / 20_000.0;
        assert!((beyond - 0.05).abs() < 0.008, "{beyond}");
    }

    /// A straight, unbent fibre deposits exactly its area `width·visibility·length`, along its
    /// own pixel row.
    #[test]
    fn a_straight_fibre_deposits_its_area_along_its_path() {
        let geometry = SheetGeometry::new(40, 10, 20.0).expect("valid"); // 0.5 mm pixels
        let fibre = Fibre {
            start: [2.1, 3.3],
            angle: 0.0,
            turn: 0.0,
            segments: 40,
            segment_mm: 0.25,
            width_mm: 0.015,
            visibility: 0.8,
        };
        let splat = Splat::new(geometry.pitch, 0.25, 0.02);
        assert_eq!(splat, Splat { along: 2, across: 1 });
        let mut cover = vec![0.0; 400];
        lay_fibre(&fibre, splat, &geometry, &mut cover, || 0.0);
        let total: f64 = cover.iter().sum();
        assert!((total - 0.015 * 0.8 * 10.0).abs() <= 1e-15, "{total}");
        // y = 3.3 mm is row 6; x from 2.1 to 12.1 mm covers columns 4..=24.
        for (i, &c) in cover.iter().enumerate() {
            let (row, col) = (i / 40, i % 40);
            assert_eq!(c > 0.0, row == 6 && (4..=24).contains(&col), "({row}, {col})");
        }
        // Interior pixels get two samples per segment, one segment per half millimetre each.
        let per_pixel = 0.015 * 0.8 * 0.5;
        assert!((cover[6 * 40 + 10] - per_pixel).abs() <= 1e-15);
        // A fibre entirely off the sheet deposits nothing.
        let mut off = vec![0.0; 400];
        lay_fibre(&Fibre { start: [-30.0, 3.0], ..fibre }, splat, &geometry, &mut off, || 0.0);
        assert!(off.iter().all(|&c| c == 0.0));
        // Pixels narrower than the fibre sample it across its width too.
        let fine = Splat::new(0.01, 0.25, 0.02);
        assert_eq!(fine.across, 6);
    }

    #[test]
    fn sheet_statistics_match_the_prototype() {
        // Production pitch (~0.43 mm) on a small sheet: formation statistics.
        let config = small_paper(96.0, 512);
        let geometry = SheetGeometry::new(224, 160, config.sheet_width_mm).expect("valid");
        let formation = Formation::draw(&config, &mut rng(b"stats"));
        let field = formation.render(&geometry);
        let (mean, std) = mean_std(field.iter().copied());
        assert!(mean.abs() < 0.08, "F mean {mean}");
        assert!((0.8..1.05).contains(&std), "F std {std}");

        // Fibre coverage: its mean is the fibres' area per sheet area, independent of the pitch:
        // density · E[length] · E[width] · E[visibility] ≈ 0.00134.
        let fibres = &config.fibres;
        let mu = 0.5 * (math::ln(fibres.length_p5_mm) + math::ln(fibres.length_p95_mm));
        let sigma = (math::ln(fibres.length_p95_mm) - math::ln(fibres.length_p5_mm)) / (2.0 * Z95);
        let expected = fibres.density_per_cm2 / 100.0
            * math::exp(mu + 0.5 * sigma * sigma)
            * 1e-3
            * 0.5
            * (fibres.width_min_um + fibres.width_max_um)
            * 0.5
            * (fibres.visibility_min + fibres.visibility_max);
        assert!((expected - 0.00134).abs() < 5e-5, "{expected}");
        let wide = SheetGeometry::new(200, 125, 400.0).expect("valid");
        let count = fibre_count(&wide, fibres).expect("valid");
        let cover = fibre_cover(&wide, fibres, count, &mut rng(b"fibres")).expect("valid");
        let (c_mean, _) = mean_std(cover.iter().copied());
        assert!((c_mean / expected - 1.0).abs() < 0.15, "C mean {c_mean} vs {expected}");
        assert!(cover.iter().all(|&c| (0.0..=1.0).contains(&c)));

        // The gains of a full sheet at the production pitch.
        let sheet = KozoSheet::generate(224, 160, &config, &mut rng(b"stats")).expect("valid");
        let (m_mean, m_std) = mean_std(sheet.mottle.iter().map(|&v| f64::from(v)));
        assert!((m_mean - 1.0).abs() < 0.02 && (0.15..0.22).contains(&m_std), "{m_mean} {m_std}");
        assert!(sheet.mottle.iter().all(|&m| (0.3..=3.0).contains(&m)));
        assert!(sheet.ink_gain.iter().all(|&g| (0.2..=1.0).contains(&g)));
        let (g_mean, _) = mean_std(sheet.ink_gain.iter().map(|&v| f64::from(v)));
        assert!(g_mean > 0.998 && g_mean < 1.0, "{g_mean}");
        assert!(sheet.ink_gain.iter().any(|&g| g < 1.0), "some fibres are visible");
    }

    #[test]
    fn generation_is_deterministic_and_seeded() {
        let config = small_paper(20.0, 64);
        let a = KozoSheet::generate(48, 32, &config, &mut rng(b"kozo-canary")).expect("valid");
        let b = KozoSheet::generate(48, 32, &config, &mut rng(b"kozo-canary")).expect("valid");
        assert_eq!(a, b);
        let other = KozoSheet::generate(48, 32, &config, &mut rng(b"kozo-other")).expect("valid");
        assert_ne!(a.mottle, other.mottle);
        assert_ne!(a.ink_gain, other.ink_gain);
        // Cross-architecture canary.
        assert_eq!(sheet_digest(&a), GOLDEN_SHEET_SHA256);
    }

    /// See [`generation_is_deterministic_and_seeded`].
    const GOLDEN_SHEET_SHA256: &str =
        "7bd4994b6ad672baf9b9497c0612fe59f09f42308ec020cb300259a04f6f07ad";

    #[test]
    fn thread_count_does_not_change_the_sheet() {
        // More modes than one formation pass, so the passes run under both pools too.
        let config = small_paper(40.0, MODE_CHUNK + 44);
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool")
                .install(|| KozoSheet::generate(97, 43, &config, &mut rng(b"parity")))
                .expect("valid")
        };
        let one = run(1);
        let three = run(3);
        assert_eq!(sheet_digest(&one), sheet_digest(&three));
        assert_eq!(one, three);
    }

    /// Timing probe at the production size (not run by default):
    /// `cargo test --release --lib ember::paper::tests::timing -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing probe"]
    fn timing() {
        use std::time::Instant;
        let config = EmberConfig::default().paper;
        let geometry = SheetGeometry::new(3456, 2234, config.sheet_width_mm).expect("valid");
        let mut source = rng(b"kozo-timing");
        let start = Instant::now();
        let formation = Formation::draw(&config, &mut source);
        let count = fibre_count(&geometry, &config.fibres).expect("valid");
        let cover = fibre_cover(&geometry, &config.fibres, count, &mut source).expect("valid");
        let fibres = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let field = formation.render(&geometry);
        let render = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let sheet =
            KozoSheet::generate(3456, 2234, &config, &mut rng(b"kozo-timing")).expect("valid");
        let total = start.elapsed().as_secs_f64();
        let (f_mean, f_std) = mean_std(field.iter().copied());
        let (c_mean, _) = mean_std(cover.iter().copied());
        let c_max = cover.iter().copied().fold(0.0, f64::max);
        let (m_mean, m_std) = mean_std(sheet.mottle.iter().map(|&v| f64::from(v)));
        let g_min = sheet.ink_gain.iter().copied().fold(1.0f32, f32::min);
        println!(
            "3456x2234 kozo sheet: {total:.3} s total (modes + fibres {fibres:.3} s, formation \
             {render:.3} s); F mean {f_mean:.4} std {f_std:.4}; C mean {c_mean:.5} max {c_max:.4}; \
             mottle mean {m_mean:.4} std {m_std:.4}; ink gain min {g_min:.4}"
        );
    }
}
