# Ember edition: design

This is the maintainers' reference for `src/ember/`: the rules every line of it follows, the
coordinate conventions, and the exact recipe of each stage, with equations. The product view
(outputs, flags, the look in words, how to verify a package) is in
[ember-edition.md](ember-edition.md); the module docs in `src/ember/` hold the derivations and
the implementation notes.

**Section numbers are stable.** The code cites this document as `docs/ember-design.md §x`
(for example §3.7, §5.2 step 3). Keep §0–§8 and their numbered items when editing, and add new
material as new subsections or appendices.

The edition ports the museum-lab look `vermilion` ("the ember alone") of the Python/GPU
prototype (`wake/ns.py`, `wake2/lagdye.py`, `wake3/{looks,tone,render,spectral,pigments,sheet,
trace_torch,trace_triton,ledger,runstore}.py`, `estuary/source.py`; not part of this
repository) to a CPU-only Rust pipeline whose output is **bit-identical on every CPU
architecture and for every thread count**. It renders:

- one frame per checkpoint of the main video (`render::main_video_checkpoints`), so frame `i`
  of `ember.mp4` shows the same orbit step as frame `i` of `main.mp4`;
- a 16-bit sRGB still of the final recorded step (knot `steps - 1`), which is the last frame,
  byte for byte.

The look: black pine-soot sumi on a kozo sheet, and a vermilion (cinnabar) accent where the
fresh waters of two bodies meet. The ink is carried by a two-dimensional Navier–Stokes flow that
the three bodies (the selected orbit, projected onto its principal plane) stir as moving
Brinkman-penalised discs. The product adds two things to the prototype's look: **ember memory**
(the vermilion glows on after the meeting, §6.3) and a **presence-clamped tone law**, under which
ink diluted with clear water weakens linearly (§6.1). For unmixed water without ember memory the
look is the prototype's law. Only the loads are bit-exact: given the same fields `P` and `E`
(and `K ≤ best`), `loads` equals the prototype's rule bit for bit (§6.3). The pictures agree up
to the `f32` storage and per-frame ageing of `E` (§5.3): the prototype evaluates each ink
sample's exact age, while the port stores `E` as `f32` and multiplies it by the frame's fade
every frame (a relative rounding of about 2⁻²⁴ per frame).

---

## 0. Determinism and engineering rules

### 0.1 Bit-identical on every CPU

Chaotic flow amplifies a single differing ULP into a different picture, so every value must be
computed by the same IEEE-754 operations in the same order on x86_64 (with or without AVX2/FMA)
and aarch64 (NEON), with any number of threads.

- **Float primitives.** Only `+ - * /`, `sqrt`, `abs`, `floor`, `ceil`, `round`, `trunc`,
  comparisons, integer operations, `as` casts and `f64 → f32` rounding. All are exactly rounded.
- **Transcendentals through `crate::ember::math` only**, a facade over the pure-Rust `libm`
  crate pinned at `=0.2.16`: `exp`, `ln`, `pow`, `cbrt`, `tanh`, `sin`, `cos`, `sin_cos`. Never
  the std methods (`exp`, `ln`, `log*`, `powf`, `powi`, `sin`, `cos`, `tan`, `sin_cos`, `tanh`,
  `sinh`, `cosh`, `atan*`, `asin`, `acos`, `cbrt`, `hypot`, `exp_m1`, `ln_1p`, `exp2`,
  `mul_add`, for `f64` and `f32`), which call the platform's C library. The unit test
  `ember_sources_use_only_portable_math` in `math.rs` scans every ember source for these
  patterns, and `every_ember_source_file_is_guarded` fails until a new `src/ember/*.rs` file is
  added to its list. `libm_bits_are_pinned` pins the bits of every wrapper.
- **No architecture-dependent code paths**: no `mul_add`, no `core::arch`/`std::arch`
  intrinsics, no `cfg(target_feature)`, no runtime feature dispatch, no `rustfft`. Plain scalar
  Rust is fine even where LLVM vectorises it: Rust never contracts `a*b + c` into an FMA and
  never reassociates floating-point operations, so the native-CPU flags in
  `.cargo/config.toml` (`target-cpu=native`, `+avx2,+fma`) do not change a bit.
- **No host constant folding.** Never call a std transcendental on a constant (LLVM may fold it
  with the build host's libm). Go through `math` (ordinary Rust code, folded deterministically)
  or write the literal.
- **Parallelism must not change results.** Rayon only splits *independent outputs* (rows,
  columns, nodes, pixels, FFT blocks), each computed by a fixed sequential recipe. Reductions
  across threads are integer-only (counts, logical ANDs of flags) or exact maxima/minima of
  non-NaN values: never `sum`, `reduce` or `fold` over floats, never float atomics. Every module
  that parallelises has a test that compares rayon pools of 1 and 3 threads bit for bit
  (`ThreadPoolBuilder::new().num_threads(n).build()?.install(..)`).
- **Order.** No `HashMap` iteration in anything that affects output. Sorts use `total_cmp` with an
  index tie-break.
- **No subnormals in hot paths.** Tiny values are flushed to exactly 0 by a comparison (the ink
  fields at `10⁻¹²`, the fluid's integrating factors when the exponent exceeds 700). FTZ/DAZ are
  never set.
- **Check `is_finite` at stage boundaries** and return `EmberError::NonFinite` instead of hashing
  NaN: every fluid step, the traced ink origins of every frame, the shaded pixels.
- **Signed zero and NaN.** Minima, maxima and clamps are explicit comparisons with documented
  `±0`/NaN behaviour (never `f64::min`/`f64::max` where `±0` could matter); normalise `-0` with
  `x + 0.0` where it could reach a digest.
- **The orbit.** The integrator (`sim.rs`) is portable too: its gravity kernel cubes distances by
  multiplication, not `powi`.

### 0.2 Golden canaries

Unit tests pin SHA-256 digests of small but complete computations: the forward FFT of a fixed
48 × 32 field (`fft.rs`), a fluid snapshot (`fluid.rs`), an ink remap (`ink.rs`), the paper sheet
(`paper.rs`), the optics' golden table and bits (`optics.rs`), the libm bits (`math.rs`), and the
end-to-end frame stream and still of a small render (`tests/ember_determinism.rs`, which also
pins that render's two vermilion statistics). CI runs them on x86_64 Linux and aarch64 macOS,
and once more on x86_64 built for `x86-64-v3` (AVX2/FMA code generation; `ci/README.md`). A
digest that changes on one architecture only is a determinism bug; a digest that changes
everywhere is an algorithm change.

Any change that alters rendered bits must bump `certificate::ALGORITHM_VERSION`, re-bless every
affected digest and verify the new values on both architectures:

- **Unit goldens** (`fft.rs`, `fluid.rs`, `ink.rs`, `paper.rs`, `optics.rs`, `math.rs`) have no
  bless switch. Each asserts with `assert_eq!`, so a failing test prints its new value next to
  the recorded one. Run `cargo test --release --lib ember` and copy the printed values into the
  constants.
- **End-to-end goldens**: `EMBER_BLESS=1 cargo test --release --test ember_determinism --
  --include-ignored --nocapture` prints them: the frame-stream and still digests
  (`GOLDEN_FRAMES_SHA256`, `GOLDEN_STILL_SHA256`), and the video renders' `frames_with_cinnabar`
  and peak cinnabar node count (`GOLDEN_FRAMES_WITH_CINNABAR`, and `GOLDEN_PEAK_CINNABAR_NODES`
  = `peak_frame_cinnabar_fraction · WIDTH·HEIGHT·4`, from the printed stats). Update all four
  constants. Every render must print the same values (see the file's header for the full
  procedure).

A `libm` bump is such a change, and it re-blesses **every** ember golden, unit and end-to-end:
the `math` wrappers feed the FFT twiddles, the fluid, the ink, the paper and the optics.
Dependabot ignores `libm`; bump it by hand.

### 0.3 Code rules

1. **Lints.** The crate denies `warnings` and `missing_docs`; CI runs
   `cargo clippy --all-targets -- -D warnings` with clippy pedantic (allowed crate-wide: the
   `cast_*` lints, `float_cmp`, `many_single_char_names`, `similar_names`, `too_many_lines`,
   `unreadable_literal`, `items_after_statements`, `must_use_candidate`, `missing_errors_doc`,
   `missing_panics_doc`, and a few more in `Cargo.toml`) and `cargo fmt --check`
   (`max_width = 100`, `use_small_heuristics = "Max"`). `clippy.toml`:
   `too-many-arguments-threshold = 8`, `cognitive-complexity-threshold = 40` (use parameter
   structs).
2. **Documentation.** Every item, `pub(crate)` and private included, has a doc comment that
   states units, conventions and invariants. Module docs explain the physics and the
   mathematics with equations.
3. **Tests.** Unit tests live in the module. Every numerical kernel has an analytic or reference
   oracle, a property or invariance test, and a thread-count parity test where it is parallel.
   Tests use tiny grids: the whole ember unit suite stays within seconds.
4. **Errors.** Fallible constructors return `EmberResult<T>` (`error.rs`). No `unwrap` or
   `expect` outside tests except on provably infallible invariants, with the reason stated.
5. **Performance.** The production render (1,000,000 steps, 3456 × 2234 pixels, 1802 frames)
   must be reasonable on a 16-core CPU: buffers are allocated once and reused, hot loops do not
   allocate, data is structure-of-arrays. Correctness and determinism come first.

---

## 1. Geometry and coordinate conventions

- **World units.** The canvas is `x ∈ [-a, a]`, `y ∈ [-1, 1]` with `a = width / height` (the
  aspect; 3456 × 2234 gives `a ≈ 1.547`); `+y` is up. Lengths are world units, times fluid time
  units (§3.6: the median body speed is the reference speed 1).
- **Output pixels** `(row, col)`, row 0 at the **top**. The pixel size is `s = 2/height` (square
  pixels); pixel centre `x = -a + (col + 0.5)·s`, `y = 1 - (row + 0.5)·s`.
- **Ink nodes** (`ink::NodeGrid`). Supersampling `q = raster.supersample` (default 2), node
  spacing `hn = s/q`, and `M = ⌈raster.ink_margin / hn⌉` margin nodes on every side (default
  margin 0.15). The grid has `rows = q·height + 2M`, `cols = q·width + 2M` nodes; node `(r, c)`
  sits at `x = -a + (c - M + 0.5)·hn`, `y = 1 - (r - M + 0.5)·hn`. Output pixel `(row, col)` is
  the mean of the `q × q` nodes `r = M + q·row + i`, `c = M + q·col + j`, `i, j ∈ 0..q`, in
  row-major order. The continuous node index of a world point is
  `fr = (1 - y)/hn - 0.5 + M`, `fc = (x + a)/hn - 0.5 + M`. Values outside the grid are 0
  (never-inked water).
- **Fluid grid** (`fluid::FluidGrid`). A doubly periodic box `lx × ly` of `nx × ny` nodes with
  `dx = dy`; node `(i, j)` (row `i`, column `j`) sits at `x = -lx/2 + j·dx`, `y = -ly/2 + i·dx`.
  Row 0 is the **bottom** (`y` minimum), the opposite of the image rows; storage is row-major,
  `index = i·nx + j`. `FluidGrid::for_canvas(a, rows, margin)`:
  `ny = next_smooth_even(rows)`, `dx = 2·(1 + margin)/ny`,
  `nx = next_smooth_even(⌈2·(a + margin)/dx⌉)`, `lx = nx·dx`, `ly = ny·dx`, where
  `next_smooth_even(n)` is the smallest even 5-smooth integer `≥ n` (prime factors 2, 3, 5 only).
  The default (1024 rows, margin 0.35) gives 1440 × 1024 at `a ≈ 1.547`.
- **Time.** Fluid time runs over `t ∈ [0, T]` with `T` the orbit's duration (§3.6). Recorded
  knot `k ∈ [0, N)` (`N = steps`; knot `N - 1` is the still) sits at `t_k = T·k/(N - 1)`,
  evaluated as `(T·k)/(N - 1)` except that `t_{N-1} = T` exactly.
- **Paper.** The sheet is measured in millimetres with `x` right and `y` **down** from the
  top-left corner (§7.5).

---

## 2. Module map

| Module | Responsibility | Key items |
|--------|----------------|-----------|
| `mod.rs` | Overview, module tree, public re-exports. | |
| `error.rs` | Errors. | `EmberError`, `EmberResult` |
| `config.rs` | Every tunable, its defaults (Appendix A) and range validation. | `EmberConfig` and its sections, `validate` |
| `math.rs` | The libm facade, portable comparison primitives, the source guard. | `exp`, `ln`, `pow`, `cbrt`, `tanh`, `sin`, `cos`, `sin_cos` |
| `orbit.rs` | Orbit → three moving discs: PCA projection and time map (§3). | `BodyTrack`, `BodyMotion`, `BodyState`, `Projection` |
| `fft.rs` | Deterministic mixed-radix FFT and real 2-D transforms (§4.1). | `Rfft2d`, `Spectrum` |
| `fluid.rs` | The Navier–Stokes wake solver (§4.2). | `FluidGrid`, `WakeSolver`, `Snapshot`, `FluidStats` |
| `trace.rs` | Backward characteristics with gated soak-zone contacts (§5.2). | `Tracer`, `FlowWindow`, `ContactRules`, `Trace` |
| `ink.rs` | Ink node grid, ink fields and the per-frame remap (§5.3). | `NodeGrid`, `InkFields`, `InkDecay`, `remap` |
| `look.rs` | Tone law, vermilion accent and ember memory → pigment loads (§6). | `Look`, `InkLoads` |
| `optics.rs` | 36-band Kubelka–Munk/Saunderson shading and the display encoding (§7.1–7.4). | `Optics` |
| `paper.rs` | The kozo sheet: formation and fibres → mottle and ink gain (§7.5). | `KozoSheet`, `PaperSample` |
| `pipeline.rs` | Planning, the frame loop, shading, digests (§8.1–8.3). | `plan_ember`, `render_ember`, `EmberRequest`, `EmberSummary` |
| `certificate.rs` | The determinism certificate, written and read back (§8.4). | `EmberCertificate`, `schedule_sha256` |

**Public API** (`three_body_problem::ember`): `EmberConfig`, `EmberError`, `EmberResult`,
`EmberRequest`, `EmberMode`, `EmberFrame`, `EmberPlan`, `EmberSummary`, `EmberStats`,
`EmberTimings`, `EmberProjection`, `plan_ember`, `render_ember`, `EmberCertificate`,
`CertificateError`, and the `certificate`, `config`, `error` and `pipeline` modules. Everything
else is `pub(crate)`. `fft_bench` is a hidden export for `benches/ember_fft.rs`.

**Around the module**: `app.rs` (`preflight_ember_edition`, `render_ember_edition`, the paper
seed and the frame schedule), `main.rs` (stage order and exit status, §8.5), `render/video.rs`
(the `*_srgb` encoders), `examples/ember_render.rs` (verification and look development, §8.6),
`tests/ember_determinism.rs` (the end-to-end golden test) and `benches/ember_fft.rs`.

---

## 3. Orbit → moving discs (`orbit.rs`)

The input is the raw recorded orbit, `positions[body][knot] ∈ ℝ³`: exactly three bodies with
`N ≥ 2` knots each, all finite, straight from the integrator (not drift- or view-transformed).
`BodyTrack::new` builds the discs in the following steps. All sums run in **scan order**
(knot-major, then body) with Neumaier-compensated summation.

**3.1 Bounding box.** `low`/`high` per axis over all `3N` points; `origin = (low + high)/2`,
`extent = max_axis(high - low)`. A non-finite or non-positive extent is `DegenerateOrbit`.

**3.2 Moments.** Normalised points `q = (p - origin)/extent`; their mean `μ` over all `3N`
points and the covariance `C = Σ (q - μ)(q - μ)ᵀ / 3N`.

**3.3 Principal axes.** Symmetric 3 × 3 eigen-decomposition by **cyclic Jacobi**: fixed pivot
order `(0,1), (0,2), (1,2)`; rotation from `θ = (a_qq - a_pp)/(2·a_pq)`,
`t = sign(θ)/(|θ| + sqrt(θ² + 1))`, `c = 1/sqrt(t² + 1)`, `s = t·c`; sweeps until every
`|off-diagonal| ≤ 10⁻³⁰⁰` or 64 sweeps. Eigenpairs sorted by eigenvalue, descending
(`total_cmp`, index tie-break). The orbit must span a plane: `λ₁ > 10⁻¹²·λ₀`, else
`DegenerateOrbit`. `e₀` is normalised and `e₁` Gram–Schmidt-orthogonalised against it.

**3.4 Anchor rule.** Eigenvector signs are arbitrary, so each axis `a ∈ {0, 1}` is oriented by
its anchor: with `v = (q - μ)·e_a`, the anchor is the first point in scan order whose
`|v| ≥ max|v|·(1 - 10⁻¹²)`; flip `e_a` if the anchor's `v < 0`.

**3.5 Projection.** `P = ((q - μ)·e₀, (q - μ)·e₁)`; per axis `centre = (min P + max P)/2`,
`half = (max P - min P)/2`, and `scale = fill / max(half_x/a, half_y)` (`fill =
projection.fill`, default 0.78). A knot's world position is `pos_k = (P_k - centre)·scale`, so
the bodies span `±fill·a` in `x` or `±fill` in `y`, whichever binds. The projected knots are
stored once (3 × N × 2 `f64`, structure of arrays).

**3.6 Duration.** On segment `left`, the velocity per source fraction is
`dpos/df = (pos_{left+1} - pos_left)·(N - 1)`. The median of `|dpos/df|` over the 3 × 200,001
samples `f_j = j/200000` (`left = min(⌊f·(N-1)⌋, N-2)`), taken as the exact middle order
statistic, gives the duration `T = median / reference_speed`: the median body speed is exactly
`reference_speed` (1), which fixes the Reynolds number.

**3.7 State at time `t`.** `f = clamp(t/T, 0, 1)`, `fi = f·(N - 1)`,
`left = min(⌊fi⌋, N - 2)`, `w = fi - left`; `pos = pos_left + w·(pos_{left+1} - pos_left)` and
`vel = (pos_{left+1} - pos_left)·(N - 1)/T` (the outgoing segment's velocity).

**3.8 Speed look-ahead** (for the solver's CFL). A table `speed_i = max_b |vel_b(T·i/400000)|`,
`i ∈ [0, 400000]`, with block maxima of 64 entries. `speed_bound(t)` is the maximum of
`speed_i` over `i ∈ [max(i₀ - 2, 0), min(i₀ + 400, 400001))`, where `i₀` is the first entry with
`T·i/400000 ≥ t`.

**3.9 Interface.**

```rust
pub(crate) struct BodyState { pub position: [f64; 2], pub velocity: [f64; 2] }
pub(crate) trait BodyMotion: Sync {          // BodyTrack; tests implement analytic motions
    fn bodies_at(&self, t: f64) -> [BodyState; 3];
    fn speed_bound(&self, t: f64) -> f64;     // bound of every body speed just after t
}
```

---

## 4. Navier–Stokes wake solver (`fft.rs`, `fluid.rs`)

### 4.1 FFT

- **Own implementation**, no `rustfft`. Self-sorting Stockham mixed-radix transforms for every
  5-smooth length, with radix-4 butterflies first, then at most one radix 2, then 3s and 5s,
  ping-ponging between two buffers (no bit reversal). The inverse is
  `IDFT(z) = swap(DFT(swap(z)))` (real and imaginary parts exchanged), which is exact.
- **Twiddles** `ω_n^e` come from `math::sin_cos` on the smallest arc the symmetries of `n` allow
  (the first octant when `8 | n`), extended by exact symmetries, with the special angles set to
  their correctly rounded values; radix 3 and 5 use correctly rounded literals of `sin(π/3)`,
  `cos(2π/5)`, `cos(4π/5)`, `sin(2π/5)`, `sin(4π/5)`.
- **Batches.** Kernels transform `LANES = 4` independent rows or columns side by side
  (`[n][LANES]` structure-of-arrays blocks, real and imaginary parts apart) with identical
  scalar arithmetic per lane, so a lane's bits do not depend on its batch.
- **`Rfft2d`**: a real `ny × nx` row-major field ↔ its half spectrum `ny × (nx/2 + 1)`, stored at
  `ky·(nx/2 + 1) + kx` (numpy `rfft2` layout). Forward, unnormalised:
  `X[ky][kx] = Σ_{i,j} x[i][j]·e^{-2πi(kx·j/nx + ky·i/ny)}`. The inverse has numpy `irfft2`
  semantics (the imaginary parts of the self-conjugate `kx = 0` and `kx = nx/2` entries are
  ignored) and one multiplication by `1/(nx·ny)`. Two real rows are packed into one complex row
  and separated with the Hermitian identities; column blocks run in parallel. `nx` and `ny` must
  be even and 5-smooth. `Spectrum { re, im }` is split complex; transforms take `&mut self` to
  reuse scratch. `forward_with`/`inverse_with` skip wavenumber columns known to be zero (the
  dealiased band).
- **Tests**: against a naive DFT (relative `10⁻¹²`) for many sizes, round trip, Parseval,
  linearity, batch independence, 1 vs 3 threads, and a golden SHA-256 of the forward transform
  of a fixed 48 × 32 field. Benchmark: `benches/ember_fft.rs` (1440 × 1024 round trip).

### 4.2 Solver

Vorticity–streamfunction form on the doubly periodic box, pseudo-spectral, 2/3 dealiasing,
`f64` state `ω̂` (half spectrum). It mirrors `wake/ns.py` in structure.

```text
∂ω/∂t + u·∇ω = ν∇²ω,   ω = ∂v/∂x - ∂u/∂y,   ψ̂ = ω̂/k²,   u = ∂ψ/∂y,   v = -∂ψ/∂x
ν = reference_speed·(2·body_radius)/reynolds
```

- **Wavenumbers.** `kx_j = 2πj/lx` (`j ≤ nx/2`), `ky_i = 2π·(i < ny/2 ? i : i - ny)/ly` (the
  Nyquist row is negative), `k² = kx² + ky²`. Dealias mask
  `|kx| < (2/3)·π/dx && |ky| < (2/3)·π/dx` (strict), evaluated exactly in integers as
  `3|m| < n`; masked modes of the state are exactly 0.
- **Dissipation.** `k_c = (2/3)·π/dx`, `H = (k²/k_c²)^12` by repeated squaring
  (`x², x⁴, x⁸, x¹⁶, x²⁴ = x¹⁶·x⁸`), `D = ν·k² + (hyperviscosity/dx)·H`.
- **Velocity.** `û = i·ky·ω̂/k²`, `v̂ = -i·kx·ω̂/k²` (0 at `k = 0`).
- **Advection.** `N(ω̂) = -mask·FFT(u·∂ₓω + v·∂ᵧω)` with `∂ₓω = IFFT(i·kx·ω̂)`,
  `∂ᵧω = IFFT(i·ky·ω̂)`.
- **Lawson IF-RK4** of step `h`: `e₂ = exp(-D·h/2)` through `math::exp` (0 when
  `D·h/2 > 700`), `e₁ = e₂²` (0 when `D·h > 700`):

  ```text
  k1 = N(ω̂)                   k2 = N(e₂·(ω̂ + h/2·k1))
  k3 = N(e₂·ω̂ + h/2·k2)       k4 = N(e₁·ω̂ + h·e₂·k3)
  ω̂ ← e₁·ω̂ + h/6·(e₁·k1 + 2·e₂·(k2 + k3) + k4)
  ```

- **Forcing split**, after every RK4 step, with the bodies at the new time `t + h`:
  1. `(u, v) = velocity(ω̂)` in physical space.
  2. Per body `b` (centre `p`, velocity `V`, radius `R = body_radius`, edge `W = mask_width`):
     `d = |x - p|` (Euclidean, no periodic wrap), `c_b = ½·(1 - tanh((d - R)/W))` through
     `math::tanh`, evaluated only where `(d - R)/W < 20` and exactly 0 beyond (the neglected
     values are below `5·10⁻¹⁸`; the pinned libm's `tanh` already rounds to 1 from about 19.07,
     so the cut changes no bit). `χ = max_b c_b`, `ū = Σ c_b·V_b / max(Σ c_b, 10⁻¹²)`.
  3. Implicit Brinkman with `a = χ / brinkman_eta_ratio`: where `a > 0`,
     `u ← (u + a·ū)/(1 + a)` (and `v` likewise).
  4. Curl and sponge: `ω = IFFT(i·kx·FFT(v) - i·ky·FFT(u))·exp(-σ·h)` with
     `σ(x, y) = sponge_rate·max(ramp(x; a + sponge_pad, lx/2 - 4dx), ramp(y; 1 + sponge_pad,
     ly/2 - 4dx))`, `ramp(c; in, out) = s²(3 - 2s)`, `s = clamp((|c| - in)/(out - in), 0, 1)`
     (precomputed; applied only where `σ > 0`).
  5. `ω̂ = mask·FFT(ω)`.
  6. `u_max = max sqrt(u² + v²)` of the penalised velocity (an exact maximum).
- **Time stepping**, `advance_to(target, bodies)`: until `t = target`, take
  `h_cfl = min(cfl·dx/max(u_max, speed_bound(t), 10⁻⁶), max_dt)` and `rem = target - t`; if
  `rem ≤ h_cfl·(1 + 10⁻⁹)` step `rem` and set `t = target` exactly; else if `rem < 2·h_cfl`
  step `rem/2`; else step `h_cfl`. A target before `t` is an error. Initially `ω̂ = 0`, `t = 0`,
  `u_max = 0`.
- **Snapshots.** `snapshot_into` stores `u`, `v` and the dealiased `ω = IFFT(ω̂)` at the current
  time as **`f32`** (IEEE rounding), with the time.
- **Cost.** 26 real 2-D transforms per step: 20 for the four advection evaluations, 6 for the
  forcing; the 23 whose input or output is dealiased skip the zero columns.
- **Statistics** (deterministic): steps, smallest and largest `h`, largest `u_max`. A non-finite
  flow, body state or step is `EmberError::NonFinite`.

---

## 5. Ink (`trace.rs`, `ink.rs`)

### 5.1 Flow window and snapshot cadence

A frame interval `(t_prev, t_frame]` is covered by `S ≥ 1` snapshot intervals, with snapshots at
`τ_0 = t_prev < τ_1 < … < τ_S = t_frame`, `τ_s = t_prev + (t_frame - t_prev)·s/S` (exactly
`t_frame` for `s = S`), all produced by the solver landing exactly on them. Between the frames
at knots `from` and `to`:

```text
S = max(⌈Δt / max_snapshot_interval⌉, ⌈max_b path_b / (max_snapshot_travel·body_radius)⌉, 1)
```

with `Δt = t_to - t_from` and `path_b` the length of body `b`'s recorded polyline through the
knots `from..=to` (not its displacement, which misses a body that turns back); `S = 0` when
`from = to`. No snapshot interval is longer than `max_snapshot_interval` (default 0.0025) or
lets a body move more than `max_snapshot_travel` radii (default 0.5). `FlowWindow` borrows the
`S + 1` snapshots and the body positions at each snapshot time; a single snapshot is an empty
window.

### 5.2 Backward trace (`trace_back(window, rules, start) → Trace`)

Trace the parcel at world point `start` at time `τ_S` back to `τ_0`, returning its origin and,
per body, the **latest** contact time in the window (or none). For `s = S … 1`, with
`t1 = τ_s`, `t0 = τ_{s-1}`, `dt = t1 - t0 > 0`, `h = -dt`:

1. **RK4 in time**, velocity bilinear in space (periodic wrap on the fluid grid, `f32` samples,
   `f64` arithmetic) and linear in time:
   `k1 = U1(x)`, `x2 = x + h/2·k1`, `k2 = ½(U1(x2) + U0(x2))`, `x3 = x + h/2·k2`,
   `k3 = ½(U1(x3) + U0(x3))`, `k4 = U0(x + h·k3)`, `xn = x + h/6·(k1 + 2k2 + 2k3 + k4)`.
   Positions are carried unwrapped (world coordinates); only lookups wrap.
2. **Gate** (`wc = vorticity_gate`, default 40; `≤ 0` disables it): `o0` is the periodic
   Catmull-Rom sample of `ω(t0)` at `xn`, `o1` the one carried from the previous step (initially
   `ω(t_frame)` at `start`); Catmull-Rom weights at offset `t ∈ [0, 1)`:
   `w0 = -½t³ + t² - ½t`, `w1 = 3/2t³ - 5/2t² + 1`, `w2 = -3/2t³ + 2t² + ½t`, `w3 = ½t³ - ½t²`.
   With `a1 = |o1|`, `a0 = |o0|`:
   `sc = clamp((wc - a1)/(a0 ≠ a1 ? a0 - a1 : 10⁻¹²), 0, 1)`,
   `g0 = a1 > wc ? 0 : (a0 > wc ? sc : 1)`, `g1 = a1 > wc ? (a0 > wc ? 1 : sc) : (a0 > wc ? 1 : 0)`.
3. **Disc test** per body `i`: `a = x - b_i(t1)`, `b = xn - b_i(t0)`,
   `r = reach = body_radius + soak_depth`, `d = b - a`, `A = |d|²`, `B = a·d`, `C = |a|² - r²`,
   `disc = B² - A·C`;
   `s0 = clamp((-B - sqrt(max(disc, 0)))/max(A, 10⁻²⁴), 0, 1)`,
   `s1 = clamp((-B + sqrt(max(disc, 0)))/max(A, 10⁻²⁴), 0, 1)`; if `disc ≤ 0` then `s1 = s0`;
   if `A < 10⁻²⁰` then `s0 = 0`, `s1 = C < 0 ? 1 : 0`.
4. **Contact**: `s_lo = max(s0, g0)`, `s_hi = min(s1, g1)`, `hit = s_hi > s_lo`,
   `t_lo = t1 - s_hi·dt`, `t_hi = t1 - s_lo·dt`.
5. **Record** once, latest first: if body `i` has no record, `hit` and `t_lo ≤ t_valve`, record
   `min(t_hi, t_valve)`. At the end, records `< t_on` are dropped. (`t_valve = T - valve_lead`,
   `t_on = pre_roll`.)
6. `x ← xn`, `o1 ← o0`.

`Tracer` implements this with shortcuts that provably return the same bits (a test compares it
with a literal eager implementation): the gate is evaluated only when a disc interval is
non-empty; no contact work once every body has a record, when `t1 - dt > t_valve`, or when
`t1 < t_on`; a disc test with `disc ≤ 0` stops before the square root. `trace_lanes` advances
four independent parcels in lock step (each lane runs exactly the single-parcel operations).

### 5.3 Fields and remap

Each ink node carries five `f32` fields (structure of arrays): the presence `P ∈ [0, 1]`, the
freshness `E_i ∈ [0, 1]` of each body, and the ember `K ∈ [0, 1]`. For a parcel last inked by
body `i` at `t*_i` that has not mixed since, `P = 1` and `E_i = exp(-(t_frame - t*_i)/fresh_tau)`
(0 if body `i` never inked it). Interpolation mixes all five linearly, which models dilution at
the node scale.

`remap(prev, next, nodes, window, rules, decay, look)`, per node, rows in parallel:

1. `tr = trace_back(window, rules, node position)` (§5.2).
2. Where a field is not overwritten below, sample `prev` at `tr.origin` by **clamped
   Catmull-Rom** on the node grid: 4 × 4 taps, taps outside the grid read 0, and the result is
   clamped to the range of the inner 2 × 2 taps (monotone: no overshoot, no negative ink).
3. Update, with `Δt = t_frame - t_prev`:

   ```text
   P'   = any record ? 1 : P(X)·floor_fade             floor_fade = floor_tau ? exp(-Δt/floor_tau) : 1
   E'_i = record_i ? exp(-(t_frame - t*_i)/fresh_tau) : E_i(X)·fresh_fade
                                                       fresh_fade = exp(-Δt/fresh_tau)
   K'   = ember_tau ? max(meeting(P', E'), K(X)·exp(-Δt/ember_tau)) : 0
   ```

   `meeting` is the look's `best` (§6.2), computed from the `f32`-rounded `P'` and `E'` exactly
   as the shader will see them. Without ember memory (`ember_tau: null`) the field stores exactly
   0, not `f32(meeting)`: rounding `meeting` to `f32` could store a value above `best`, and the
   loads would no longer be the prototype's (§6.3). Values below `10⁻¹²` are flushed to exactly
   `+0`, and the results are stored as `f32`.

A node whose traced origin is not finite (a NaN or infinite velocity on its pathline) inherits
clear water and is counted; the pipeline turns any such count into `EmberError::NonFinite`.
Tests cover pure translation and rotation flows, a disc inking exactly its swept band, analytic
moving-disc contact times, the valve and the pre-roll, zero outside the grid, a golden digest
and 1 vs 3 threads.

---

## 6. Look (`look.rs`)

The look maps a node's `P`, `E_0..2` and `K` to two pigment loads, carbon (pine soot) and
cinnabar. All arithmetic is exactly rounded `f64`, with one `exp` at construction.

### 6.1 Tone law

With the hold gain `g = exp(hold/fresh_tau)`, precomputed:

```text
h_i    = min(P, g·E_i)                       presence-clamped hold
c_i    = floor·P + (1 - floor)·h_i           species strength of body i's ink
c_mono = floor·P + (1 - floor)·max_i h_i     shared-carbon strength (the youngest ink wins)
```

For an unmixed parcel (`P = 1`) `g·E_i = exp(-(t_ref - t*_i)/fresh_tau)` with
`t_ref = t_frame - hold`, so `c_i = floor + (1 - floor)·exp(-max(t_ref - t*_i, 0)/fresh_tau)`:
the prototype's reservoir feed `ExpFeed(t_ref, fresh_tau, floor)` at the uptake time. That is
exact as a law. The stored `E_i` carries the `f32` rounding and per-frame ageing of §5.3, so
the values agree with the prototype's to about 2⁻²⁴ per frame, not bit for bit. Ink younger
than `hold` is at full strength; older ink decays towards the `floor` wash.

**Dilution.** The hold saturates at the presence, not at 1, so the law is linear under dilution
with clear water: a node holding a fraction `w` of an unmixed parcel of age `α` has `P = w`,
`E_i = w·exp(-α/fresh_tau)` and hence `h_i = w·min(1, g·exp(-α/fresh_tau))`. (Saturating at 1
would draw ink diluted down to `1/g ≈ 1.6 %` at full strength, and let two bodies' 2 % dilutions
meet at full strength.) Mixing two *inked* parcels stays concave: the youngest ink wins in
`c_mono`, and the 50/50 interface of two fresh inks is a full-strength meeting. With
`floor_tau` set, `P` also carries the floor wash's fade, so the cap fades the whole deposit.

### 6.2 The vermilion accent

Cinnabar forms only where the waters of two bodies meet while both are fresh:

```text
best = max over pairs (0,1), (1,2), (0,2) of min(c_i, c_j)
best = best > meeting_threshold ? best : 0
```

With the production constants a meeting needs both uptakes no older than
`fresh_tau·ln((1 - floor)/(0.3 - floor)) ≈ 0.1445` before `t_ref` (in unmixed water; diluted
water needs `P > 0.3` as well).

A body that never inked a parcel has `c_i = floor·P` where the prototype has 0. The loads agree
because `floor ≤ meeting_threshold` (4·10⁻⁴ ≤ 0.3): such a `c_i` can never pass the strict
meeting test. This inequality is a precondition of `Look::new` and checked by
`EmberConfig::validate`: the fields keep one presence for all bodies and flush old freshness to
0, so they cannot tell which bodies ever inked old water, and a look with
`floor > meeting_threshold` is not representable.

### 6.3 Ember memory

In the prototype the vermilion exists only while both inks are fresh (about 0.6 fluid time
units), so the still of the final step usually has none. With `ember_tau` set (default 1.0) the
cinnabar **glows on**: the ember field `K` (in the units of `best`) is advected, diluted and
cooled by the remap (§5.3, `K' = max(best, K·e^{-Δt/ember_tau})`), and the look shows it
crisply while it is hotter than the meeting threshold:

```text
red      = max(best, K > meeting_threshold ? K : 0)
carbon   = red > 0 ? carbon_keep·max(c_mono, red) : c_mono
cinnabar = cinnabar_strength·red
```

- **Crisp display.** An ember shows at its full strength above the threshold and not at all
  below it, exactly like a fresh meeting: it never fades into a pale pink wash. A full-strength
  ember goes out about `ember_tau·ln(K₀/meeting_threshold) ≈ 1.2·ember_tau` after the meeting.
- **The still may have no vermilion.** Memory lengthens the window but does not guarantee red
  in the still. The still shows vermilion only when two bodies' waters met within about
  `ember_tau·ln(K₀/meeting_threshold)` (≈ 1.2 fluid units at the default) before the final step,
  with both uptakes before the valve. That is a short window (the default orbit of seed
  `0x46205528` lasts 19.1 units), so many stills are pure sumi. That seed's waters rarely meet
  at all, and its production still has `still_cinnabar_fraction` 0. The video shows every
  meeting. The certificate records this: `stats.still_cinnabar_fraction`,
  `stats.frames_with_cinnabar` and `stats.peak_frame_cinnabar_fraction` (§8.2).
- **Soot.** The ember keeps the soot it formed with, `carbon_keep·max(c_mono, red)`, so a glowing
  ember on old, pale water stays the deep vermilion of a fresh meeting instead of a graphic pure
  red. For a fresh meeting `best ≤ c_mono`, so the carbon is the prototype's
  `carbon_keep·c_mono`. The rule also applies to fresh single-body ink laid into water that
  still glows: it renders vermilion, not black (the water a body carries glows with the ember
  of its last meeting).
- **Without memory** (`ember_tau: null`) the fields store `K = 0`, and with any `K ≤ best` the
  loads are the prototype's rule bit for bit:
  `carbon = c_mono·(best > 0 ? carbon_keep : 1)`, `cinnabar = cinnabar_strength·best`.

A smooth variant in which cooled embers faded out gradually was tried and rejected: its pink
tails covered half the sheet.

### 6.4 Validity

`EmberConfig::validate` enforces what the look relies on: `0 ≤ floor < 1`,
`floor ≤ meeting_threshold ≤ 1`, `fresh_tau > 0`, `hold ≥ 0`, and
`hold/fresh_tau ≤ ln(10⁶) ≈ 13.8`. The last bound comes from the flush: freshness below `10⁻¹²`
is stored as 0, and the hold gain `g` multiplies it, so ink crossing the flush loses
`g·10⁻¹²` of strength in one step; the bound keeps that below `10⁻⁶` (production:
`g = e^{0.5/0.12} ≈ 64.5`, a cut of `6.5·10⁻¹¹`).

---

## 7. Optics and paper (`optics.rs`, `paper.rs`)

### 7.1 Reflectance

36 bands, `λ = 380, 390, …, 730` nm. The ink sits in the paper's fibres (Duncan additivity of
absorption `K` and scattering `S`, semi-infinite sheet). For pigment loads `L0` (pine soot) and
`L1` (cinnabar), and the pixel's paper mottle `m` and ink gain `g` (§7.5):

```text
k_ink = L0·kt_soot + L1·kt_cin + (m - 1)·K_p        (the mottle term is added before the gain)
s_ink = L0·st_soot + L1·st_cin
K     = K_p + g·k_ink,   S = 1 + g·s_ink            (S_p = 1: the Kubelka–Munk unit)
R∞    = 1/(1 + a + sqrt(a·(a + 2))),   a = K/S
R     = ks + (1 - k1)(1 - k2)·R∞/(1 - k2·R∞)        Saunderson: ks = 0.0164, k1 = 0.04, k2 = 0.6
R    -= (ks - ks_film)·(1 - exp(-max(L0 + L1, 0)/c_film))    nikawa film: ks_film = 0.005, c_film = 0.35
XYZ   = Σ_b R_b·W_b                                  gallery light, perfect diffuser Y = 1
```

`kt`/`st` are the pigment tables times their strength scales (soot 1.2440916174042291, cinnabar
10.962081999030831), which make one load unit of every pigment as strong as one unit of neutral
carbon (peak optical density 0.5 over kozo). The paper's `K_p` is derived at construction from
its observed reflectance by the inverse Saunderson correction and the Kubelka–Munk remission
function, `R' = (R_obs - ks)/((1 - k1)(1 - k2))`, `R_int = R'/(1 + k2·R')`,
`K_p = (1 - R_int)²/(2·R_int)`, and checked against the prototype's `PAPER_K_P`. The Saunderson
body term is evaluated in the algebraically identical form
`(1 - k1)(1 - k2)·S/((1 - k2)·S + K + sqrt(K·(K + 2S)))` (one division, one square root). The
kozo sheet keeps `K ≥ 0` (`m ≥ 0.3`, `g ≤ 1`), so every band lies in `(0, 1)`.

A pixel is the mean XYZ of its `q × q` nodes, in fixed row-major order; a node without pigment
uses its pixel's cached bare-paper XYZ, and a pixel whose nodes are all bare is its cached
bare-paper code value.

### 7.2 Tables and their provenance

`optics.rs` embeds exact decimal copies of the prototype's tables: `KOZO_R_OBS`,
`SOOT_PINE_K`/`_S`, `CINNABAR_K`/`_S` (per unit load, before the strength scale), `W_GALLERY`
(36 × 3), `WHITE_XYZ`, `M_TOTAL` and the medium black; a test checks each against checksums
computed independently from the source text. They are embedded rather than recomputed, which
avoids porting Mie theory and the spectrum fit:

- **Kozo**: a Jakob–Hanika 2019 sigmoid-polynomial spectrum fitted to sRGB (0.935, 0.915, 0.865).
- **Pine soot**: Mie spheres (`m = 1.95 + 0.79i` in a medium of index 1.5, log-normal
  `d_g = 200` nm, `σ_g = 1.8`), `K = 2µ_a`, `S = 0.75µ_s(1 - g)`, normalised so that load 1
  reaches L* 19.87.
- **Cinnabar**: `n = 2.97` with an Urbach absorption edge (gap 2.16 eV, 20 meV, α 2/µm),
  `d_g = 1.5` µm, `σ_g = 1.6`, in glue of index 1.53, normalised to a peak optical density of
  1.2.
- **Strength scales**: bisected so that each pigment's peak optical density over kozo is 0.5,
  relative to neutral carbon.
- **`W_GALLERY`**: the exact 1 nm integral of the CIE 1931 2° observer times the CIE LED-V1
  gallery illuminant times a piecewise-linear band hat, normalised to `Σ W_Y = 1`. It is *not*
  the observer times the illuminant sampled at the band centres.
- **`M_TOTAL`**: gallery XYZ → linear sRGB, `M_rgb·M_adapt` with CAT16 at `D = 1` from the
  gallery white to D65.

### 7.3 Display encoding (`encode_srgb16`)

1. **Black-point compensation** (ICC relative colorimetric with BPC, linear XYZ scaling): the
   medium black (pine soot at load 8 under a full nikawa film) maps to a display black of
   `Y = 0.004`: `xyz' = (xyz - kb)·s + kd` with `kb = (Y_black/Y_white)·white`,
   `kd = 0.004·white` and the uniform scale `s = (1 - 0.004)/(1 - Y_black/Y_white) ≈ 1.01115`.
2. `lin = M_TOTAL·xyz'`; the gallery white maps to (1, 1, 1).
3. **Gamut map**, only if a channel leaves `[-10⁻⁶, 1 + 10⁻⁶]`: to D65 XYZ with the inverse of
   the prototype's IEC sRGB matrix, to OKLab (`lab = M2·cbrt(max(M1·xyz, 0))`), then the chroma
   scale `μ ∈ [0, 1]` is bisected 24 times keeping lightness and hue, and the largest in-gamut
   `lo` is kept and clipped to `[0, 1]`.
4. Clip to `≥ 0`; sRGB transfer (IEC 61966-2-1) `x ≤ 0.0031308 ? 12.92x : 1.055·x^(1/2.4) -
   0.055`, clipped to `[0, 1]`; `round(v·65535)` as a 16-bit code.

### 7.4 Golden values

One sample (mottle `m`, ink gain `g`, film on, BPC, sRGB8 = `round(v16/257)`), computed by the
prototype's float64 path; `optics.rs` reproduces the XYZ to `10⁻¹²` relative and the sRGB8
codes exactly.

| Loads (carbon, cinnabar) | m | g | XYZ | sRGB8 |
|---|---|---|---|---|
| (0, 0) | 1 | 1 | 0.9402834114004027, 0.8278798068379165, 0.25613089936975747 | 240, 234, 221 |
| (1, 0) | 1 | 1 | 0.020521034654061254, 0.01814195130463335, 0.006133908911087946 | 21, 20, 19 |
| (0.06, 1.2) | 1 | 1 | 0.23182060787844666, 0.11761247536190546, 0.005893647895779405 | 177, 34, 16 |
| (0.06, 0.36) | 1 | 1 | 0.1553078585890647, 0.08612930392130692, 0.008443525495676164 | 142, 41, 30 |
| (0.1, 0) | 1 | 1 | 0.055784391761169054, 0.04952957037451743, 0.017225604415632652 | 56, 55, 55 |
| (0.01, 0) | 1 | 1 | 0.2097779838271311, 0.18654086630062808, 0.06522611495495041 | 117, 117, 117 |
| (0.002, 0) | 1 | 1 | 0.4330375076985446, 0.38492773329308283, 0.1328277885679573 | 166, 165, 164 |
| (0.0004, 0) | 1 | 1 | 0.671475105314176, 0.5961081472721036, 0.19987477811714788 | 203, 202, 198 |
| (1, 0) | 1.2 | 0.7 | 0.02188044939077052, 0.019352530570086512, 0.006564053822515822 | 24, 23, 22 |
| (0.01, 0) | 0.4 | 0.5 | 0.2973610590397358, 0.26442950408735566, 0.09234109757079008 | 139, 139, 138 |

In OKLCh: paper (0.939, 0.018, 85°), full carbon (0.2628, 0.0015, 67°), the (0.06, 1.2)
vermilion (0.5067, 0.171, 29.3°).

### 7.5 The kozo sheet (`paper.rs`)

`KozoSheet::generate(width, height, &PaperConfig, rng)` produces per-pixel `mottle` and
`ink_gain` (`f32`). The sheet spans `sheet_width_mm` across the output width (pixel pitch
`p = sheet_width_mm/width`, height `p·height`); coordinates in mm, `x` right, `y` down.

- **Formation `F`** (flocs): `F = Σ_j sqrt(2/N)·cos(kx_j·x + ky_j·y + φ_j)` over
  `N = formation_modes` modes with log-uniform wavelength `λ_j ∈ [floc_min, floc_max]` mm,
  angle `θ_j` and phase `φ_j` uniform in `[0, 2π)`, `kx_j = (2π/λ_j)·cos θ_j/(1 + anisotropy)`,
  `ky_j = (2π/λ_j)·sin θ_j`. Each pixel holds the **exact** box average
  `sqrt(2/N)·sinc(kx·p/2)·sinc(ky·p/2)·cos(kx·x_c + ky·y_c + φ)`, computed with per-column and
  per-row angle-addition tables (`N·(width + height)` sine/cosine pairs), summed per pixel in
  ascending mode order, in passes of 256 modes.
- **Fibres `C`** (covered area fraction): `round(density·area/100 mm²)` fibres over the sheet
  padded by `margin_mm` (more than 2²⁴ is a configuration error). Log-normal length
  (5th/95th percentiles `length_p5_mm`, `length_p95_mm`), uniform width and visibility, a
  fraction `aligned_fraction` aligned with the flow (`x`, normal spread `aligned_spread_deg`),
  the rest uniform; `n = max(1, ⌈length/vertex_spacing⌉)` segments of length `ℓ`, with a
  constant curvature `bend ~ N(0, bend_rad_per_mm)` and a random walk `wander·sqrt(ℓ)·z`
  between them. With `step = min(vertex_spacing, p/3)`, each segment is sampled at `m` points
  along and `across = max(1, ⌈width_max/step⌉)` points across it, and every sample deposits
  `width·visibility·ℓ/(m·across)` into the pixel containing it; `C = min(deposit/p², 1)`.
  Deposits accumulate sequentially in fibre, segment, along, across order.
- **Gains**: `ink_gain = clip(1 - ink_gain_fibre·C, 0.2, 2)`,
  `mottle = clip(1 + mottle_formation·F - mottle_fibre·C, 0.3, 3)`.
- **Randomness** comes from `sim::Sha3RandomByteStream` (`next_f64`), in a fixed order: per mode
  wavelength, angle, phase; per fibre length, width, visibility, alignment, aligned angle, free
  angle, bend, start `x`, start `y`, then its segments' wander normals. Normals are Box–Muller
  pairs through `math::ln`, `sqrt` and `math::sin_cos`, the second deviate cached across fibres.
- **Seed**: the package seed bytes followed by `"\0cosmic-ember/kozo-sheet/v1"`
  (`app::ember_paper_seed`), so every package has its own sheet and the certificate can record
  its digest.

---

## 8. Pipeline, certificate and integration

### 8.1 Planning (`plan_ember`)

`plan_ember(request)` checks, in this order, and derives everything the render needs before the
fluid starts: `EmberConfig::validate`; the output size (non-empty, at most 16,384 per side); the
kozo sheet's fibre count at that size (at most 2²⁴ = 16,777,216, §7.5: it depends on the fibre
density and the output's aspect together, which `validate` cannot see); the
frame schedule (non-empty, strictly increasing, ending on the final knot `N - 1`, `N ≥ 2`); the
orbit's projection (`BodyTrack::new`, §3); the valve time `t_valve = T - valve_lead`, which must
exceed `pre_roll` (`OrbitTooShort`); the fluid grid and the ink node grid. It costs one
projection (about 50 ms for a million knots) and allocates nothing proportional to the output.
`render_ember` plans again itself; `app::preflight_ember_edition` runs the plan right after the
orbit is selected, so a rejected orbit fails in seconds rather than after the main render.

Barring resource exhaustion, a request that plans successfully can fail later only on a
non-finite flow or a failing sink. Planning checks ranges; it does not budget memory or time,
which grow with the output size, the fluid grid and the snapshot cadence.

### 8.2 The frame loop (`render_ember`)

```text
plan (§8.1); solver = WakeSolver::new(grid); look = Look::new; optics = Optics::new
sheet = KozoSheet::generate(width, height, paper, Sha3RandomByteStream(paper_seed))
paper cache: per pixel, the bare-paper XYZ and its 16-bit code
fields = 0; window[0] = snapshot at t = 0, bodies at t = 0; t_prev = 0; previous knot = 0
for each scheduled knot k_f (frame f):
    t_f = t_{k_f};  S = snapshot intervals from the previous knot to k_f (§5.1)
    for s in 1..=S: solver.advance_to(τ_s); snapshot → window[s]; bodies(τ_s)
    if S > 0 and t_f ≥ pre_roll:
        remap(fields → next, window, rules, decay(t_f - t_prev), look); swap   (§5.3)
        any non-finite origin → NonFinite
    if mode == Video or f is the last frame:
        shade every pixel (§7.1) → 16-bit sRGB; non-finite → NonFinite
        rgb48le = little-endian bytes of the samples
        if Video: SHA-256 stream ← rgb48le; sink(frame)
    window[0] ← window[S]; t_prev = t_f
still = the last frame; still digest = SHA-256(rgb48le of the still)
```

- **Modes.** `EmberMode::Video` shades every frame and hands it to the sink in order;
  `EmberMode::StillOnly` shades only the last one and never calls the sink. The fluid and the ink
  run through every interval in both modes, so the still is identical.
- **Before the pre-roll** the fields stay 0 (bare paper); the first remap after it drops every
  contact earlier than `pre_roll` (§5.2 step 5).
- **The snapshot window** keeps its buffers between frames (it grows to the largest `S` seen) and
  reuses the interval's last snapshot as the next interval's first.
- **Statistics** (`EmberStats`, deterministic, in the certificate): fluid steps, smallest and
  largest time step, peak penalised flow speed, snapshots, contacted nodes summed over frames,
  and for the still the fractions of the visible ink nodes (`width·height·q²`, margin excluded)
  carrying any pigment (`still_ink_fraction`) and carrying cinnabar
  (`still_cinnabar_fraction`), and the number of gamut-mapped pixels. Two frame-level
  statistics cover the vermilion over the whole render (§6.3):
  - `frames_with_cinnabar` (`u64`): the number of shaded frames in which any visible node
    carries cinnabar.
  - `peak_frame_cinnabar_fraction` (`f64`): the largest cinnabar node fraction of any shaded
    frame, with the same denominator as `still_cinnabar_fraction`.

  Both are computed in both modes. `StillOnly` shades only the still, so there they count the
  still alone: 0 or 1 frame, and the peak equals `still_cinnabar_fraction`. Both are an integer
  count and an exact maximum, so they are thread-count invariant (§0.1). Wall-clock
  `EmberTimings` (fluid, ink, shade, sink, total) are informational.

### 8.3 Pixel stream

The digested pixels are `rgb48le`: 16-bit sRGB samples, R, G, B per pixel, row-major from the
top-left pixel, little-endian (written explicitly, independent of the host's byte order). The
frame stream digest covers all frames concatenated in schedule order. The PNG, WebP and MP4 files
are encodings of these pixels; their bytes depend on encoder versions and are outside the
contract (the PNG decodes back to the still's `rgb48le`).

### 8.4 The certificate (`certificate.rs`, `metadata/ember.json`)

Layout (`schema_version` 2, `CERTIFICATE_SCHEMA_VERSION`), in file order. Version 1 was used by
test renders only and never published; version 2 adds `stats.frames_with_cinnabar` and
`stats.peak_frame_cinnabar_fraction`, and requires the nullable keys to be present (below). A
reader rejects any other version.

| Field | Contents |
|-------|----------|
| `schema_version`, `edition`, `algorithm`, `contract` | Layout version, `"ember"`, `ALGORITHM_VERSION` (`ember-v1`), and the certified statement. |
| `inputs` | `seed` (hex), `steps`, `dt`, `gravitational_constant`, `integrator`, `bodies` (initial masses, positions and velocities as decimals and as `0x`-prefixed 16-digit `f64` bit patterns), `width`, `height`, `paper_seed_sha256`, `frames` (`count`, `first_step`, `last_step`, `frame_rate`, `sha256`). |
| `config` | The full `EmberConfig`. |
| `derived` | `duration`, `valve_time`, `fluid_grid` `[nx, ny]`, `fluid_dx`, `ink_grid` `[cols, rows]`, and `projection` (`origin`, `extent`, `axes`, `scale`, `variances`). |
| `outputs` | `frames_rgb48le_sha256` (`null` for still-only renders), `frames_emitted`, `still_rgb48le_sha256`, `encoding`. |
| `stats` | `EmberStats` (§8.2). |
| `build` | Informational: `crate_version` (`CARGO_PKG_VERSION`, 1.1.0 for the release that ships the ember edition), `target_arch`, `target_os`, `threads`. |
| `timings_seconds` | Informational: `fluid`, `ink`, `shade`, `sink`, `total`. |

Digests are lowercase hex SHA-256: `inputs.frames.sha256 = schedule_sha256(steps)` hashes the
knot indices as little-endian `u64`s in schedule order; `inputs.paper_seed_sha256 =
paper_seed_sha256(seed bytes ++ "\0cosmic-ember/kozo-sheet/v1")`; the outputs hash `rgb48le`
(§8.3). Everything except `build` and `timings_seconds` is deterministic and must match between
two renders of the same inputs.

**Writing.** `EmberCertificate::new(context, summary)` assembles it; `write_json` writes
pretty-printed JSON with a final newline, flushes and syncs the file, and returns every I/O
error.

**Reading.** `EmberCertificate::read_json`/`from_json` parse it into the same types. The reader
rejects another `schema_version` (`UnsupportedSchema`) or `edition` (`WrongEdition`), missing and
unknown fields at every level, and malformed bit patterns (`Json`, naming the field and its
position). The nullable fields (`outputs.frames_rgb48le_sha256`, `config.look.floor_tau`,
`config.look.ember_tau`) must be present too, as `null` or a value: a certificate without the
frames digest is rejected, not read as still-only. The crate enables `serde_json`'s
`float_roundtrip`, so every decimal reads back exactly and a written certificate reads back
equal to itself, every float bit for bit. `CertificateInputs::bodies()` nevertheless rebuilds
the initial conditions from the bit patterns, never from the decimals next to them: the
decimals are for people, and the bit patterns stay authoritative for readers whose JSON parser
is not exact.

**Versioning.** Bump `ALGORITHM_VERSION` whenever rendered bits change (§0.2), and
`CERTIFICATE_SCHEMA_VERSION` whenever the layout changes.

**What is certified.** The frames are a pure function of `inputs` and `config`: the recorded
bodies, not the seed. Mapping a seed to an orbit runs the main generator's orbit search, whose
scores use platform floating point (`rustfft` with runtime SIMD dispatch, the platform libm);
on an exact near-tie a different machine could in principle select a different orbit. A
cross-machine check therefore compares `inputs.bodies` first, or re-renders from them (§8.6).

### 8.5 Integration in the generator

- **Frame schedule**: `app::ember_frame_schedule(steps)` = `render::main_video_checkpoints(steps)`
  (every 555th step and the final one at the default 1,000,000 steps: 1,802 frames), at
  `DEFAULT_VIDEO_FPS` = 60.
- **Orbit**: `sim::get_positions(bodies, steps)` re-simulates the selected initial conditions raw
  (warm-up of `steps`, then `steps` recorded knots) with `DEFAULT_DT` and `sim::G`.
- **Stage order** (`main.rs`): the preflight (§8.1) right after the orbit selection; the ember
  stage after the main still, videos and spectral outputs (whose buffers are freed first) and
  before the asset manifest. `--no-ember` skips both; `--image-only` renders `StillOnly`.
- **Encoding**: frames stream straight into two encoders, with an explicit BT.709 conversion
  and sRGB tags (`render/video.rs`):
  - web: `web_compatible_srgb` H.264 with `crf = app::EMBER_WEB_CRF` (22), overriding that
    constructor's default CRF 18 (`app::ember_video_options`);
  - archival: `high_quality_srgb` HEVC 4:2:2 10-bit, or `software_fast_srgb` under
    `--fast-encode`.
- **Failure**: when the ember preflight or stage fails, the generator logs the error, removes
  every ember file, writes the rest of the package (metadata included) as a `--no-ember` run
  does, and exits with status **3** ("package complete except the ember edition"). The exit
  statuses are 0 (complete), 3 (complete except the ember edition), 2 (rejected by the argument
  parser: an unknown flag or a malformed value) and 1 (any other failure, including an invalid
  `--seed` or a resolution above 16,384 per side). `run.py` uploads a new mint's status-3
  package with its core files. It retries the ember edition later as a backfill, and gives up
  on a seed after `MAX_BACKFILL_ATTEMPTS` (3) failed ember attempts with the same generator
  binary ([ember-edition.md](ember-edition.md#in-the-sync-loop-runpy)).

### 8.6 Verification (`examples/ember_render.rs`)

`cargo run --release --example ember_render -- verify <package>/metadata/ember.json` reads the
certificate with the typed reader and, before the re-render (about as long as the original
render: 23–35 min at the default size on the measured hosts), fails with a message naming each
field that this build cannot reproduce: `outputs` that disagree about the frames
(`frames_emitted > 0` with a `null` frames digest, which would otherwise verify the still
alone, or a frames digest with `frames_emitted ≠ inputs.frames.count`), `algorithm`
(≠ `ALGORITHM_VERSION`), `inputs.integrator`, `inputs.dt` and `inputs.gravitational_constant`
(compared bit for bit with
`DEFAULT_DT` and `sim::G`), `inputs.paper_seed_sha256` (against the build's paper-seed
derivation), `inputs.frames.sha256` (against the build's schedule), and `config` (it must
round-trip: `serde_json::to_value(&config)` equals the recorded JSON). It then re-renders from
the recorded bodies and compares both digests (exit 0 on a match, 1 on a mismatch, 2 on an
error). The `render` subcommand renders any orbit for look development, with a partial
configuration override.

---

## Appendix A. Default configuration (`EmberConfig::default`)

The production look: the prototype's `vermilion` look ("the ember alone") on the masters'
fluid, plus ember memory. Lengths in world units (the canvas is 2 high), times in fluid time
units, paper sizes in millimetres of the depicted sheet.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `fluid.rows` | 1024 | Fluid grid rows (rounded up to even 5-smooth; §1). |
| `fluid.box_margin` | 0.35 | Periodic box margin beyond the canvas. |
| `fluid.reynolds` | 300 | Reynolds number on the body diameter and the reference speed. |
| `fluid.reference_speed` | 1.0 | Median body speed (§3.6). |
| `fluid.body_radius` | 0.05 | Disc radius `R`. |
| `fluid.cfl` | 0.5 | Courant number. |
| `fluid.max_dt` | 0.002 | Largest time step. |
| `fluid.brinkman_eta_ratio` | 0.01 | Brinkman permeability as a fraction of the step. |
| `fluid.mask_width` | 0.004 | `tanh` edge width `W` of the body mask. |
| `fluid.sponge_rate` | 25 | Sponge damping rate `σ₀`. |
| `fluid.sponge_pad` | 0.06 | Gap between the canvas and the sponge ramp. |
| `fluid.hyperviscosity` | 144 | Hyperviscosity coefficient (per `dx`). |
| `fluid.max_snapshot_interval` | 0.0025 | Largest time between snapshots (§5.1). |
| `fluid.max_snapshot_travel` | 0.5 | Largest body travel between snapshots, in radii (§5.1). |
| `projection.fill` | 0.78 | Fraction of the canvas the orbit fills (§3.5). |
| `contact.soak_depth` | 0.03 | Soak zone beyond the radius (`reach = 0.08`). |
| `contact.vorticity_gate` | 40 | `|ω|` gate `wc` (§5.2 step 2). |
| `contact.pre_roll` | 0.5 | No ink before this time (`t_on`). |
| `contact.valve_lead` | 0.25 | Inking stops this long before the end (`t_valve = T - 0.25`). |
| `look.floor` | 0.0004 | Floor wash strength. |
| `look.fresh_tau` | 0.12 | E-folding age of freshness. |
| `look.hold` | 0.5 | Full-strength age (`g = e^{0.5/0.12} ≈ 64.5`). |
| `look.floor_tau` | `null` | Fade of the presence (and the floor wash); `null` keeps it. |
| `look.ember_tau` | 1.0 | Ember cooling e-folding time (§6.3); `null` is the prototype's rule. |
| `look.cinnabar_strength` | 1.2 | Cinnabar load per unit of meeting strength. |
| `look.carbon_keep` | 0.06 | Carbon kept where cinnabar shows. |
| `look.meeting_threshold` | 0.3 | Both species must exceed this to meet. |
| `paper.sheet_width_mm` | 1490 | Sheet width across the output. |
| `paper.formation_modes` | 1024 | Formation modes `N`. |
| `paper.floc_min_mm`, `paper.floc_max_mm` | 1.5, 8.0 | Floc wavelength range. |
| `paper.anisotropy` | 1.2 | Along-flow floc stretch. |
| `paper.mottle_formation` | 0.2 | Mottle per unit of formation. |
| `paper.mottle_fibre` | 0.6 | Mottle reduction per unit of fibre coverage. |
| `paper.ink_gain_fibre` | 0.5 | Ink-gain reduction per unit of fibre coverage. |
| `paper.fibres.density_per_cm2` | 1.5 | Visible fibres per cm². |
| `paper.fibres.length_p5_mm`, `length_p95_mm` | 5, 15 | Log-normal length percentiles. |
| `paper.fibres.width_min_um`, `width_max_um` | 10, 20 | Fibre width range (µm). |
| `paper.fibres.visibility_min`, `visibility_max` | 0.3, 1.0 | Visibility range. |
| `paper.fibres.aligned_fraction` | 0.35 | Fraction aligned with the flow. |
| `paper.fibres.aligned_spread_deg` | 20 | Angle spread of the aligned fibres. |
| `paper.fibres.bend_rad_per_mm` | 0.04 | Curvature standard deviation. |
| `paper.fibres.wander_rad_per_sqrt_mm` | 0.06 | Direction random walk. |
| `paper.fibres.vertex_spacing_mm` | 0.25 | Polyline vertex spacing. |
| `paper.fibres.margin_mm` | 15 | Seeding margin around the sheet. |
| `raster.supersample` | 2 | Ink nodes per pixel along each axis (`q`). |
| `raster.ink_margin` | 0.15 | Ink carried beyond the canvas edge. |
