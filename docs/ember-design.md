# Ember edition: design

This is the maintainers' reference for `src/ember/`: the rules every line of it follows, the
coordinate conventions, and the exact recipe of each stage, with equations. The product view
(outputs, flags, the look in words, how to verify a package) is in
[ember-edition.md](ember-edition.md); the module docs in `src/ember/` hold the derivations and
the implementation notes.

**Section numbers are stable.** The code cites this document as `docs/ember-design.md §x`
(for example §3.7, §5.2 step 3). Keep §0–§8 and their numbered items when editing, and add new
material as new subsections or appendices.

The edition ports the museum-lab "Bodies Draw" pipeline of the Python prototype (`wake/ns.py`,
`wake2/lagdye.py`, `wake3/{looks,tone,render,spectral,pigments,sheet,trace_torch,trace_triton,
ledger,runstore}.py`, `estuary/source.py`; not part of this repository) to a Rust pipeline whose
output is **bit-identical on every CPU architecture and for every thread count**. The prototype
used PyTorch and Triton GPU code (its tracers `trace_torch.py` and `trace_triton.py` among it);
the port never did. The edition follows the artist's rule of no GPU for anything: every stage
runs on the CPU, down to the video encoders (§8.5). It renders:

- one frame per checkpoint of the main video (`render::main_video_checkpoints`), so frame `i`
  of `ember.mp4` shows the same orbit step as frame `i` of `main.mp4`;
- a 16-bit sRGB still of the final recorded step (knot `steps - 1`), which is the last frame,
  byte for byte.

The look is `tidal_11_exp_film`, the artist's choice from a look-development study: black
pine-soot sumi on a kozo sheet, laid into the water by three tidally stretched bodies. Fresh ink
stays black for a moment, then fades to a pale grey wash on a clock set in film time, the same on
every orbit. There is no red anywhere. The ink is carried by a two-dimensional Navier–Stokes flow
that the three bodies (the selected orbit, projected onto its principal plane) stir as
Brinkman-penalised ellipses of constant area, each stretched by the tidal field of the other two
(§3.10). Beyond the prototype, the edition has the tidal shapes, solid bodies that hold no ink
(§5.3), a **presence-clamped tone law**, under which ink diluted with clear water weakens
linearly (§6.1), and the film-time clock (§6.1). For unmixed water the tone law is the
prototype's reservoir feed. The values agree with it up to the `f32` storage and per-frame
ageing of `E` (§5.3): the prototype evaluates each ink sample's exact age, while the port stores
`E` as `f32` and multiplies it by the frame's fade every frame (a relative rounding of about
2⁻²⁴ per frame).

**Changes from `ember-v1`** (sumi and vermilion on kozo, disc bodies). `ember-v2`:

- draws sumi only: the cinnabar pigment, the ember field `K` and its memory are gone, and with
  them the old §6.2 ("The vermilion accent") and §6.3 ("Ember memory"); the old §6.4 is now §6.2;
- times the tone law in film time: `look.hold_fraction` and `look.fade_fraction` of the orbit's
  duration replace `hold` and `fresh_tau` in fluid units (§6.1);
- stretches the bodies into tidal ellipses (§3.10), which enter the fluid with their deformation
  flow (§4.2) and the tracer as elliptical soak zones (§5.2);
- makes the bodies solid: no ink lies inside them (§5.3);
- resolves the fluid, the ink and the tracing more finely: `fluid.rows` 1024 → 1536,
  `raster.supersample` 2 → 3, `fluid.max_snapshot_travel` 0.5 → 0.28 (§4.2, §5.1);
- writes certificate schema 3 (§8.4), and the generator reports its algorithm with
  `--ember-algorithm` (§8.5).

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
  multiplication, not `powi`. The tidal shapes (§3.10) use only `+ - * /`, `sqrt` and
  comparisons, and their reference is an order statistic.

### 0.2 Golden canaries

Unit tests pin SHA-256 digests of small but complete computations:

- `fft.rs`: the forward FFT of a fixed 48 × 32 field (`GOLDEN_48X32`);
- `fluid.rs`: two snapshots of a fixed run on the 48 × 32 box, one of three circling discs
  (`GOLDEN_SNAPSHOT`) and one of three turning ellipses of aspect 2.25
  (`GOLDEN_STRETCHED_SNAPSHOT`), which exercises the elliptical signed distance and the
  deformation velocity of the penalisation (§4.2);
- `ink.rs`: an ink remap (`remap_output_matches_the_golden_hash`);
- `optics.rs`: the golden table (§7.4) and the bits of the shading and encoding over a grid
  (`GOLDEN_SHA256`);
- `paper.rs`: the paper sheet (`GOLDEN_SHEET_SHA256`);
- `math.rs`: the libm bits (`libm_bits_are_pinned`);
- `tests/ember_determinism.rs`: the end-to-end frame stream and still of a small render of a
  tilted figure-eight with tidal bodies, and two of its statistics, `stats.contact_events` and
  the still's inked node count.

CI runs them on x86_64 Linux, aarch64 Linux and aarch64 macOS, and once more on x86_64 built
for `x86-64-v3` (AVX2/FMA code generation; `ci/README.md`). A digest that changes on one architecture only is a
determinism bug; a digest that changes everywhere is an algorithm change.

Any change that alters rendered bits must bump `certificate::ALGORITHM_VERSION`, re-bless every
affected digest and verify the new values on both architectures:

- **Unit goldens** (`fft.rs`, `fluid.rs`, `ink.rs`, `paper.rs`, `optics.rs`, `math.rs`) have no
  bless switch. Each asserts with `assert_eq!`, so a failing test prints its new value next to
  the recorded one. Run `cargo test --release --lib ember` and copy the printed values into the
  constants.
- **End-to-end goldens**: `EMBER_BLESS=1 cargo test --release --test ember_determinism --
  --include-ignored --nocapture` prints them: the frame-stream and still digests
  (`GOLDEN_FRAMES_SHA256`, `GOLDEN_STILL_SHA256`), the render's `stats.contact_events`
  (`GOLDEN_CONTACT_EVENTS`) and the still's inked node count (`GOLDEN_STILL_INK_NODES` =
  `still_ink_fraction · WIDTH·HEIGHT·q²`, recovered exactly: the render divides one integer count
  by that denominator once). Update all four constants. Every render must print the same values,
  and the ignored `the_golden_render_stretches_its_bodies` must pass: with `tidal.max_aspect = 1`
  the golden orbit draws a different still, so the goldens cover the shape arithmetic (see the
  file's header for the full procedure).

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
   allocate, data is structure-of-arrays. Correctness and determinism come first. Everything
   runs on the CPU: no GPU and no hardware video encoder, anywhere. The cost of the default
   resolution is in §4.2.

---

## 1. Geometry and coordinate conventions

- **World units.** The canvas is `x ∈ [-a, a]`, `y ∈ [-1, 1]` with `a = width / height` (the
  aspect; 3456 × 2234 gives `a ≈ 1.547`); `+y` is up. Lengths are world units, times fluid time
  units (§3.6: the median body speed is the reference speed 1).
- **Output pixels** `(row, col)`, row 0 at the **top**. The pixel size is `s = 2/height` (square
  pixels); pixel centre `x = -a + (col + 0.5)·s`, `y = 1 - (row + 0.5)·s`.
- **Ink nodes** (`ink::NodeGrid`). Supersampling `q = raster.supersample` (default 3), node
  spacing `hn = s/q`, and `M = ⌈raster.ink_margin / hn⌉` margin nodes on every side (default
  margin 0.15). The grid has `rows = q·height + 2M`, `cols = q·width + 2M` nodes; node `(r, c)`
  sits at `x = -a + (c - M + 0.5)·hn`, `y = 1 - (r - M + 0.5)·hn`. Output pixel `(row, col)` is
  the mean of the `q × q` nodes `r = M + q·row + i`, `c = M + q·col + j`, `i, j ∈ 0..q`, in
  row-major order. The continuous node index of a world point is
  `fr = (1 - y)/hn - 0.5 + M`, `fc = (x + a)/hn - 0.5 + M`. Values outside the grid are 0
  (never-inked water). The defaults at 3456 × 2234 give `M = 503` and 11,374 × 7,708 nodes.
- **Fluid grid** (`fluid::FluidGrid`). A doubly periodic box `lx × ly` of `nx × ny` nodes with
  `dx = dy`; node `(i, j)` (row `i`, column `j`) sits at `x = -lx/2 + j·dx`, `y = -ly/2 + i·dx`.
  Row 0 is the **bottom** (`y` minimum), the opposite of the image rows; storage is row-major,
  `index = i·nx + j`. `FluidGrid::for_canvas(a, rows, margin)`:
  `ny = next_smooth_even(rows)`, `dx = 2·(1 + margin)/ny`,
  `nx = next_smooth_even(⌈2·(a + margin)/dx⌉)`, `lx = nx·dx`, `ly = ny·dx`, where
  `next_smooth_even(n)` is the smallest even 5-smooth integer `≥ n` (prime factors 2, 3, 5 only).
  The default (1536 rows, margin 0.35) gives 2160 × 1536 at `a ≈ 1.547` (`dx ≈ 0.001758`).
- **Time.** Fluid time runs over `t ∈ [0, T]` with `T` the orbit's duration (§3.6). Recorded
  knot `k ∈ [0, N)` (`N = steps`; knot `N - 1` is the still) sits at `t_k = T·k/(N - 1)`,
  evaluated as `(T·k)/(N - 1)` except that `t_{N-1} = T` exactly.
- **Film time.** The frame schedule (§8.5) shows the whole orbit, 1,802 frames at 60 fps (about
  30 s) at the default 1,000,000 steps, so film time is proportional to fluid time and a fraction
  of `T` is the same fraction of the film. The look is timed that way (§6.1).
- **Paper.** The sheet is measured in millimetres with `x` right and `y` **down** from the
  top-left corner (§7.5).

---

## 2. Module map

| Module | Responsibility | Key items |
|--------|----------------|-----------|
| `mod.rs` | Overview, module tree, public re-exports. | |
| `error.rs` | Errors. | `EmberError`, `EmberResult` |
| `config.rs` | Every tunable, its defaults (Appendix A) and range validation. | `EmberConfig` and its sections (`TidalConfig` among them), `validate` |
| `math.rs` | The libm facade, portable comparison primitives, the source guard. | `exp`, `ln`, `pow`, `cbrt`, `tanh`, `sin`, `cos`, `sin_cos` |
| `orbit.rs` | Orbit → three moving, tidally stretched bodies: PCA projection, time map and shapes (§3). | `BodyTrack`, `BodyMotion`, `BodyState`, `Shape`, `Projection` |
| `fft.rs` | Deterministic mixed-radix FFT and real 2-D transforms (§4.1). | `Rfft2d`, `Spectrum` |
| `fluid.rs` | The Navier–Stokes wake solver with penalised elliptical bodies (§4.2). | `FluidGrid`, `WakeSolver`, `Snapshot`, `FluidStats` |
| `trace.rs` | Backward characteristics with gated elliptical soak-zone contacts (§5.2). | `Tracer`, `FlowWindow`, `ContactRules`, `SoakFrame`, `Trace` |
| `ink.rs` | Ink node grid, ink fields, solid bodies and the per-frame remap (§5.3). | `NodeGrid`, `InkFields`, `InkDecay`, `remap` |
| `look.rs` | Tone law, timed in film time → the pine-soot load (§6). | `Look` |
| `optics.rs` | 36-band Kubelka–Munk/Saunderson shading of the pine-soot load and the display encoding (§7.1–7.4). | `Optics` |
| `paper.rs` | The kozo sheet: formation and fibres → mottle and ink gain (§7.5). | `KozoSheet`, `PaperSample` |
| `pipeline.rs` | Planning, the frame loop, shading, digests (§8.1–8.3). | `plan_ember`, `render_ember`, `EmberRequest`, `EmberSummary` |
| `certificate.rs` | The determinism certificate, written and read back (§8.4). | `EmberCertificate`, `schedule_sha256`, `ALGORITHM_VERSION` |

**Public API** (`three_body_problem::ember`): `EmberConfig`, `EmberError`, `EmberResult`,
`EmberRequest`, `EmberMode`, `EmberFrame`, `EmberPlan`, `EmberSummary`, `EmberStats`,
`EmberTimings`, `EmberProjection`, `plan_ember`, `render_ember`, `EmberCertificate`,
`CertificateError`, and the `certificate`, `config`, `error` and `pipeline` modules. Everything
else is `pub(crate)`. `fft_bench` is a hidden export for `benches/ember_fft.rs`.

**Around the module**: `app.rs` (`preflight_ember_edition`, `render_ember_edition`,
`ember_masses`, the paper seed and the frame schedule), `main.rs` (stage order, exit status and
`--ember-algorithm`, §8.5), `render/video.rs` (the `*_srgb` encoders, all software),
`examples/ember_render.rs` (verification and look development, §8.6),
`tests/ember_determinism.rs` (the end-to-end golden test) and `benches/ember_fft.rs`.

---

## 3. Orbit → moving bodies (`orbit.rs`)

The input is the raw recorded orbit, `positions[body][knot] ∈ ℝ³`: exactly three bodies with
`N ≥ 2` knots each, all finite, straight from the integrator (not drift- or view-transformed),
and the three initial masses (`EmberRequest::masses`). `BodyTrack::new(positions, masses,
aspect, config)` builds the bodies in the following steps. All sums run in **scan order**
(knot-major, then body) with Neumaier-compensated summation. A non-positive or non-finite
aspect, fill, reference speed or mass is `InvalidConfig`; a degenerate recording is
`DegenerateOrbit`.

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
`vel = (pos_{left+1} - pos_left)·(N - 1)/T` (the outgoing segment's velocity). The body's
`shape` at `t` is §3.10's.

**3.8 Speed look-ahead** (for the solver's CFL). A table
`speed_i = max_b (|vel_b| + deformation_speed_b)(T·i/400000)`, `i ∈ [0, 400000]`: the fastest
material speed of any body, its centre's speed plus its outline's deformation speed (§3.10), with
block maxima of 64 entries. A non-finite entry is `DegenerateOrbit`. `speed_bound(t)` is the
maximum of `speed_i` over `i ∈ [max(i₀ - 2, 0), min(i₀ + 400, 400001))`, where `i₀` is the first
entry with `T·i/400000 ≥ t`. It is the prototype's rule, not a strict bound on the continuous
motion.

**3.9 Interface.**

```rust
pub(crate) struct BodyState { pub position: [f64; 2], pub velocity: [f64; 2], pub shape: Shape }
pub(crate) struct Shape { pub semi: [f64; 2], pub axis: [f64; 2], pub spin: f64, pub strain: f64 }
pub(crate) trait BodyMotion: Sync {          // BodyTrack; tests implement analytic motions
    fn bodies_at(&self, t: f64) -> [BodyState; 3];
    fn speed_bound(&self, t: f64) -> f64;     // look-ahead material speed just after t (§3.8)
}
```

**3.10 Tidal shapes.** Each body is an ellipse of the area of the disc of radius
`R = fluid.body_radius`, stretched along the principal axis of the tidal field of the other two
bodies (the private `Tidal` model): a disc where that field is isotropic, up to
`tidal.max_aspect : 1` at the orbit's closest moments. Its axes turn and stretch with the field,
and its material follows the irrotational flow that carries the outline.

- **Weights.** `w_j = m_j/m̄` with `m̄ = (m_0 + m_1 + m_2)/3`, the initial masses relative to
  their mean.
- **Tidal tensor** on body `i` at the world positions of §3.7 (on the canvas, not in 3-D),
  Plummer-softened with `ε = tidal.softening`: for `j ≠ i` in ascending order,
  `r = x_j - x_i`, `ρ² = |r|² + ε²`, `f = w_j/(ρ²·ρ²·sqrt(ρ²))`, and
  `p += f·(3r_x² - ρ²)`, `q += f·3r_x·r_y`, `s += f·(3r_y² - ρ²)`: the summed tensor
  `Σ_j w_j·(3·r·rᵀ - ρ²·I)/ρ⁵ = [[p, q], [q, s]]`.
- **Anisotropy and axis.** `half = (p - s)/2`, `root = sqrt(half² + q²)`; the anisotropy is
  `Δ = 2·root = λ₁ - λ₂`. The stretch axis is the eigenvector of `λ₁ = (p + s)/2 + root`: the
  better conditioned of `(half + root, q)` and `(q, root - half)` (the first when its squared
  norm is at least the second's), normalised; `(1, 0)` when both vanish (an isotropic tensor,
  `Δ = 0`).
- **Reference.** `Δ_ref` is the order statistic at index `round((3·4001 - 1)·stretch_quantile)`
  of all three bodies' `Δ` at the 4,001 uniform times `T·i/4000` (evaluated like the knot times
  of §1), sorted with `total_cmp`. It is set once `T` is known and recorded as
  `derived.tidal_reference`.
- **Semi-axes.** With `x = Δ/Δ_ref`, the axis ratio is `A = 1 + (max_aspect - 1)·x/(1 + x)`,
  `k = sqrt(A)`, `a = R·k`, `b = R/k` (area kept: `a·b = R²` up to rounding). A body shows half
  its extra stretch at the reference (`A = (1 + max_aspect)/2` at `Δ = Δ_ref`, so 2 at the
  defaults) and approaches `max_aspect` for `Δ ≫ Δ_ref`. When `Δ_ref ≤ 0` or `max_aspect ≤ 1`
  every body is the disc `a = b = R`.
- **Rates.** `t` is clamped to `[0, T]` (a NaN maps to 0) and the shapes are evaluated at `t` and
  at `t + δ`, `δ = 10⁻⁴` (positions clamped by §3.7). The later axis `l` is negated when
  `axis·l < 0` (the eigenvector's sign is arbitrary); then `spin = (axis_x·l_y - axis_y·l_x)/δ`
  and `strain = (a(t + δ) - a(t))/(a(t)·δ)`. At `T`, where the positions stop, both are 0.
- **`Shape`** holds `semi = [a, b]` (`a ≥ b > 0`), `axis = (cos θ, sin θ)` of the long axis,
  `spin = dθ/dt` (counter-clockwise positive) and `strain = d ln a/dt` (`d ln b/dt = -strain`).
  For a world offset `d` from the centre, the body coordinates are
  `x' = cos θ·d_x + sin θ·d_y`, `y' = -sin θ·d_x + cos θ·d_y` (`to_body`), and:
  - `signed_distance(d)`: Taubin's first-order `F/|∇F|` of `F = sqrt((x'/a)² + (y'/b)²) - 1`,
    evaluated as `(q - 1)·q/sqrt((x'/a²)² + (y'/b²)²)` with `q = sqrt((x'/a)² + (y'/b)²)`, and
    `-min(a, b)` where `q < 10⁻¹²` (at the centre the gradient vanishes). Negative inside, exact
    for a disc, accurate to second order near the outline.
  - `contains(d)`: `(x'/a)² + (y'/b)² < 1` (never, for an outline of zero size).
  - `deformation_velocity(d)`: the gradient of `φ = k·x'y' + ½·e·(x'² - y'²)` with
    `k = spin·(a² - b²)/(a² + b²)` and `e = strain`, i.e. `(k·y' + e·x', k·x' - e·y')` in body
    coordinates, rotated back to the world. It is irrotational and divergence-free, moves the
    elliptical outline exactly (the kinematic boundary condition) and keeps its area. For a
    disc `k = e = 0`: a disc's material does not turn with its axes.
  - `deformation_speed() = (|k| + |e|)·a`, the largest deformation speed on or inside the
    outline; `extent() = max(a, b)`.
- **Where the shapes act**: the CFL look-ahead (§3.8), the penalisation (§4.2), the snapshot
  cadence (§5.1), the soak zones (§5.2) and the solid bodies (§5.3).
- **Validation** (`EmberConfig::validate`): `1 ≤ max_aspect ≤ 10`, `0 < stretch_quantile ≤ 1`,
  `softening > 0`.
- **Tests**: a disc's exact signed distance; the sign and first-order accuracy of an ellipse's;
  the deformation flow is divergence-free and irrotational, carries the outline and stays within
  its speed bound; the tidal axis points at a lone companion with `Δ = 3w/r³`; the semi-axes keep
  the area and saturate; the track's reference is the configured quantile and its rates are the
  shapes' own derivatives; `max_aspect = 1` keeps rigid discs.

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
  of a fixed 48 × 32 field. Benchmark: `benches/ember_fft.rs` (round trips at 2160 × 1536, the
  default grid, 1440 × 1024 and 720 × 512).

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
  2. Per body `b` (centre `p`, velocity `V`, outline with signed distance `δ_b` and deformation
     velocity `D_b`, §3.10; edge `W = mask_width`): `z = δ_b(x - p)/W` (Euclidean offset, no
     periodic wrap) and `c_b = ½·(1 - tanh z)` through `math::tanh`, evaluated only at the nodes
     of the body's **node box** where `z < 20`, and exactly 0 elsewhere. The box holds, per
     axis, the nodes `j` (at `(j - n/2)·dx`, `n = nx` or `ny`) with
     `j - n/2 ∈ [⌊(p - reach)/dx⌋ - 1, ⌈(p + reach)/dx⌉ + 1]`, clipped to the grid, where
     `reach = extent + 20·W` (`BodyBox`). Then `χ = max_b c_b` and the mask-weighted material
     velocity `ū = Σ c_b·(V_b + D_b(x - p_b)) / max(Σ c_b, 10⁻¹²)`, summed in body order. For a
     disc `δ = |x - p| - R` and `D = 0`.
  3. Implicit Brinkman with `a = χ / brinkman_eta_ratio`: where `a > 0`,
     `u ← (u + a·ū)/(1 + a)` (and `v` likewise).
  4. Curl and sponge: `ω = IFFT(i·kx·FFT(v) - i·ky·FFT(u))·exp(-σ·h)` with
     `σ(x, y) = sponge_rate·max(ramp(x; a + sponge_pad, lx/2 - 4dx), ramp(y; 1 + sponge_pad,
     ly/2 - 4dx))`, `ramp(c; in, out) = s²(3 - 2s)`, `s = clamp((|c| - in)/(out - in), 0, 1)`
     (precomputed; applied only where `σ > 0`).
  5. `ω̂ = mask·FFT(ω)`.
  6. `u_max = max sqrt(u² + v²)` of the penalised velocity (an exact maximum).

  The neglected mask values: beyond `z = 20` they are below `5·10⁻¹⁸`, and the pinned libm's
  `tanh` already rounds to 1 from about `z = 19.07`, so the `z` cut changes no bit. For a disc
  the box holds every node with `z < 20`. Away from a stretched body's outline Taubin's distance
  falls below the Euclidean one, so the box of a stretched body also drops values (below
  `2·10⁻¹³` at the defaults). The box is part of the definition.
- **Time stepping**, `advance_to(target, bodies)`: until `t = target`, take
  `h_cfl = min(cfl·dx/max(u_max, speed_bound(t), 10⁻⁶), max_dt)` and `rem = target - t`; if
  `rem ≤ h_cfl·(1 + 10⁻⁹)` step `rem` and set `t = target` exactly; else if `rem < 2·h_cfl`
  step `rem/2`; else step `h_cfl`. `speed_bound` is the bodies' look-ahead material speed
  (§3.8). A target before `t` is an error. Initially `ω̂ = 0`, `t = 0`, `u_max = 0`.
- **Snapshots.** `snapshot_into` stores `u`, `v` and the dealiased `ω = IFFT(ω̂)` at the current
  time as **`f32`** (IEEE rounding), with the time.
- **Cost.** 26 real 2-D transforms per step: 20 for the four advection evaluations, 6 for the
  forcing; the 23 whose input or output is dealiased skip the zero columns.
- **Resolution.** The fluid is simulated at the highest resolution that fits the per-package
  budget. The default 1536 rows (2160 × 1536) hold 2.25 times the nodes of 1024 rows
  (1440 × 1024, the masters' grid and `ember-v1`'s default), and the CFL step shrinks with `dx`.
  For seed `0x46205528` (an orbit of 19.124 fluid units) `ember-v2` takes 78,356 steps where
  `ember-v1` took 45,438 (1.72 times), each about 2.2 times as costly on the production host
  (48.9 ms with the solver's 32 threads, measured while the host was loaded, against 22.1 ms
  from `ember-v1`'s certificate): an estimated 3.8 times the fluid time. 2048 rows would cost an
  estimated 2.5 times more again for little visible gain. Measured on an Apple M4 Max (16 cores; other work
  shared the CPU, so the times are upper bounds), the standalone ember render of that seed took
  2 h 05 m (`timings_seconds`: fluid 4,614 s at 58.9 ms per step, ink 2,538 s, shading 283 s,
  sink 61 s) and its whole package 3 h 40 m. On the production host, where the `ember-v1` render
  of the same seed took 23.4 minutes and a sync run of an `ember-v1` package about 70 minutes
  end to end, an `ember-v2` package is estimated at about 2 hours and its ember stage at about
  75–80 minutes.
- **Statistics** (deterministic): steps, smallest and largest `h`, largest `u_max`. A non-finite
  flow, body state or step is `EmberError::NonFinite`.
- **Tests**: a single Fourier mode decays exactly as `e^{-D·t}`; the advection term matches an
  analytic two-mode interaction; a Lamb–Oseen dipole translates and spreads as predicted; the
  CFL and landing rules; a divergence-free snapshot whose curl is the vorticity; a moving disc's
  mirror-antisymmetric wake; the sponge; a disc turning in place leaves still water exactly
  still, while a turning ellipse drives the water inside it with its deformation flow; 1 vs 3
  threads; and the two golden snapshots (§0.2).

---

## 5. Ink (`trace.rs`, `ink.rs`)

### 5.1 Flow window and snapshot cadence

A frame interval `(t_prev, t_frame]` is covered by `S ≥ 1` snapshot intervals, with snapshots at
`τ_0 = t_prev < τ_1 < … < τ_S = t_frame`, `τ_s = t_prev + (t_frame - t_prev)·s/S` (exactly
`t_frame` for `s = S`), all produced by the solver landing exactly on them. Between the frames
at knots `from` and `to`:

```text
S      = max(⌈Δt / max_snapshot_interval⌉, ⌈max_b path_b / (max_snapshot_travel·body_radius)⌉, 1)
path_b = Σ_{k ∈ (from, to]} ( |pos_b(t_k) - pos_b(t_{k-1})|
                              + deformation_speed_b(t_{k-1})·(t_k - t_{k-1}) )
```

with `Δt = t_to - t_from`. `path_b` is the farthest any material of body `b` can travel: the
length of its centre's recorded polyline through the knots `from..=to` (not its displacement,
which misses a body that turns back) plus, per knot interval, its outline's deformation speed
(§3.10) at the interval's start times the interval (a tidally turning ellipse sweeps water even
where its centre rests). `S = 0` when `from = to`. No snapshot interval is longer than
`max_snapshot_interval` (default 0.0025) or lets a body's material move more than
`max_snapshot_travel` radii (default 0.28, i.e. 0.014 world units, under half the shortest
semi-axis `R/√max_aspect ≈ 0.0289`, so the soak test's straight segments (§5.2) keep the
thinnest stretched body resolved in time). `FlowWindow` borrows the `S + 1` snapshots and the
bodies (centres and outlines) at each snapshot time; a single snapshot is an empty window.

### 5.2 Backward trace (`trace_back(window, rules, start) → Trace`)

Trace the parcel at world point `start` at time `τ_S` back to `τ_0`, returning its origin and,
per body, the **latest** contact time in the window (or none). For `s = S … 1`, with
`t1 = τ_s`, `t0 = τ_{s-1}`, `dt = t1 - t0 > 0`, `h = -dt`:

1. **RK4 in time**, velocity bilinear in space (periodic wrap on the fluid grid, `f32` samples,
   `f64` arithmetic) and linear in time:
   `k1 = U1(x)`, `x2 = x + h/2·k1`, `k2 = ½(U1(x2) + U0(x2))`, `x3 = x + h/2·k2`,
   `k3 = ½(U1(x3) + U0(x3))`, `k4 = U0(x + h·k3)`, `xn = x + h/6·(k1 + 2k2 + 2k3 + k4)`.
   Positions are carried unwrapped (world coordinates); only lookups wrap.
2. **Step ends: gate and soak frames.**
   - *Gate* (`wc = vorticity_gate`, default 40; `≤ 0` disables it, `[g0, g1] = [0, 1]`): `o0`
     is the periodic Catmull-Rom sample of `ω(t0)` at `xn`, `o1` the one carried from the
     previous step (initially `ω(t_frame)` at `start`); Catmull-Rom weights at offset
     `t ∈ [0, 1)`: `w0 = -½t³ + t² - ½t`, `w1 = 3/2t³ - 5/2t² + 1`, `w2 = -3/2t³ + 2t² + ½t`,
     `w3 = ½t³ - ½t²`. With `a1 = |o1|`, `a0 = |o0|`:
     `sc = clamp((wc - a1)/(a0 ≠ a1 ? a0 - a1 : 10⁻¹²), 0, 1)`,
     `g0 = a1 > wc ? 0 : (a0 > wc ? sc : 1)`,
     `g1 = a1 > wc ? (a0 > wc ? 1 : sc) : (a0 > wc ? 1 : 0)`.
   - *Soak frames.* Body `i`'s soak zone is its outline (semi-axes `a_i ≥ b_i` along
     `(cos θ_i, sin θ_i)`, §3.10) grown by `soak_depth`, taken as the ellipse of semi-axes
     `a_i + soak_depth`, `b_i + soak_depth` (exact for a disc). Its soak frame (`SoakFrame`)
     rotates a world offset `v` from the body's centre into the body's axes and scales each
     coordinate by the reciprocal of its semi-axis:
     `F(v) = ((cos θ_i·v_x + sin θ_i·v_y)·ia, (-sin θ_i·v_x + cos θ_i·v_y)·ib)` with
     `ia = 1/(a_i + soak_depth)`, `ib = 1/(b_i + soak_depth)` rounded once per body state
     (multiplied, never divided). It maps the zone onto the unit circle. `F_1` is the frame of
     the body at `t1`, `F_0` the one at `t0`.
3. **Zone test** per body `i`, the prototype's `_seg_iv` in soak-frame coordinates with radius 1:
   the parcel moves along the straight segment from `a = F_1(x - c_i(t1))` (`s = 0`) to
   `b = F_0(xn - c_i(t0))` (`s = 1`), with `c_i` the body's centre. With `d = b - a`,
   `A = |d|²`, `B = a·d`, `C = |a|² - 1`, `disc = B² - A·C`:
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
with a literal eager implementation): the soak frames of both ends of every step are computed
once per window; the gate is evaluated only when a zone interval is non-empty; no contact work
once every body has a record, when `t1 - dt > t_valve`, or when `t1 < t_on`; a zone test with
`disc ≤ 0` stops before the square root. `trace_lanes` advances four independent parcels in lock
step (each lane runs exactly the single-parcel operations).

### 5.3 Fields and remap

Each ink node carries four `f32` fields (structure of arrays): the presence `P ∈ [0, 1]` and the
freshness `E_i ∈ [0, 1]` of each body. For a parcel last inked by body `i` at `t*_i` that has not
mixed since, `P = 1` and `E_i = exp(-(t_frame - t*_i)/τ)` with the look's fade time
`τ = fade_fraction·T` (§6.1; 0 if body `i` never inked it). Interpolation mixes all four
linearly, which models dilution at the node scale.

`remap(prev, next, grid, window, rules, decay)`, per node, rows in parallel, with
`InkDecay { frame_time, fade_tau, fresh_fade, floor_fade }` (§8.2):

1. `tr = trace_back(window, rules, node position)` (§5.2).
2. **Solid bodies.** A node strictly inside any body's outline at the frame time
   (`Shape::contains` for the bodies of the window's last snapshot, `τ_S = t_frame`; §3.10) holds
   no water: it stores `P' = E'_i = +0` and is not counted as contacted. A body's interior is
   never ink, so the flow cannot carry ink into a body and leak it back into its wakes, and a
   body always reads as bare paper (without the rule, dense orbits drew their bodies as black
   blobs).
3. Where a field is not overwritten below, sample `prev` at `tr.origin` by **clamped
   Catmull-Rom** on the node grid: 4 × 4 taps, taps outside the grid read 0, and the result is
   clamped to the range of the inner 2 × 2 taps (monotone: no overshoot, no negative ink).
4. Update, with `Δt = t_frame - t_prev`:

   ```text
   P'   = any record ? 1 : P(X)·floor_fade             floor_fade = floor_tau ? exp(-Δt/floor_tau) : 1
   E'_i = record_i ? exp(-(t_frame - t*_i)/τ) : E_i(X)·fresh_fade
                                                       fresh_fade = exp(-Δt/τ)
   ```

   Values below `10⁻¹²` (`FLUSH`) are flushed to exactly `+0`, and the results are stored as
   `f32`.

A node whose traced origin is not finite (a NaN or infinite velocity on its pathline) inherits
clear water and is counted; the pipeline turns any such count into `EmberError::NonFinite`.
Tests cover pure translation and rotation flows, a disc inking exactly its swept band, analytic
moving-disc contact times, the valve and the pre-roll, zero outside the grid, the clamp, the
flush, a node-by-node reference (the eager tracer, the literal sampler and the literal
inside-a-body test), a broken flow, a golden digest and 1 vs 3 threads.

---

## 6. Look (`look.rs`)

The look maps a node's `P` and `E_0..2` to one pigment load, the pine-soot carbon
(`Look::carbon`). `Look::new(config, T)` receives the orbit's duration from the plan. All
arithmetic is exactly rounded `f64`, with one `exp` at construction.

### 6.1 Tone law

The film shows the whole orbit (§1), so the tone law is timed as fractions of the orbit's
duration `T`, which are the same fractions of the film on every orbit, whatever its length in
fluid units. With the hold gain `g` precomputed:

```text
τ      = fade_fraction·T                         e-folding time of the fade (fluid units)
hold   = hold_fraction·T                         full-strength time (fluid units)
g      = exp(hold/τ)
h_i    = min(P, g·E_i)                           presence-clamped hold
carbon = floor·P + (1 - floor)·max_i h_i         the youngest ink of any body wins
```

The certificate records `hold` and `τ` as `derived.hold_time` and `derived.fade_time`.

**Film time.** At the defaults (`hold_fraction = 0.8/30`, `fade_fraction = 0.75/30`) fresh ink
is black for 0.8 s of the 30-second film and then fades with an e-folding time of 0.75 s
(`g = e^{0.8/0.75} ≈ 2.91`). The fresh part `(1 - floor)·e^{-(age - hold)/τ}` falls to the
floor's own strength `τ·ln((1 - floor)/floor) ≈ 7.8·τ` (5.9 s) after the hold, so ink reaches
the pale floor wash about six seconds after the hold (6.7 s after it was laid).

**Unmixed parcels.** For `P = 1`, `g·E_i = exp(-(t_ref - t*_i)/τ)` with `t_ref = t_frame - hold`,
so `carbon = floor + (1 - floor)·exp(-max(t_ref - t*, 0)/τ)` for the youngest uptake
`t* = max_i t*_i`: the prototype's reservoir feed `ExpFeed(t_ref, τ, floor)` at the uptake time.
That is exact as a law. The stored `E_i` carries the `f32` rounding and per-frame ageing of
§5.3, so the values agree with the prototype's to about 2⁻²⁴ per frame, not bit for bit. Ink
younger than `hold` is at full strength (carbon 1); older ink decays towards the `floor` wash.
Bare water (`P = 0`) has carbon exactly 0 and shows the bare paper (§7.1).

**Dilution.** The hold saturates at the presence, not at 1, so the law is linear under dilution
with clear water: a node holding a fraction `w` of an unmixed parcel of age `α` has `P = w`,
`E_i = w·exp(-α/τ)` and hence `h_i = w·min(1, g·exp(-α/τ))`: its carbon is `w` times the unmixed
one. (Saturating at 1 would draw ink diluted down to `1/g ≈ 34 %` at full strength.) Mixing two
*inked* parcels stays concave: the youngest ink wins. With `floor_tau` set, `P` also carries the
floor wash's fade, so the cap fades the whole deposit, fresh ink included, by the same factor.

### 6.2 Validity

`EmberConfig::validate` enforces what the look relies on: `0 ≤ floor < 1`,
`0 < fade_fraction ≤ 1`, `0 ≤ hold_fraction ≤ 1`, a finite positive `floor_tau` when set, and
`hold_fraction/fade_fraction ≤ ln(10⁶) ≈ 13.8` (the ratio is `hold/τ`, the exponent of `g`;
the bound is computed as `ln(10⁻⁶) - ln(10⁻¹²)` through `math`, so every CPU accepts the same
configurations). The last bound comes from the flush: freshness below `10⁻¹²` is stored as 0,
and the hold gain `g` multiplies it, so ink crossing the flush loses `g·10⁻¹²` of strength in
one step; the bound keeps that below `10⁻⁶` (production: `g = e^{0.8/0.75} ≈ 2.91`, a cut of
`2.9·10⁻¹²`).

---

## 7. Optics and paper (`optics.rs`, `paper.rs`)

### 7.1 Reflectance

36 bands, `λ = 380, 390, …, 730` nm. The ink sits in the paper's fibres (Duncan additivity of
absorption `K` and scattering `S`, semi-infinite sheet). For the pine-soot load `L` (the look's
carbon, §6.1) and the pixel's paper mottle `m` and ink gain `g` (§7.5):

```text
k_ink = L·kt_soot + (m - 1)·K_p                     (the mottle term is added before the gain)
s_ink = L·st_soot
K     = K_p + g·k_ink,   S = 1 + g·s_ink            (S_p = 1: the Kubelka–Munk unit)
R∞    = 1/(1 + a + sqrt(a·(a + 2))),   a = K/S
R     = ks + (1 - k1)(1 - k2)·R∞/(1 - k2·R∞)        Saunderson: ks = 0.0164, k1 = 0.04, k2 = 0.6
R    -= (ks - ks_film)·(1 - exp(-max(L, 0)/c_film))      nikawa film: ks_film = 0.005, c_film = 0.35
XYZ   = Σ_b R_b·W_b                                  gallery light, perfect diffuser Y = 1
```

`kt`/`st` are the pine-soot tables times the strength scale 1.2440916174042291, which makes one
load unit of pine soot as strong as one unit of neutral carbon (peak optical density 0.5 over
kozo). The paper's `K_p` is derived at construction from its observed reflectance by the
inverse Saunderson correction and the Kubelka–Munk remission function,
`R' = (R_obs - ks)/((1 - k1)(1 - k2))`, `R_int = R'/(1 + k2·R')`,
`K_p = (1 - R_int)²/(2·R_int)`, and checked against the prototype's `PAPER_K_P`. The Saunderson
body term is evaluated in the algebraically identical form
`(1 - k1)(1 - k2)·S/((1 - k2)·S + K + sqrt(K·(K + 2S)))` (one division, one square root). The
kozo sheet keeps `K ≥ 0` (`m ≥ 0.3`, `g ≤ 1`), so every band lies in `(0, 1)`.

A pixel is the mean XYZ of its `q × q` nodes, in fixed row-major order; a node without pigment
(carbon exactly 0) uses its pixel's cached bare-paper XYZ, and a pixel whose nodes are all bare
is its cached bare-paper code value.

### 7.2 Tables and their provenance

`optics.rs` embeds exact decimal copies of the prototype's tables: `KOZO_R_OBS`,
`SOOT_PINE_K`/`_S` (per unit load, before the strength scale), `W_GALLERY` (36 × 3),
`WHITE_XYZ`, `M_TOTAL` and the medium black `MEDIUM_BLACK_XYZ`; a test checks each, and the
test-only reference tables (`PAPER_K_P`, `M_ADAPT`, `M_RGB`, `CAT16`, `D65_WHITE`), against
checksums computed independently from the source text. They are embedded rather than
recomputed, which avoids porting Mie theory and the spectrum fit:

- **Kozo**: a Jakob–Hanika 2019 sigmoid-polynomial spectrum fitted to sRGB (0.935, 0.915, 0.865).
- **Pine soot**: Mie spheres (`m = 1.95 + 0.79i` in a medium of index 1.5, log-normal
  `d_g = 200` nm, `σ_g = 1.8`), `K = 2µ_a`, `S = 0.75µ_s(1 - g)`, normalised so that load 1
  reaches L* 19.87.
- **Strength scale**: bisected so that pine soot's peak optical density over kozo is 0.5,
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
   `lo` is kept and clipped to `[0, 1]`. Neutral soot on warm kozo stays inside sRGB (a test
   checks loads 0 to 8), so the map is a safeguard; `stats.still_gamut_mapped_pixels` counts its
   use in the still.
4. Clip to `≥ 0`; sRGB transfer (IEC 61966-2-1) `x ≤ 0.0031308 ? 12.92x : 1.055·x^(1/2.4) -
   0.055`, clipped to `[0, 1]`; `round(v·65535)` as a 16-bit code.

### 7.4 Golden values

One sample (carbon load, mottle `m`, ink gain `g`, film on, BPC, sRGB8 = `round(v16/257)`),
computed by the prototype's float64 path; `optics.rs` reproduces the XYZ to `10⁻¹²` relative and
the sRGB8 codes exactly.

| Carbon load | m | g | XYZ | sRGB8 |
|---|---|---|---|---|
| 0 | 1 | 1 | 0.9402834114004027, 0.8278798068379165, 0.25613089936975747 | 240, 234, 221 |
| 1 | 1 | 1 | 0.020521034654061254, 0.01814195130463335, 0.006133908911087946 | 21, 20, 19 |
| 0.1 | 1 | 1 | 0.055784391761169054, 0.04952957037451743, 0.017225604415632652 | 56, 55, 55 |
| 0.01 | 1 | 1 | 0.2097779838271311, 0.18654086630062808, 0.06522611495495041 | 117, 117, 117 |
| 0.002 | 1 | 1 | 0.4330375076985446, 0.38492773329308283, 0.1328277885679573 | 166, 165, 164 |
| 0.0004 | 1 | 1 | 0.671475105314176, 0.5961081472721036, 0.19987477811714788 | 203, 202, 198 |
| 1 | 1.2 | 0.7 | 0.02188044939077052, 0.019352530570086512, 0.006564053822515822 | 24, 23, 22 |
| 0.01 | 0.4 | 0.5 | 0.2973610590397358, 0.26442950408735566, 0.09234109757079008 | 139, 139, 138 |

In OKLCh: paper (0.939, 0.018, 85°), full carbon (0.2628, 0.0015, 67°). The load 0.0004 row is
the floor wash (§6.1).

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
orbit's projection, duration and tidal model (`BodyTrack::new`, §3, which also checks the
masses); the valve time `t_valve = T - valve_lead`, which must exceed `pre_roll`
(`OrbitTooShort`); the fluid grid and the ink node grid. It costs one projection and the orbit's
tables (the median speed over 3 × 200,001 samples, the tidal reference over 3 × 4,001 and the
shape-aware speed table of 400,001 entries, each evaluating every body's shape at `t` and
`t + δ`: fixed counts, whatever the number of knots) and allocates nothing proportional to the
output. `render_ember` plans again itself; `app::preflight_ember_edition` runs the plan right
after the orbit is selected, so a rejected orbit fails in seconds rather than after the main
render.

Barring resource exhaustion, a request that plans successfully can fail later only on a
non-finite flow or a failing sink. Planning checks ranges; it does not budget memory or time,
which grow with the output size, the fluid grid and the snapshot cadence. At the defaults a
standalone render peaked at 4.9 GB resident on an Apple M4 Max, far below the peak of a whole
package of the same seed there (85 GB resident), which belongs to the main render.

### 8.2 The frame loop (`render_ember`)

```text
plan (§8.1); solver = WakeSolver::new(grid); look = Look::new(look config, T); optics = Optics::new
sheet = KozoSheet::generate(width, height, paper, Sha3RandomByteStream(paper_seed))
paper cache: per pixel, the bare-paper XYZ and its 16-bit code
fields = 0; window[0] = snapshot at t = 0, bodies at t = 0; t_prev = 0; previous knot = 0
for each scheduled knot k_f (frame f):
    t_f = t_{k_f};  S = snapshot intervals from the previous knot to k_f (§5.1)
    for s in 1..=S: solver.advance_to(τ_s); snapshot → window[s]; bodies(τ_s)
    if S > 0 and t_f ≥ pre_roll:
        floor_fade = floor_tau ? exp(-(t_f - t_prev)/floor_tau) : 1
        decay = InkDecay(t_f, τ, exp(-(t_f - t_prev)/τ), floor_fade)
        remap(fields → next, window, rules, decay); swap   (§5.3)
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
  reuses the interval's last snapshot, and its bodies, as the next interval's first.
- **Thread pools.** When the caller's pool has more than `FLUID_MAX_THREADS` (32) threads, the
  solver runs in its own pool of 32 threads (its transforms stop scaling); the remap and the
  shading use the caller's pool. Results never depend on either.
- **Statistics** (`EmberStats`, deterministic, in the certificate): `fluid_steps`, `min_dt`,
  `max_dt`, `max_flow_speed` (the peak penalised flow speed), `snapshots`, `contact_events` (the
  nodes that recorded a contact, summed over the remapped frames; nodes inside a body are not
  counted), `still_ink_fraction` (the fraction of the still's visible ink nodes,
  `width·height·q²`, margin excluded, with a non-zero carbon load) and
  `still_gamut_mapped_pixels`. The counts are integers summed exactly and the fraction is one
  count divided once, so all are thread-count invariant (§0.1). Every interval is remapped in
  both modes, so `contact_events` is the same in `Video` and `StillOnly`. Wall-clock
  `EmberTimings` (fluid, ink, shade, sink, total) are informational.
- **Summary.** `EmberSummary` also carries `duration`, `valve_time`, `hold_time` (`hold`),
  `fade_time` (`τ`), `tidal_reference` (`Δ_ref`), the grids and the projection: the
  certificate's `derived` section (§8.4).

### 8.3 Pixel stream

The digested pixels are `rgb48le`: 16-bit sRGB samples, R, G, B per pixel, row-major from the
top-left pixel, little-endian (written explicitly, independent of the host's byte order). The
frame stream digest covers all frames concatenated in schedule order. The PNG, WebP and MP4 files
are encodings of these pixels; their bytes depend on encoder versions and are outside the
contract (the PNG decodes back to the still's `rgb48le`).

### 8.4 The certificate (`certificate.rs`, `metadata/ember.json`)

Layout (`schema_version` 3, `CERTIFICATE_SCHEMA_VERSION`), in file order. Version 1 was only
written by test renders; version 2, published with `ember-v1`, added
`stats.frames_with_cinnabar` and `stats.peak_frame_cinnabar_fraction` and required the nullable
keys to be present. Version 3
is the layout of `ember-v2`: the cinnabar statistics and the look's vermilion settings
(`fresh_tau`, `hold`, `ember_tau`, `cinnabar_strength`, `carbon_keep`, `meeting_threshold`) are
gone; it adds the `tidal` configuration, the film-time look settings (`fade_fraction`,
`hold_fraction`) and `derived.hold_time`, `derived.fade_time` and `derived.tidal_reference`. A
reader rejects any other version, so an `ember-v1` certificate reads as `UnsupportedSchema`.

| Field | Contents |
|-------|----------|
| `schema_version`, `edition`, `algorithm`, `contract` | Layout version (3), `"ember"`, `ALGORITHM_VERSION` (`ember-v2`), and the certified statement. |
| `inputs` | `seed` (hex), `steps`, `dt`, `gravitational_constant`, `integrator`, `bodies` (initial masses, positions and velocities as decimals and as `0x`-prefixed 16-digit `f64` bit patterns), `width`, `height`, `paper_seed_sha256`, `frames` (`count`, `first_step`, `last_step`, `frame_rate`, `sha256`). |
| `config` | The full `EmberConfig` (Appendix A), `tidal` included. |
| `derived` | `duration`, `valve_time`, `hold_time`, `fade_time`, `tidal_reference`, `fluid_grid` `[nx, ny]`, `fluid_dx`, `ink_grid` `[cols, rows]`, and `projection` (`origin`, `extent`, `axes`, `scale`, `variances`). |
| `outputs` | `frames_rgb48le_sha256` (`null` for still-only renders), `frames_emitted`, `still_rgb48le_sha256`, `encoding`. |
| `stats` | `EmberStats` (§8.2): `fluid_steps`, `min_dt`, `max_dt`, `max_flow_speed`, `snapshots`, `contact_events`, `still_ink_fraction`, `still_gamut_mapped_pixels`. |
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
unknown fields at every level, malformed bit patterns (`Json`, naming the field and its
position), and outputs that disagree about the frames (`InconsistentOutputs`: frames emitted
without a frames digest, or a frames digest over another number of frames than
`inputs.frames.count`). The nullable fields (`outputs.frames_rgb48le_sha256`,
`config.look.floor_tau`) must be present too, as `null` or a value: a certificate without the
frames digest is rejected, not read as still-only. The crate enables `serde_json`'s
`float_roundtrip`, so every decimal reads back exactly and a written certificate reads back
equal to itself, every float bit for bit. `CertificateInputs::bodies()` nevertheless rebuilds
the initial conditions from the bit patterns, never from the decimals next to them: the
decimals are for people, and the bit patterns stay authoritative for readers whose JSON parser
is not exact.

**Versioning.** Bump `ALGORITHM_VERSION` whenever rendered bits change (§0.2), and
`CERTIFICATE_SCHEMA_VERSION` whenever the layout changes; each keeps its version history in its
doc comment. The sync loop re-renders the published editions of an older algorithm (§8.5).

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
  `app::ember_masses(bodies)` passes the three initial masses (`EmberRequest::masses`, the tidal
  weights of §3.10); anything but three bodies is `DegenerateOrbit`.
- **Stage order** (`main.rs`): the preflight (§8.1) right after the orbit selection; the ember
  stage after the main still, videos and spectral outputs (whose buffers are freed first) and
  before the asset manifest. `--no-ember` skips both; `--image-only` renders `StillOnly`.
- **Encoding**: frames stream straight into two encoders, with an explicit BT.709 conversion
  and sRGB tags (`render/video.rs`):
  - web: `web_compatible_srgb` H.264 with `crf = app::EMBER_WEB_CRF` (22), overriding that
    constructor's default CRF 18 (`app::ember_video_options`);
  - archival: `high_quality_srgb` HEVC 4:2:2 10-bit, or `software_fast_srgb` under
    `--fast-encode`.

  Every encoder is software (`libx264`, `libx265`) on every platform: nothing in the generator
  uses a GPU or a hardware encoder. The main edition's `--fast-encode` (`fast_encode`) is
  `libx264` too; the macOS VideoToolbox encoder it used before `ember-v2` is removed.
- **`--ember-algorithm`** prints `ALGORITHM_VERSION` (`ember-v2`) on its own line and exits 0
  without rendering or writing anything; given with any other argument it is a parse error
  (status 2). `run.py` probes it before planning, reads the `algorithm` of every live
  certificate in one ssh call, and withdraws each edition of an older algorithm (its
  certificate, then its `metadata/assets.json` entries, then its media), which the ember
  backfill then renders again in the current look. Nothing is withdrawn unless both ids can be
  read; `--keep-stale-ember` keeps the live editions
  ([ember-edition.md](ember-edition.md#in-the-sync-loop-runpy)).
- **Failure**: when the ember preflight or stage fails, the generator logs the error, removes
  every ember file, writes the rest of the package (metadata included) as a `--no-ember` run
  does, and exits with status **3** ("package complete except the ember edition"). The exit
  statuses are 0 (complete, or `--ember-algorithm`), 3 (complete except the ember edition), 2
  (rejected by the argument parser: an unknown flag, a malformed value or `--ember-algorithm`
  with another argument) and 1 (any other failure, including an invalid `--seed` or a resolution
  above 16,384 per side). `run.py` uploads a new mint's status-3 package with its core files. It
  retries the ember edition later as a backfill, and gives up on a seed after
  `MAX_BACKFILL_ATTEMPTS` (3) failed ember attempts with the same generator binary
  ([ember-edition.md](ember-edition.md#in-the-sync-loop-runpy)).

### 8.6 Verification (`examples/ember_render.rs`)

`cargo run --release --example ember_render -- verify <package>/metadata/ember.json` reads the
certificate with the typed reader (which rejects an `ember-v1` certificate by its schema
version) and, before the re-render (about as long as the original render: 2 h 05 m at the
default size on an Apple M4 Max, an upper bound measured with other work on the machine), fails
with a message naming each field that this build cannot reproduce: `outputs` that disagree about
the frames (`frames_emitted > 0` with a `null` frames digest, which would otherwise verify the
still alone, or a frames digest with `frames_emitted ≠ inputs.frames.count`), `algorithm`
(≠ `ALGORITHM_VERSION`), `inputs.integrator`, `inputs.dt` and `inputs.gravitational_constant`
(compared bit for bit with `DEFAULT_DT` and `sim::G`), `inputs.paper_seed_sha256` (against the
build's paper-seed derivation), `inputs.frames.sha256` (against the build's schedule), and
`config` (it must round-trip: `serde_json::to_value(&config)` equals the recorded JSON). It then
re-renders from the recorded bodies, their masses included (`app::ember_masses`), and compares
both digests (exit 0 on a match, 1 on a mismatch, 2 on an error). The `render` subcommand
renders any orbit for look development, with a partial configuration override (for example
`--config '{"tidal": {"max_aspect": 2.0}}'`).

At full scale, the package of seed `0x46205528` rendered on the M4 Max carries the still and
frame-stream digests of a standalone render of the same orbit on that machine, bit for bit: the
production path reproduces the edition exactly. Across architectures, the golden render and the
unit goldens (§0.2) are bit-identical on aarch64 macOS, x86_64 Linux and `x86-64-v3`, and at
full scale the production host (x86_64, native build with AVX2 and FMA) re-rendered the same
edition to the M4 Max's still and frame-stream digests bit for bit, in 78 minutes.

---

## Appendix A. Default configuration (`EmberConfig::default`)

The production look, `tidal_11_exp_film`: sumi laid by tidally stretched bodies, black for
0.8 s and fading to grey over about six seconds of the film, on the masters' fluid resolved 1.5
times more finely. Lengths in world units (the canvas is 2 high), times in fluid time units
except the look's, which are fractions of the orbit's duration `T` (and so of the film), paper
sizes in millimetres of the depicted sheet.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `fluid.rows` | 1536 | Fluid grid rows (rounded up to even 5-smooth; 2160 × 1536 at 3456 × 2234; §1, §4.2). |
| `fluid.box_margin` | 0.35 | Periodic box margin beyond the canvas. |
| `fluid.reynolds` | 300 | Reynolds number on the body diameter and the reference speed. |
| `fluid.reference_speed` | 1.0 | Median body speed (§3.6). |
| `fluid.body_radius` | 0.05 | Radius `R` of the disc of equal area to every body (§3.10). |
| `fluid.cfl` | 0.5 | Courant number. |
| `fluid.max_dt` | 0.002 | Largest time step. |
| `fluid.brinkman_eta_ratio` | 0.01 | Brinkman permeability as a fraction of the step. |
| `fluid.mask_width` | 0.004 | `tanh` edge width `W` of the body mask. |
| `fluid.sponge_rate` | 25 | Sponge damping rate `σ₀`. |
| `fluid.sponge_pad` | 0.06 | Gap between the canvas and the sponge ramp. |
| `fluid.hyperviscosity` | 144 | Hyperviscosity coefficient (per `dx`). |
| `fluid.max_snapshot_interval` | 0.0025 | Largest time between snapshots (§5.1). |
| `fluid.max_snapshot_travel` | 0.28 | Largest body travel between snapshots, in radii: under half the shortest semi-axis `R/√3` (§5.1). |
| `projection.fill` | 0.78 | Fraction of the canvas the orbit fills (§3.5). |
| `contact.soak_depth` | 0.03 | Soak zone depth beyond the outline (semi-axes `a + 0.03`, `b + 0.03`; 0.08 for a disc; §5.2 step 2). |
| `contact.vorticity_gate` | 40 | `|ω|` gate `wc` (§5.2 step 2). |
| `contact.pre_roll` | 0.5 | No ink before this time (`t_on`). |
| `contact.valve_lead` | 0.25 | Inking stops this long before the end (`t_valve = T - 0.25`). |
| `tidal.max_aspect` | 3.0 | Largest axis ratio `a/b`; 1 keeps discs (§3.10). |
| `tidal.stretch_quantile` | 0.95 | Quantile of the orbit's tidal anisotropy at which a body shows half its extra stretch (aspect 2): nearly round most of the time, stretched at the closest 5 % of moments. |
| `tidal.softening` | 0.1 | Plummer softening length `ε` of the tidal field. |
| `look.floor` | 0.0004 | Floor wash strength. |
| `look.fade_fraction` | 0.75/30 = 0.025 | E-folding time `τ` of the fade as a fraction of `T`: 0.75 s of the 30-second film (§6.1). |
| `look.hold_fraction` | 0.8/30 ≈ 0.02667 | Full-strength time as a fraction of `T`: 0.8 s (`g = e^{0.8/0.75} ≈ 2.91`). |
| `look.floor_tau` | `null` | Fade of the presence (and the floor wash), in fluid units; `null` keeps it. |
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
| `raster.supersample` | 3 | Ink nodes per pixel along each axis (`q`; 9 per pixel). |
| `raster.ink_margin` | 0.15 | Ink carried beyond the canvas edge. |
