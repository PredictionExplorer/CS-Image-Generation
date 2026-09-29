# The ember edition

The ember edition draws the selected orbit a second time: not as light, but as the ink trail of the
water the three bodies stir. The bodies move as small discs through a two-dimensional
Navier–Stokes fluid. The discs do not rotate, but the water sweeping past them spins in their
boundary layers, and water that passes a body there picks up black pine-soot sumi. The flow
carries that ink, stretches it into filaments and lets it fade with age. Where the fresh waters
of **two** bodies meet, the ink turns vermilion (cinnabar), and the vermilion glows on in the
water for a while after the meeting, like an ember.
The result is shaded spectrally as sumi and cinnabar on a mottled, fibrous kozo sheet under a warm
gallery light. The look is the museum-lab prototype's `vermilion` ("the ember alone") plus
**ember memory**, the product's addition that lets the red glow on
(see [the look](#4-the-look-lookrs)).

Every package contains the ember edition next to the main render, unless the generator runs with
`--no-ember`. It is rendered **on the CPU only**, and its frames are **bit-identical on every CPU
architecture** (x86_64 with or without AVX2/FMA, aarch64/NEON) and for every thread count. Each
package carries a certificate, `metadata/ember.json`, with the SHA-256 digests anyone can
reproduce.

The Rust implementation lives in `src/ember/`. The orchestration is `render_ember_edition` in
`src/app.rs`. The maintainers' design reference, with every equation, convention, determinism
rule and default, is [ember-design.md](ember-design.md).

## Outputs

| File | Format | Notes |
|------|--------|-------|
| `images/source/ember.png` | 16-bit RGB PNG, sRGB | The still: the orbit's final step (`steps - 1`), identical to the last video frame. It carries the `sRGB`, `gAMA`, `cHRM` and `cICP` chunks. About 36 MB at 3456 × 2234. |
| `images/web/ember_full.webp` | WebP, full size | Website image. |
| `images/web/ember_preview.webp` | WebP, at most 640 px wide | Card/preview image. |
| `videos/web/ember.mp4` | H.264, 8-bit 4:2:0, CRF 22 | Browser-compatible video (about 84 MB for 30 s at 3456 × 2234: paper grain under a drifting wash is costly to encode; CRF 22 is indistinguishable from CRF 18 at 1:1). |
| `videos/hq/ember.mp4` | HEVC Main 4:2:2 10, CRF 17, preset slower | Archival video, about 284 MB for 30 s at 3456 × 2234. Under `--fast-encode` it is software H.264 10-bit 4:2:0 instead. |
| `metadata/ember.json` | JSON | The determinism certificate (see below). |

The sizes were measured for seed `0x46205528`. Together the ember files add about 0.4 GB to a
default package, while the rest of the package, the 64 spectral bins aside, is about 52 MB.

Both videos have one frame per frame of `main.mp4`, at 60 fps. With the default 1,000,000 steps
that is 1,802 frames: every 555th step, then the final step. Frame `i` of `ember.mp4` shows the
same orbit step as frame `i` of `main.mp4`.

The files are listed in `metadata/assets.json` with the roles `ember_source_master`,
`ember_web_full`, `ember_web_preview`, `ember_web` and `ember_hq`, each with
`"color_space": "srgb"`. These roles are additive, so the manifest stays at `schema_version` 2.
The certificate, `metadata/ember.json`, has no manifest entry. A package is complete only with
all six files. How `run.py` adds the edition to packages that lack it is described in
[In the sync loop](#in-the-sync-loop-runpy).

### Flags

| Flag | Effect on the ember edition |
|------|-----------------------------|
| *(none)* | Still, WebPs, both videos and the certificate. |
| `--image-only` | Still, WebPs and the certificate. The fluid and the ink still run through every frame interval, so the still is identical to the full render's. Only the per-frame shading, the frame stream and the encoders are skipped. |
| `--fast-encode` | The HQ slot uses software `libx264`, never a hardware encoder. The pixels and digests do not change. |
| `--no-ember` | Skips the ember edition. The main outputs are unchanged. |
| `--metadata-only` | Skips all rendering, the ember edition included. |

Every run except `--metadata-only` first removes the ember files an earlier run left in the same
output directory. A package therefore never holds ember files its `metadata/assets.json` does not
list, whatever the flags of this run.

The ember stage runs after the main still, videos and spectral outputs, once their buffers have
been freed, and before the asset manifest is written. Because it runs last, a **preflight** runs
right after the orbit is selected, before the main render: it re-simulates the orbit (a fraction
of a second) and makes the renderer's up-front checks (configuration, output size, the paper's
fibre count at that size, frame schedule, orbit projection, and an orbit long enough in fluid
time for the pre-roll and the valve). An orbit the ember edition would reject is known there, in
seconds, and the stage is skipped instead of failing after the whole main render. Only failures
that appear while simulating (a non-finite flow) or encoding can still occur late.

### When the ember edition fails

A failure of the ember edition never costs the rest of the package. When its preflight or its
stage fails, the generator logs the error, removes every ember file, writes the rest of the
package exactly as a `--no-ember` run does (including `metadata/assets.json` without ember
entries, `metadata/generation.json` and `metadata/nft_traits.json`), and exits with status
**3**: *package complete except the ember edition*.

| Exit status | Meaning |
|-------------|---------|
| 0 | The package is complete (with the ember edition unless `--no-ember`). |
| 1 | Any other failure; the package is incomplete. This includes an invalid `--seed` (for example an odd number of hex digits) and a resolution above 16,384 per side. |
| 2 | Rejected by the argument parser: an unknown flag or a malformed value. |
| 3 | The package is complete except the ember edition, which failed. |

A crash ends the process with a signal rather than a status: a panic aborts (release builds use
`panic = "abort"`), and an out-of-memory kill is a SIGKILL. `run.py` treats every status other
than 0 and 3, and every signal, as a failure. What it does with a status-3 package is described in
[In the sync loop](#in-the-sync-loop-runpy).

Short test runs should pass `--no-ember`; otherwise they end with status 3. The orbit must last
more than `pre_roll + valve_lead = 0.75` fluid time units, and its duration grows with the number
of steps (the median body path, measured in canvas units). Four seeds tried with `--sims 200`
lasted 0.11–0.23 units at 20,000 steps (rejected), 0.97–2.2 at 100,000 and 5.4–7.6 at the default
1,000,000; the prototype's selected seeds last 7.4–14.1. `--steps 1` is always rejected, because
the schedule needs 2 recorded steps.

### In the sync loop (`run.py`)

`run.py` keeps the asset host in sync with the minted tokens (see
[README, Automation](../README.md#automation); the operator's view of what follows is
[The ember backfill](../README.md#the-ember-backfill)). It treats a package as complete only
with all six ember files, and each run sorts the API-listed seeds by what their remote package
lacks:

- **New or incomplete packages** lack a core file: a new mint, or a broken upload. They are
  generated and uploaded in full, first.
- **Backfill seeds** lack only ember files: the package predates the edition, or it was
  uploaded after its ember edition failed. After the new packages, each run generates at most
  `--max-backfill` of them (default 1, env `COSMICSIG_MAX_BACKFILL`; 0 pauses the backfill),
  those with the fewest failed runs of any kind first.

A mint that arrives while a backfill package is rendering waits for it: at most
`--max-backfill` packages, about an hour each on the production host, plus the timer's
5-minute delay.

**Backfill modes.** `--backfill-mode` (env `COSMICSIG_BACKFILL_MODE`) sets what a backfill seed
replaces on the asset host:

| Mode | Upload for a backfill seed |
|------|----------------------------|
| `ember` (default) | Only the ember edition. Published art and metadata are never touched. |
| `full` | The whole regenerated package replaces the remote one (the behaviour before backfill modes existed). |

In `ember` mode the seed is still generated in full locally. Before uploading, `run.py` fetches
the live `metadata/nft_traits.json` and `metadata/assets.json` over ssh, and checks that the
local render shows the **same orbit**. Three fields must be equal: `simulation.masses` (compared as exact JSON numbers),
`generation.borda.selected_index` and `generation.borda.retry_count`. The check is needed
because the orbit search scores candidates with platform floating point (see
[the determinism contract](#the-determinism-contract)), so a rebuilt binary could in principle
select another orbit for the same seed.

- **Same orbit.** `run.py` uploads the five ember media files, then a merged
  `metadata/assets.json`, then `metadata/ember.json` last. In the merged manifest the live
  non-ember entries are kept verbatim, the local `ember_*` entries are added or replaced, and
  `generated_at` comes from the local manifest. The published main art, the spectral files,
  `metadata/generation.json` and `metadata/nft_traits.json` are never touched. The remote
  package looks complete only once the certificate has landed, so an interrupted upload leaves
  the seed a backfill seed and a later run retries it.
- **Different orbit.** `run.py` uploads nothing, logs an ERROR naming the seed and, for each
  field that differs, the live and the regenerated value, and records a failed attempt.

`full` mode replaces minted art with a re-render. A rebuilt binary need not reproduce the live
art bit for bit (for example, x86 production builds use an AVX2 spectral kernel that has changed
since the live packages were made), so use `full` only deliberately. A whole package, a new
mint's or a `full`-mode one, is uploaded in three transfers: the media, then `metadata/`
without the certificate, then `metadata/ember.json`. An interrupted upload again leaves the
package incomplete.

**Exit status 3.** For a new mint, `run.py` deletes any stale ember files from the remote
package directory, then uploads the status-3 package with its core files. The token gets its
artwork and traits at once, and the seed becomes a backfill seed with one failed attempt. A
backfill seed that exits 3 uploads nothing in either mode, because its core package is already
live, and records a failed attempt.

**Retry cap.** Most ember failures (a rejected orbit, a non-finite flow) are deterministic for a
given orbit and binary, and would fail again on every retry with the same binary. After
`MAX_BACKFILL_ATTEMPTS` failed attempts (3; `--max-backfill-attempts`, env
`COSMICSIG_MAX_BACKFILL_ATTEMPTS`), `run.py` therefore leaves a backfill seed out of the plan,
and logs a WARNING naming it on every run: `ember backfill given up after N attempts with this
generator binary`. An encoder or write error, which also ends with status 3, may not be
deterministic (a full disk, an encoder crash), so read a given-up seed's log before assuming it
is broken, and delete its ledger entry (see below) to retry it. These count as failed attempts:

- generator exit status 3;
- in `ember` mode, regenerated metadata that cannot show its orbit or give its ember entries;
- in `ember` mode, an orbit mismatch. It also gives the seed up at once, whatever the cap: the
  same binary would regenerate the same orbit on every retry. The seed is listed under
  `orbit_mismatches` in `backfill_failures.json`, and every run logs `ember backfill given up:
  this generator binary regenerates a different orbit than the live package`;
- a regenerated package whose ember files are incomplete.

Upload and ssh failures, timeouts, a generator killed by a signal (such as SIGTERM), a live
package that cannot be used and a run that is shutting down do not count: they only move the
seed back in the queue. The counts are kept in `backfill_failures.json`, in
`run.py`'s working directory, together with the generator binary's identity (resolved path,
size and modification time). When the binary changes, every count resets, so a rebuilt
generator retries every seed it gave up on. Deleting a seed's entry, or the file, retries
sooner.

**Upgrading.** Merging to `main` deploys automatically once `CI passed` succeeds (see
[docs/deployment.md](deployment.md) and
[README, Upgrading a deployment](../README.md#upgrading-a-deployment)). `run.py` probes the
generator with `--help` on every run, and `--preflight` does too. A binary that does not list
`--no-ember` predates the edition. `run.py` then logs an ERROR, checks packages against the core
files only, so new mints are still uploaded, and pauses the backfill until the generator is
rebuilt.

After the upgrade, each of the 48 existing tokens lacks only the ember edition. The backfill adds
it at one package per timer run: about 66 min each (a 61-minute package, then the upload and
the timer's delay), so about 2 days in all. Each package grows by about 0.4 GB on the asset host
(see [Outputs](#outputs)).

**Versions.** The crate version is 1.1.0. Packages generated by this binary record it as
`pipeline_version` in `metadata/nft_traits.json` and as `build.crate_version` in
`metadata/ember.json`. The default `ember` backfill keeps each existing token's live
`nft_traits.json`, so existing tokens keep `pipeline_version` 1.0.0 once their ember edition is
live. Consumers must find the edition through the manifest's `ember_*` roles, not through
`pipeline_version`
([augur-explorer-integration.md §2.1](augur-explorer-integration.md#21-assetsjson-schema_version-2)).

## How a frame is made

### 1. Orbit to moving discs (`orbit.rs`)

The ember edition re-simulates the selected orbit **raw**, with the same integrator, time step and
number of steps. The main render's trajectory cannot be reused: it has been through the seed's
projection, view rotation and drift. The three recorded paths (3 bodies × N steps) become discs on
the canvas in five steps:

1. Normalise by the bounding box: `q = (p - origin) / extent`.
2. Run a principal component analysis over all 3N points: the cyclic Jacobi eigen-decomposition
   of the 3×3 covariance, in a fixed order. The two leading axes span the canvas. Each axis's sign
   is fixed by a deterministic anchor rule.
3. Scale the projection so the orbit fills 78% of the canvas (`y ∈ [-1, 1]`,
   `x ∈ [-a, a]` with `a = width / height`).
4. Map orbit steps to fluid time so that the **median body speed is exactly 1**. Knot `k` of `N`
   sits at `t_k = T·k/(N-1)`, where `T` is the orbit's duration in fluid units.
5. Interpolate positions linearly between knots.

### 2. The fluid (`fft.rs`, `fluid.rs`)

The fluid is two-dimensional, incompressible and viscous. It is written in vorticity–streamfunction
form on a doubly periodic box somewhat larger than the canvas:

```text
∂ω/∂t + u·∇ω = ν ∇²ω − (hyperviscosity)          ψ̂ = ω̂ / |k|²,   u = ∂ψ/∂y,   v = −∂ψ/∂x
```

- **Scheme.** Pseudo-spectral, with 2/3-rule dealiasing and an `(|k|/k_c)^24` hyperviscous
  filter. Time integration is Lawson integrating-factor RK4 with an adaptive CFL step
  (Courant 0.5, `dt ≤ 2·10⁻³`).
- **Grid.** 1024 rows. At the default aspect the box is 1440 × 1024 nodes.
- **Bodies.** Each body is a Brinkman-penalised disc of radius 0.05 with a smooth `tanh` edge,
  applied implicitly after each step at the body's new position. The Reynolds number is 300,
  based on the disc diameter and the reference speed.
- **Sponge.** A sponge layer outside the canvas absorbs the wakes before they wrap around the
  periodic box.

The FFT is the crate's own mixed-radix (2, 3, 4, 5) implementation, with portable twiddle tables.

### 3. Ink (`trace.rs`, `ink.rs`)

The ink lives on a node grid with 2 × 2 nodes per output pixel, plus a margin of 0.15 world units
around the canvas. Each node stores:

- a presence `P ∈ [0, 1]`, and
- for each body `i`, a freshness `E_i = exp(−(t_frame − t*_i)/τ)`.

Here `t*_i` is the last time the node's water touched body `i`'s soak zone, and `τ = 0.12`. A
fifth field, the ember `K`, holds the vermilion that formed where two bodies' inks met (see
[the look](#4-the-look-lookrs)). All fields mix linearly, so interpolating them models dilution at
the grid scale.

Between two frames the fluid is advanced through a few velocity snapshots. The snapshots are close
enough in time that no body moves more than half a radius between them. Each node then does two
things:

1. **Traces back.** It follows its water backwards to the previous frame along exact
   characteristics: RK4 in time, bilinear velocity in space, linear in time between snapshots.
   Along the way it records, per body, the latest moment the path crossed the body's soak zone
   (radius `R + 0.03`) while the local vorticity exceeded the gate `|ω| > 40`.
2. **Updates its fields.**
   - A contact re-inks the node at full strength.
   - Without one, the node inherits the previous frame's fields at the traced origin (clamped
     Catmull-Rom interpolation), faded by the elapsed time.

Two rules apply to all ink:

- **Pre-roll.** Nothing inks before `t = 0.5`, so the impulsive start settles first. The first
  frames are bare paper.
- **Valve.** The bodies stop inking 0.25 time units before the end. The last frame shows ink
  released into the flow rather than ink attached to the bodies.

### 4. The look (`look.rs`)

The look turns each node's fields into two pigment loads, carbon and cinnabar. It is the
prototype's reservoir feed with a 0.5 hold, with the hold capped by the presence:

```text
h_i  = min(P, E_i · e^{hold/τ})                      hold = 0.5, τ = 0.12
c_i  = floor·P + (1 − floor)·h_i                     floor = 4·10⁻⁴
mono = floor·P + (1 − floor)·max_i h_i
best = max over body pairs of min(c_i, c_j);  best = 0 unless best > 0.3
carbon   = mono · (best > 0 ? 0.06 : 1)
cinnabar = 1.2 · best
```

Ink up to `hold` old is at full strength. Older ink decays towards a faint floor wash. Vermilion
appears where two bodies' fresh ink overlaps. There, most of the carbon gives way to the
cinnabar.

**Dilution.** In unmixed water `P = 1`, and the law is exactly the prototype's. Where inked water
has mixed with clear water, `P` is the inked fraction, and capping the hold at `P` rather than at
1 makes every strength scale with it: water holding 2% of fresh ink is 2% as dark, not full
black, and two bodies' 2% traces cannot meet in full vermilion. (The prototype ages every ink
sample exactly and never dilutes, so it has no such case.)

**Ember memory.** In the prototype the vermilion exists only while both inks are fresh (about
0.6 fluid time units), so a still of the orbit's final step usually has none: of six museum-lab
orbits rendered to step 1,000,000, only one showed any. The product therefore lets the red *glow
on*. Every node also carries an ember `K`, advected and diluted by the flow like the ink:

```text
K        ← max(best, K_prev · e^{−Δt/ember_tau})      ember_tau = 1.0
red      = max(best, K > 0.3 ? K : 0)
carbon   = red > 0 ? 0.06 · max(mono, red) : mono
cinnabar = 1.2 · red
```

An ember stays crisp and saturated while it is hotter than the meeting threshold, shrinks as
mixing dilutes it, and goes out about `ember_tau · ln(K₀ / 0.3)` (≈ 1.2 fluid units) after the
meeting. It keeps the soot it formed with, so it stays the deep vermilion of a fresh meeting, and
it never fades into a pale pink wash: the vermilion stays an accent (under 1% of the sheet in
the museum-lab orbits). Water inside a body travels with it, so a body whose water met another's
glows red for a while, and fresh ink it lays into still-glowing water is vermilion, not black.

**A still may have no vermilion.** Ember memory lengthens the window, but the still shows
vermilion only when two bodies' waters met within about `ember_tau · ln(K₀ / 0.3)` (≈ 1.2 fluid
units) before the final step. Many stills are pure sumi. An example is seed `0x46205528`, whose
waters rarely meet at all: its production still has no cinnabar. The video shows every meeting.
Three statistics in `metadata/ember.json` record how much vermilion a package has:
`stats.still_cinnabar_fraction`, `stats.frames_with_cinnabar` and
`stats.peak_frame_cinnabar_fraction` (see [the certificate](#the-certificate-metadataemberjson)).

With `ember_tau: null` the look follows the prototype's law, and the loads are the prototype's
rule bit for bit for the same ink fields. The pictures are the same law up to `f32` storage and
the per-frame ageing of the freshness `E`. The prototype ages each ink sample exactly, while the
port stores `E` as `f32` and fades it frame by frame. A smooth variant in which cooled embers
faded out gradually was tried and rejected: its pink tails covered half the sheet.

The default configuration is this look: the prototype's vermilion plus ember memory
(`ember_tau = 1.0`). Every parameter and its default is listed in
[ember-design.md, Appendix A](ember-design.md#appendix-a-default-configuration-emberconfigdefault).

### 5. Paper and optics (`paper.rs`, `optics.rs`)

**The sheet.** The kozo sheet is 1,490 mm wide across the output. Its texture has two parts:

- **Formation.** 1,024 random cosine modes (flocs 1.5–8 mm), averaged exactly over each pixel.
- **Fibres.** Visible fibres at about 1.5 per cm², 35% of them aligned with the machine direction.

Together they give a per-pixel absorption mottle and an ink gain. The sheet is seeded from the
package seed bytes followed by `"\0cosmic-ember/kozo-sheet/v1"`, so every seed has its own sheet.

**Shading.** Each node is shaded over 36 wavelength bands (380–730 nm):

- Kubelka–Munk layers of pine-soot and cinnabar ink on the paper. The paper's absorption is
  derived from a smooth kozo reflectance spectrum fitted to the paper colour (sRGB
  0.935, 0.915, 0.865).
- A Saunderson surface correction and a thin glue (nikawa) film.
- Integration under a warm gallery LED (CIE LED-V1), adapted to D65 with CAT16.

**Pixels.** Each pixel is the mean XYZ of its 2 × 2 nodes. It then goes through black-point
compensation, then gamut mapping (only if a colour falls outside sRGB: the OKLab chroma is
reduced by bisection, keeping lightness and hue), and is encoded as 16-bit sRGB. A pixel whose
four nodes are all bare paper uses a cached paper value.

## The determinism contract

The raw frames (`rgb48le`, 16-bit little-endian sRGB samples) and the still are a pure function of
five inputs:

- the selected orbit's initial conditions, the number of steps, the integrator time step and the
  gravitational constant;
- the output size;
- the paper seed;
- the full `EmberConfig`;
- the frame schedule: `main.mp4`'s checkpoints for `steps`, recorded as `inputs.frames`. The ink
  is remapped once per frame interval, so the schedule shapes the still too. Changing it, for
  example through `DEFAULT_TARGET_FRAMES`, changes every digest, the still's included.

Rendering them again on any IEEE-754 CPU, with any number of threads, reproduces every bit. The
contract holds because of these rules:

- **Arithmetic.** Only exactly rounded arithmetic is used: `+ − × ÷`, `sqrt`, comparisons,
  conversions. Rust never fuses `a·b + c` into an FMA, and it never reorders floating-point
  operations.
- **Transcendentals.** `exp`, `tanh`, `sin`, `cos`, `pow`, `cbrt` and `ln` come from the
  pure-Rust [`libm`](https://docs.rs/libm) crate (a port of musl), pinned at `=0.2.16`, never from
  the platform's C library. A unit test in `src/ember/math.rs` rejects any other call.
  Dependabot ignores `libm`; bump it only by hand, together with the golden hashes.
- **No architecture-dependent code paths.** The FFT is the crate's own (no `rustfft`), and there
  are no intrinsics, no `target_feature` dispatch and no `mul_add`.
- **Parallelism over independent outputs only.** Rows, columns, nodes and pixels are split across
  threads, and each output is computed by a fixed sequential recipe. The only reductions across
  threads are integer counts and exact maxima; floating-point sums run in a fixed order within one
  output. Every parallel module has a test that compares 1 and 3 threads bit for bit.
- **The orbit itself.** It comes from the same deterministic integrator. The gravity kernel cubes
  distances by multiplication, not `powi`.

The encoded files (MP4, WebP, even the PNG's compressed bytes) are *derived* artefacts. Their
bytes also depend on encoder and library versions, so they are outside the contract. The PNG is
lossless: decoding it to `rgb48le` reproduces the certified still digest.

The contract is keyed on the recorded initial conditions, not on the seed. Turning a seed into an
orbit runs the main generator's orbit search, whose scores use platform floating point (`rustfft`
with runtime SIMD dispatch, the platform's libm). On an exact near-tie between two candidate
orbits, a different machine could in principle select the other one. A cross-machine check
therefore compares `inputs.bodies` first, or re-renders from them with the `verify` tool below.

### The certificate: `metadata/ember.json`

| Field | Contents |
|-------|----------|
| `schema_version`, `edition`, `algorithm` | Layout version, `"ember"`, and the rendering algorithm version (`ember-v1`, bumped whenever rendered bits change). |
| `contract` | The statement being certified. |
| `inputs.seed`, `inputs.steps`, `inputs.dt`, `inputs.gravitational_constant`, `inputs.integrator` | The simulation inputs. |
| `inputs.bodies` | The selected orbit's initial masses, positions and velocities, as decimals (for people) **and** as exact IEEE-754 bit patterns (`bits`, `0x` followed by 16 hex digits; authoritative). |
| `inputs.width`, `inputs.height`, `inputs.paper_seed_sha256` | Output size and the SHA-256 of the paper seed. |
| `inputs.frames` | Frame count, first and last step, frame rate, and a SHA-256 of the schedule (little-endian `u64` step indices). |
| `config` | The full `EmberConfig`: fluid, projection, contact, look, paper and raster parameters. |
| `derived` | The orbit duration and valve time in fluid units, the fluid and ink grids, and `projection` (origin, extent, principal axes, scale, variances). |
| `outputs.frames_rgb48le_sha256` | SHA-256 of all frames concatenated as `rgb48le`; `null` for `--image-only`. |
| `outputs.still_rgb48le_sha256` | SHA-256 of the still as `rgb48le`. |
| `outputs.frames_emitted`, `outputs.encoding` | Frame count and the colour encoding of the digested pixels. |
| `stats` | Deterministic render statistics: fluid steps, time-step range, peak flow speed, snapshots, contact events, the still's ink and cinnabar coverage (`still_ink_fraction`, `still_cinnabar_fraction`), gamut-mapped pixels, and the vermilion over the whole render: `frames_with_cinnabar` (frames in which any node carries cinnabar) and `peak_frame_cinnabar_fraction` (the largest cinnabar node fraction of any frame). An `--image-only` render counts only the still for these two. |
| `build` | Informational only: crate version (1.1.0 for this release), CPU architecture, OS and thread count of the producing machine. |
| `timings_seconds` | Informational only: wall-clock seconds of the `fluid`, `ink`, `shade` and `sink` stages and the `total`. |

Everything except `build` and `timings_seconds` must match between two renders of the same
inputs. `ember::EmberCertificate::read_json` reads a certificate back into typed Rust values. It
rejects other layout versions, unknown or missing fields (a nullable field such as the frames
digest must be present, as `null` or a value) and malformed bit patterns, and rebuilds the
initial conditions bit for bit from `inputs.bodies[*].bits`. The layout, the digest
definitions and the versioning rules are in
[ember-design.md §8.4](ember-design.md#84-the-certificate-certificaters-metadataemberjson).

### Verifying a package on another machine

`examples/ember_render.rs` re-renders a package from its certificate alone and compares both
digests:

```bash
cargo run --release --example ember_render -- verify output/<name>/metadata/ember.json
```

The re-render takes as long as the original, so the tool first checks that this build can
reproduce the certificate at all. It stops with a message naming each field that rules it out:
outputs that disagree about the frames (frames emitted without a frames digest, or a count other
than `inputs.frames.count`), the rendering algorithm version, the integrator, the time step and
the gravitational constant (bit for bit), the paper-seed digest, the frame-schedule digest, and
whether the recorded configuration round-trips through this build's `EmberConfig`. It then prints `MATCH` or
`MISMATCH` for the still and the frames. The exit status is 0 on a match, 1 on a mismatch and 2
on an error.

### Verifying on two machines

1. Build the same commit on both machines with `cargo build --release --locked`. The native-CPU
   flags in `.cargo/config.toml` do not affect the ember edition.
2. Render the same seed on each. Use `--image-only` for a faster check of the still alone, but use
   the same mode on both machines, because the frames digest exists only for full renders:

   ```bash
   ./target/release/three_body_problem --seed 0x46205528 --output verify
   ```

3. Compare the certificates without the informational sections:

   ```bash
   jq -S 'del(.build, .timings_seconds)' output/verify/metadata/ember.json > ember-$(uname -m).json
   diff ember-x86_64.json ember-arm64.json    # must print nothing
   ```

   A difference in `inputs.bodies` means that the two machines selected different orbits (see
   the contract above). Verify each certificate with the `verify` tool instead.

4. Check that the shipped PNG really is the certified still:

   ```bash
   ffmpeg -v error -i output/verify/images/source/ember.png -f rawvideo -pix_fmt rgb48le - \
     | shasum -a 256          # equals outputs.still_rgb48le_sha256
   ```

5. To check thread-count invariance, render once more with `RAYON_NUM_THREADS=3`. The digests
   must not change.

## Colour handling of the encodes

The frames are sRGB (BT.709 primaries, D65, IEC 61966-2-1 transfer). The ember videos use the
`*_srgb` variants in `src/render/video.rs`:

- **Explicit BT.709 conversion.** The RGB → Y′CbCr conversion is an explicit
  `scale=out_color_matrix=bt709:out_range=tv`. `FFmpeg` 7.1 picks the matrix from
  `-colorspace` on its own, but earlier releases (the production host runs 6.1.1) convert RGB
  with their BT.601 default while still tagging BT.709. A mismatched matrix shifts saturated
  colours by up to 24 levels on an 8-bit scale (measured).
- **Tags on every frame.** `setparams` tags the frames as BT.709 / IEC 61966-2-1 / BT.709 / tv
  range. `FFmpeg` 7.1 ignores the `-color_primaries` and `-color_trc` output options in favour of
  the frames' own tags.
- **Result.** The H.264 and HEVC streams and the MP4 `colr` box all carry the same sRGB tags.

The ignored test `srgb_variants_round_trip_through_bt709` checks this with the local `FFmpeg`:
solid sRGB colours, including the vermilion `(177, 34, 16)`, must round-trip within 4 levels
(8-bit 4:2:0) or 1 level (10-bit) on an 8-bit scale, a BT.601 decode must be more than 10 levels
off (so the check discriminates), and both tag locations must be correct. It passes with
`FFmpeg` 7.1.1 and with the production host's `FFmpeg` 6.1.1. Its worst errors were recorded
only for 7.1.1: 3 levels (8-bit) and 0.25 levels (10-bit), against 22–24 for a BT.601 decode.
Run it on any encoding host:

```bash
cargo test --release --lib srgb_variants_round_trip -- --ignored --nocapture
```

The WebPs are derived from the sRGB PNG with the same recipe as the main images. WebP is sRGB by
definition.

## Runtime

### Measured cost

One default render on the production-class host: `three_body_problem --seed 0x46205528` (100,000
sims, 1,000,000 steps, 3456 × 2234, all outputs). The certificate's `timings_seconds` holds the
fluid, ink, shade, sink and total times; the generator's log line `Ember timings: … stage total …`
adds the orbit, PNG, WebP and certificate.

Host: AMD Ryzen Threadripper PRO 9985WX (64 cores, 128 threads), 503 GB RAM, Ubuntu x86_64.
Orbit: 19.12 fluid time units, 45,438 fluid steps (22 ms each), 9,009 velocity snapshots.

| Stage | Wall time | Share |
|-------|-----------|-------|
| Fluid (`timings_seconds.fluid`) | 1004 s | 71% |
| Ink (`timings_seconds.ink`) | 267 s | 19% |
| Shading (`timings_seconds.shade`) | 36 s | 3% |
| Sink and encoder back-pressure (`timings_seconds.sink`) | 98 s | 7% |
| **Render** (`timings_seconds.total`) | **1406 s (23.4 min)** | 100% |
| **Ember stage** (log: stage total, with orbit, PNG, WebPs, certificate and the encoders' tail) | **1518 s (25.3 min)** | |
| Whole package (main render, spectral gallery and sweep, ember edition) | 61 min | |

The package's peak memory, 117 GB, is the main renderer's histogram pass. The ember render itself
peaks at 2.4 GB (measured while re-rendering this package with `ember_render verify`: fluid state,
the snapshot window, two ink-field buffers of 39 million nodes, the paper and one frame); the
encoders add their own (x265's look-ahead holds several GB at this size).

**Cross-architecture check at production scale.** An Apple M4 Max (aarch64, 16 threads) re-rendered
this package from its certificate with `ember_render verify` in 2058 s: the still and all 1802
frames (83 GB of `rgb48le`) matched the x86_64 render (128 threads) bit for bit.

On an Apple M4 Max (16 cores) the fluid runs at the same 22 ms per step. A half-size render
(1728 × 1117) of the museum-lab seed 21 orbit (14.9 fluid units, 51,555 steps) took 1152 s of
fluid, 181 s of ink and 39 s of shading.

`--image-only` was not timed separately: it skips the per-frame shading and all encoding, so it
costs about the fluid plus the ink.

### How the cost scales

| Stage | Scales with | Notes |
|-------|-------------|-------|
| Orbit re-simulation | steps | About 0.2 s for 1,000,000 steps, done twice: by the preflight (0.21 s including the projection) and by the ember stage. |
| Fluid | orbit duration / time step | 26 real 1440 × 1024 FFTs per step, 22 ms. The grid does not depend on the output size; the step count depends on the orbit's duration and peak speeds (45k–52k steps for 15–19 fluid units). The solver runs in its own pool of at most 32 threads: its transforms are too small to feed more (at 128 threads a step takes 32 ms instead of 22). |
| Ink | frames × ink nodes × snapshots per frame | About 39 million nodes (7584 × 5140, margin included) at 3456 × 2234, four times as many as at 1728 × 1117. Uses every core. |
| Shading | frames × pixels | 36-band Kubelka–Munk for inked nodes; cached paper elsewhere. Uses every core. |
| Encoding | frames × pixels | Runs concurrently with rendering. The frames go to both encoders through pipes, so an encoder slower than the render would show up as sink back-pressure. On the production host both encoders, the archival HEVC (x265 preset slower, 4:2:2 10-bit) included, kept pace with the ~1.3 fps render at 3456 × 2234, and their tail after the last frame added about 2 min. |

`--image-only` skips the per-frame shading and all encoding, but the fluid and ink still run over
the whole orbit.
