# The ember edition

The ember edition draws the selected orbit a second time: not as light, but as the ink trail of the
water the three bodies stir. The bodies move through a two-dimensional Navier–Stokes fluid as small
solid bodies, each stretched by the tidal field of the other two: a disc when it is alone, an
ellipse of up to 3 : 1 at the orbit's closest encounters. Water that brushes past a body where its
boundary layer spins picks up black pine-soot sumi. The flow carries that ink and stretches it
into filaments. The ink stays black for a moment after a body
lays it, then fades to a pale grey wash, on a clock set in film time, so every orbit's film fades
at the same pace. The result is shaded spectrally as sumi on a mottled, fibrous kozo sheet under a
warm gallery light. There is no colour but the ink (see [the look](#4-the-look-lookrs)).

The bodies move exactly as they do in the main artwork. The edition follows the main edition's
view of the orbit (its projection space, its viewing angle, its drift and its framing), so at
every step each body is where `main.mp4` draws the head of that body's trail. Next to the film
that runs in step with `main.mp4`, the edition has a **slow film**: the same film ten times
slower, with every frame of it simulated (see [the slow film](#6-the-slow-film)).

Every package contains the ember edition next to the main render, unless the generator runs with
`--no-ember`. It is rendered **on the CPU only**, like everything in the generator: nothing uses a
GPU or a hardware video encoder. Its frames are **bit-identical on every CPU architecture**
(x86_64 with or without AVX2/FMA, aarch64/NEON) and for every thread count. Each package carries a
certificate, `metadata/ember.json`, with the SHA-256 digests anyone can reproduce.

The Rust implementation lives in `src/ember/`. The orchestration is `render_ember_edition` in
`src/app.rs`. The maintainers' design reference, with every equation, convention, determinism
rule and default, is [ember-design.md](ember-design.md).

## Looks

The certificate's `algorithm` names the look that rendered an edition, and
`three_body_problem --ember-algorithm` prints the look this build renders. The number grows
whenever a change alters the rendered bits.

| `algorithm` | Look | Status |
|-------------|------|--------|
| `ember-v1` | Sumi and vermilion on kozo, disc bodies. The ink faded to grey on a fixed fluid-time clock, and turned vermilion (cinnabar) where the fresh waters of two bodies met. Ink could lie inside the bodies, so dense orbits could draw a body as a black blob. | Retired by the artist. The sync loop withdraws every published `ember-v1` edition and renders it again in the current look ([In the sync loop](#in-the-sync-loop-runpy)). |
| `ember-v2` | Sumi on kozo, tidally stretched solid bodies, the fade timed in film time, on a finer fluid grid and ink raster. Chosen by the artist from a look study (`tidal_11_exp_film`). The orbit was shown in its own principal plane, scaled to fill 78% of the sheet, whatever the main artwork's view. | Superseded. The sync loop renders every published `ember-v2` edition again in the current look ([In the sync loop](#in-the-sync-loop-runpy)). |
| `ember-v3` | The `ember-v2` ink, with the bodies moving as in the main artwork: the edition follows the main edition's projection space, viewing angle, drift and framing. Adds the slow film. Fixes a contact test that could ink water far from any body when a body's stretch axis flipped its sign. | Current. |

The rest of this document describes `ember-v3`.

## Outputs

| File | Format | Notes |
|------|--------|-------|
| `images/source/ember.png` | 16-bit RGB PNG, sRGB | The still: the orbit's final step (`steps - 1`), identical to the last video frame. It carries the `sRGB`, `gAMA`, `cHRM` and `cICP` chunks. About 37 MB at 3456 × 2234. |
| `images/web/ember_full.webp` | WebP, full size | Website image. |
| `images/web/ember_preview.webp` | WebP, at most 640 px wide | Card/preview image. |
| `videos/web/ember.mp4` | H.264, 8-bit 4:2:0, CRF 22 | Browser-compatible video (about 136 MB for 30 s at 3456 × 2234: paper grain under a drifting wash is costly to encode; for the `ember-v1` look, CRF 22 was indistinguishable from CRF 18 at 1:1). |
| `videos/web/ember_slow.mp4` | H.264, 8-bit 4:2:0, CRF 22 | The slow film: the same film ten times slower, with the encoder settings of `videos/web/ember.mp4`. Up to 18,011 frames (300 s) at the default 1,000,000 steps. It has no archival copy. |
| `videos/hq/ember.mp4` | HEVC Main 4:2:2 10, CRF 17, preset slower | Archival video, about 375 MB for 30 s at 3456 × 2234. Under `--fast-encode` it is software H.264 10-bit 4:2:0 instead. |
| `metadata/ember.json` | JSON | The determinism certificate (see below). |

The sizes were measured for seed `0x46205528` with the `ember-v2` look (the package of
[Measured cost](#measured-cost)). Together the ember files made up about 0.55 GB of that
package's 0.81 GB (80 files), while the rest of the package, the 64 spectral bins aside, was
about 52 MB. The slow film did not exist then; its size, and the sizes of the other files with
the `ember-v3` look, are still to be measured.

`ember.mp4` (web and archival) has one frame per frame of `main.mp4`, at 60 fps. With the
default 1,000,000 steps that is 1,802 frames: every 555th step, then the final step. Frame `i`
of `ember.mp4` shows the same orbit step as frame `i` of `main.mp4`, with the bodies at the
heads of that frame's trails. The slow film shows nine more frames between every two of them,
also at 60 fps, from a little before the ink first appears to the end of the orbit.

The files are listed in `metadata/assets.json` with the roles `ember_source_master`,
`ember_web_full`, `ember_web_preview`, `ember_web`, `ember_slow_web` and `ember_hq`, each with
`"color_space": "srgb"`. These roles are additive, so the manifest stays at `schema_version` 2.
The certificate, `metadata/ember.json`, has no manifest entry. A package is complete only with
all seven files. How `run.py` adds the edition to packages that lack it, and replaces the
editions of an older look, is described in [In the sync loop](#in-the-sync-loop-runpy).

### Flags

| Flag | Effect on the ember edition |
|------|-----------------------------|
| *(none)* | Still, WebPs, the three videos and the certificate. |
| `--image-only` | Still, WebPs and the certificate. The fluid and the ink still run through every frame interval, so the still is identical to the full render's. Only the per-frame shading, the slow film's in-between frames, the frame streams and the encoders are skipped. |
| `--fast-encode` | The HQ slot uses software `libx264`, like every `--fast-encode` video: the generator never uses a hardware encoder. The pixels and digests do not change. |
| `--no-ember` | Skips the ember edition. The main outputs are unchanged. |
| `--metadata-only` | Skips all rendering, the ember edition included. The ember preflight still runs (unless `--no-ember`) and logs what the edition would cost for this orbit; if it rejects the orbit, that is logged as a warning and the exit status stays 0. |
| `--drift brownian` | The edition cannot follow a brownian drift, whose path is a random walk that is not recorded: the run completes the package without the ember edition and exits with status 3. `none`, `linear` and `elliptical` (the default) are followed. |
| `--ember-algorithm` | Renders nothing: prints the id of the look this build renders (`ember-v3`) and exits with status 0. It takes no other flag. |

Every run except `--metadata-only` first removes the ember files an earlier run left in the same
output directory. A package therefore never holds ember files its `metadata/assets.json` does not
list, whatever the flags of this run.

The ember stage runs after the main still, videos and spectral outputs, once their buffers have
been freed, and before the asset manifest is written. Because it runs last, a **preflight** runs
once the orbit is selected and the main edition's view of it is known, before the main render:
it re-simulates the orbit and makes the renderer's up-front checks (configuration, output size,
the paper's fibre count at that size, frame schedule, the view and the track it gives the
bodies on the canvas, the bodies' masses, and an orbit long enough in fluid time for the
pre-roll and the valve). An orbit the ember edition would reject is known there, in seconds,
and the stage is skipped instead of failing after the whole main render. Only failures that
appear while simulating (a non-finite flow) or encoding can still occur late.

The preflight also logs what the edition will cost and how the orbit sits on the sheet, in two
lines: `Ember preflight:` (the orbit's duration in fluid time, the inking window, and the frame
counts of the film and the slow film) and `Ember plan:` (a lower estimate of the fluid steps,
the peak body speed relative to the median, how close a body's centre comes to the sheet's edge,
and the share of the orbit during which two bodies overlap). With `--metadata-only` these lines
are available for any seed without rendering anything.

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
of steps (the median body path, measured in canvas units). With `ember-v2`, four seeds tried
with `--sims 200` lasted 0.11–0.23 units at 20,000 steps (rejected), 0.97–2.2 at 100,000 and
5.4–7.6 at the default 1,000,000; the prototype's selected seeds last 7.4–14.1. With `ember-v3`
the path is measured in the main edition's view, drift included, so a seed's duration differs
from its `ember-v2` one; the same seeds have not been measured again. `--steps 1` is always
rejected, because the schedule needs 2 recorded steps.

### In the sync loop (`run.py`)

`run.py` keeps the asset host in sync with the minted tokens (see
[README, Automation](../README.md#automation); the operator's view of what follows is
[The ember backfill](../README.md#the-ember-backfill)). It treats a package as complete only
with all seven ember files, and each run sorts the API-listed seeds by what their remote package
lacks:

- **New or incomplete packages** lack a core file: a new mint, or a broken upload. They are
  generated and uploaded in full, first.
- **Backfill seeds** lack only ember files. The package predates the edition, it was uploaded
  after its ember edition failed, its stale edition was withdrawn, or it holds an edition of an
  older look, which lacks a file of the current one (an `ember-v2` edition has no slow film) or
  was kept online to be replaced (see *Stale editions* below). After the new packages, each run
  generates at most `--max-backfill` of them (default 1, env `COSMICSIG_MAX_BACKFILL`; 0 pauses
  the backfill), those with the fewest failed runs of any kind first.

A mint that arrives while a backfill package is rendering waits for it: at most
`--max-backfill` packages, each a full render (hours on the production host; see
[Runtime](#runtime)), plus the timer's 5-minute delay. The per-seed timeout is 10 hours
(`--timeout`).

**Stale editions.** The certificate's `algorithm` names the look that rendered an edition, and
`three_body_problem --ember-algorithm` prints the look the generator renders. Before planning,
each run reads every live certificate's `algorithm` with one ssh call (the top-level line
`  "algorithm": "<id>",` of the pretty-printed file, read with `sed`). For each listed seed whose
id is older (a lower `ember-v<N>`), it withdraws the edition by default, in this order:

1. `metadata/ember.json` is deleted;
2. `metadata/assets.json` is replaced by the live manifest without its `ember_*` entries (staged
   and renamed);
3. the six media files are deleted, and with them any other ember file that manifest listed
   (one the current look no longer has).

The package is then a backfill seed, and the same run already plans it. The token has no ember
edition until the backfill renders it again in the current look: the retired look never stays
online. Only a readable, older id counts. An unreadable certificate, a generator without
`--ember-algorithm`, a failed listing, a newer live id (a rolled-back generator) and a seed that
is not listed all withdraw nothing. A package that also lacks a core file is left alone, because
the same run regenerates it in full. A failed listing or withdrawal makes the run exit with
status 1, and a later run retries it; `--dry-run` only logs what it would withdraw.

With `--keep-stale-ember` (env `COSMICSIG_KEEP_STALE_EMBER=yes`) nothing is withdrawn. Each run
still reads the certificates, logs `Kept N stale ember editions online (ember-v2 -> ember-v3):
the ember backfill replaces each in place`, and plans every older edition's package as a backfill
seed. The edition stays online until its turn (`--max-backfill` per run), when the backfill
replaces it in place (see *Backfill modes* below): no token is without an ember edition, and both
looks are online until the pass is over. A re-render that fails, or is given up, leaves the old
edition online. The switch does not keep the old editions for good: with `--max-backfill 0` as
well, every live edition is held exactly as it is, while `--max-backfill 0` alone pauses only the
re-rendering, so stale editions are still withdrawn. The operator's view is in
[deployment.md](deployment.md#when-a-deploy-changes-the-ember-look).

**Backfill modes.** `--backfill-mode` (env `COSMICSIG_BACKFILL_MODE`) sets what a backfill seed
replaces on the asset host:

| Mode | Upload for a backfill seed |
|------|----------------------------|
| `ember` (default) | Only the ember edition. Published art and metadata are never touched. |
| `full` | The whole regenerated package replaces the remote one (the behaviour before backfill modes existed). |

In `ember` mode the seed is still generated in full locally. Before uploading, `run.py` fetches
the live `metadata/nft_traits.json` and `metadata/assets.json` over ssh, and checks that the
local render shows the **same orbit in the same view**. Ten fields must be equal, compared as
exact JSON values (numbers by their exact decimal value): the orbit, `simulation.masses`,
`generation.borda.selected_index` and `generation.borda.retry_count`, and the view,
`generation.structure.stack_label`, `generation.projection`, `generation.symmetry`,
`generation.drift.mode`, `generation.drift.scale`, `generation.drift.arc_fraction` and
`generation.drift.orbit_eccentricity`. The check is needed because the orbit search scores
candidates with platform floating point (see
[the determinism contract](#the-determinism-contract)), so a rebuilt binary could in principle
select another orbit for the same seed, and because the ember bodies follow the main edition's
view, so the edition must be rendered in the view of the published art. The viewing rotation and
the frame themselves are not recorded in the live package, so `run.py` assumes that they match
once the ten fields do; it compares the layer stack as the rotation's recorded proxy. A live
`nft_traits.json` that lacks one of the ten fields cannot be used: nothing is rendered or
uploaded until it is repaired on the asset host.

- **Same orbit and view.** The log shows `same orbit and view as the live package, by every
  recorded field (…); uploading only its ember edition`. `run.py` stages the edition and then
  swaps it in. It uploads the six ember media files, a merged `metadata/assets.json` and
  `metadata/ember.json`, each as `<name>.part` beside its destination; once all have landed,
  one ssh call deletes the live certificate (and any ember file the live manifest listed that
  the current look does not have), then renames the media, the manifest and the certificate,
  last, into place. In the merged manifest the live non-ember entries are kept verbatim, the
  local `ember_*` entries are added or replaced, and `generated_at` comes from the local
  manifest. The published main art, the spectral files, `metadata/generation.json` and
  `metadata/nft_traits.json` are never touched. A live ember edition (one kept by
  `--keep-stale-ember`, or what a cut-off swap or withdrawal left) is replaced in place by the
  swap, and stays online untouched until then: a transfer that fails leaves the package byte for
  byte as it was, and the staged files are deleted. Staging needs room on the asset host for the
  old and the new media at once. A swap that is cut off leaves a package without a certificate,
  whose media are each the old file or the new one, never a truncated one: it is a backfill
  seed, and a later run renders it again and repeats the upload.
- **Different orbit or view.** `run.py` uploads nothing, logs an ERROR (`… shows a DIFFERENT
  ORBIT OR VIEW than the live one (…)`) naming the seed and, for each field that differs or
  that the regenerated file lacks, the live and the regenerated value, ends the seed with
  `IDENTITY MISMATCH  seed=0x…`, and records a failed attempt. For a withdrawn edition, the
  token then stays without an ember edition until someone decides: a rebuilt generator, or
  `full` mode. A kept edition stays online as it is.

`full` mode replaces minted art with a re-render. A rebuilt binary need not reproduce the live
art bit for bit (for example, x86 production builds use an AVX2 spectral kernel that has changed
since the live packages were made), so use `full` only deliberately. A whole package, a new
mint's or a `full`-mode one, is uploaded in three transfers: the media, then `metadata/`
without the certificate, then `metadata/ember.json`. Its manifest and certificate are deleted
before the upload starts, so an interrupted one leaves the package incomplete, and a later run
regenerates and uploads it in full.

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
- in `ember` mode, regenerated metadata that cannot show its orbit and view or give its ember
  entries;
- in `ember` mode, an orbit or view mismatch. It also gives the seed up at once, whatever the
  cap: the same binary would regenerate the same package on every retry. The seed is listed
  under `identity_mismatches` in `backfill_failures.json` (`orbit_mismatches`, the list's name
  before the view was checked, is still read from an older file), and every run logs `ember
  backfill given up: this generator binary regenerates a different orbit or view than the live
  package`;
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
generator on every run, and `--preflight` does too. A binary whose `--help` does not list
`--no-ember` predates the edition: `run.py` then logs an ERROR, checks packages against the core
files only, so new mints are still uploaded, and pauses the backfill until the generator is
rebuilt. A binary without `--ember-algorithm` works, but withdraws no stale edition (a WARNING on
every run).

After a deploy that bumps the algorithm, every listed token's edition is withdrawn by the first
run (or, with `--keep-stale-ember`, kept online until it is replaced) and rendered again at one
package per timer run: a full package, then the upload and the timer's delay. A package with the `ember-v2` look is estimated to take about 2 hours on the
production host, against about 70 minutes with `ember-v1` (see [Runtime](#runtime)). Multiply it
by the number of tokens for the whole pass: for 48 tokens, an estimated 4–4.5 days. An
`ember-v3` package costs more, because of its slow film, and has not been timed yet.

**Versions.** Packages generated by a build record its crate version as `pipeline_version` in
`metadata/nft_traits.json` and as `build.crate_version` in `metadata/ember.json`. The default
`ember` backfill keeps each existing token's live `nft_traits.json`, so an existing token keeps
its original `pipeline_version` once its ember edition is live again, while its certificate
records the build that rendered the edition. The crate version is 1.1.0 for all three looks, so
only the certificate's `algorithm` tells the editions of different looks apart. Consumers must
find the edition through the manifest's `ember_*` roles, not through `pipeline_version`
([augur-explorer-integration.md §2.1](augur-explorer-integration.md#21-assetsjson-schema_version-2)).

## How a frame is made

### 1. Orbit to moving bodies (`view.rs`, `orbit.rs`)

The ember edition re-simulates the selected orbit **raw**, with the same integrator, time step and
number of steps, and then applies the main edition's view to it. It cannot take the main render's
trajectory: that one was transformed with the platform's math library, so its bits differ from
machine to machine (see [the determinism contract](#the-determinism-contract)). Instead, the main
pipeline records the view it resolved, and the edition applies that view again with portable
arithmetic. The three recorded paths (3 bodies × N steps) become bodies on the canvas in seven
steps:

1. **Projection space.** The seed's projection: plain positions, or one of three phase-space
   projections that mix positions with velocities (phase portrait, cross braid, hodograph).
2. **Viewing rotation.** The rotation the main edition chose as the best-composed of its
   candidate viewing angles.
3. **Drift.** The offset the main edition's drift adds to all three bodies at every step: none,
   a constant velocity, or an arc of an ellipse (the default).
4. **Frame.** The main edition's frame: the bounding box of the transformed orbit with its 5%
   margin, widened to the output's aspect ratio, and the scale that seeds with a rotational
   symmetry apply to the primary copy of their strokes. The canvas is `y ∈ [-1, 1]`,
   `x ∈ [-a, a]` with `a = width / height`.
5. Map orbit steps to fluid time so that the **median body speed is exactly 1** on the canvas.
   Knot `k` of `N` sits at `t_k = T·k/(N-1)`, where `T` is the orbit's duration in fluid units.
6. Interpolate positions linearly between knots.
7. Give each body its **tidal shape**, from the canvas positions and the initial masses.

After step 4 each body is where the main edition draws the head of that body's trail, at every
recorded step: the two agree to far below a millionth of a pixel (a test holds them to `10⁻⁹`
pixels for every projection space, drift and symmetry). For a seed with a symmetry, the main
artwork draws several copies of every stroke, and the ember bodies follow the primary copy, the
one that is neither rotated nor mirrored. In the phase-space projections the paths on the sheet
are not paths in physical space; the bodies stir the water along them all the same. The recipe
is in [ember-design.md §3](ember-design.md#3-orbit--moving-bodies-viewrs-orbitrs).

The tidal shape is the one a small fluid body takes in the field of the other two. On the canvas,
body `j` exerts on body `i` the Plummer-softened tidal tensor

```text
Q_ij = w_j·(3·r·rᵀ − ρ²·I)/ρ⁵        r = x_j − x_i,   ρ² = |r|² + ε²,   ε = 0.1,   w_j = m_j / m̄
```

with the masses taken relative to their mean. Body `i` is stretched along the leading axis of
`Q_ij + Q_ik`, by its anisotropy `Δ = λ₁ − λ₂` (the difference of its eigenvalues). With
`x = Δ / Δ_ref`, the axis ratio is

```text
A = 1 + (max_aspect − 1)·x/(1 + x)        max_aspect = 3
a = R·√A,   b = R/√A                      R = 0.05: the ellipse keeps the disc's area
```

`Δ_ref` is the orbit's own 95th-percentile anisotropy (`stretch_quantile`), taken over all three
bodies at 4,001 evenly spaced times, and recorded as `derived.tidal_reference`. A body is a disc
where the field is isotropic (a body far from the other two), 2 : 1 at the reference, and nearly
3 : 1 at the orbit's closest moments, so the bodies are nearly round most of the time. The axes
turn and stretch with the field (the rates come from forward differences over `10⁻⁴` fluid
units). The body's material does not rotate rigidly with its axes: it follows the irrotational,
area-preserving flow that carries the elliptical outline, as a fluid star's tidal bulge does.

### 2. The fluid (`fft.rs`, `fluid.rs`)

The fluid is two-dimensional, incompressible and viscous. It is written in vorticity–streamfunction
form on a doubly periodic box somewhat larger than the canvas:

```text
∂ω/∂t + u·∇ω = ν ∇²ω − (hyperviscosity)          ψ̂ = ω̂ / |k|²,   u = ∂ψ/∂y,   v = −∂ψ/∂x
```

- **Scheme.** Pseudo-spectral, with 2/3-rule dealiasing and an `(|k|/k_c)^24` hyperviscous
  filter. Time integration is Lawson integrating-factor RK4 with an adaptive CFL step
  (Courant 0.5, `dt ≤ 2·10⁻³`). The step also respects the bodies' own speed, deformation
  included, over a short look-ahead.
- **Grid.** 1536 rows, 1.5 times finer than the museum-lab masters' 1024. At the default aspect
  the box is 2160 × 1536 nodes.
- **Bodies.** Each body is Brinkman-penalised: a mask with a smooth `tanh` edge (width 0.004) on
  the signed distance to its elliptical outline, applied implicitly after each step at the body's
  new position and shape. Inside the mask the water is driven to the body's material velocity: the
  centre's velocity plus the deformation flow. The Reynolds number is 300, based on the diameter
  of the equal-area disc and the reference speed.
- **Sponge.** A sponge layer outside the canvas absorbs the wakes before they wrap around the
  periodic box.

The FFT is the crate's own mixed-radix (2, 3, 4, 5) implementation, with portable twiddle tables.

### 3. Ink (`trace.rs`, `ink.rs`)

The ink lives on a node grid with 3 × 3 nodes per output pixel, plus a margin of 0.15 world units
around the canvas. Each node stores:

- a presence `P ∈ [0, 1]`, and
- for each body `i`, a freshness `E_i = exp(−(t_frame − t*_i)/τ)`.

Here `t*_i` is the last time the node's water touched body `i`'s soak zone, and `τ` is the fade
time of [the look](#4-the-look-lookrs). Both fields mix linearly, so interpolating them models
dilution at the grid scale.

Between two frames the fluid is advanced through a few velocity snapshots. The snapshots are at
most `2.5·10⁻³` fluid units apart, and close enough that no body's material moves more than 0.28
body radii between them, its outline's turning and stretching included: under half the shortest
semi-axis of a 3 : 1 body. Their number per frame interval is a multiple of ten, so that every
frame of [the slow film](#6-the-slow-film) falls on a snapshot. Each node then does two things:

1. **Traces back.** It follows its water backwards to the previous frame along exact
   characteristics: RK4 in time, bilinear velocity in space, linear in time between snapshots.
   Along the way it records, per body, the latest moment the path crossed the body's soak zone
   (its outline grown by 0.03: the ellipse of semi-axes `a + 0.03`, `b + 0.03`) while the local
   vorticity exceeded the gate `|ω| > 40`.
2. **Updates its fields.**
   - A contact re-inks the node at full strength.
   - Without one, the node inherits the previous frame's fields at the traced origin (clamped
     Catmull-Rom interpolation), faded by the elapsed time.

Three rules apply to all ink:

- **Solid bodies.** A node inside a body's outline at the frame time holds no water and stores no
  ink. The flow cannot carry ink into a body and leak it back into the wakes, and a body always
  reads as bare paper.
- **Pre-roll.** Nothing inks before `t = 0.5`, so the impulsive start settles first. The first
  frames are bare paper.
- **Valve.** The bodies stop inking 0.25 time units before the end. The last frame shows ink
  released into the flow rather than ink attached to the bodies.

### 4. The look (`look.rs`)

The look turns each node's fields into one pigment load, pine-soot carbon. It is the museum-lab
reservoir feed with a hold, with the hold capped by the presence, and timed as fractions of the
orbit's duration `T`:

```text
τ      = fade_fraction · T                    fade_fraction = 0.75/30
hold   = hold_fraction · T                    hold_fraction = 0.8/30
h_i    = min(P, E_i · e^{hold/τ})
carbon = floor·P + (1 − floor)·max_i h_i      floor = 4·10⁻⁴
```

Ink up to `hold` old is at full strength. Older ink decays towards a faint floor wash. Where the
waters of several bodies mix, the youngest ink wins.

**Film time.** The film shows the whole orbit in 30 seconds (1,802 frames at 60 fps at the default
1,000,000 steps), so fractions of `T` are film time. Ink is black for 0.8 s after a body lays it,
then fades with an e-folding time of 0.75 s, and is near the floor wash about six seconds later.
The pace is the same on every orbit, whether it lasts 8 or 19 fluid units. The certificate
records both times in fluid units, as `derived.hold_time` and `derived.fade_time`.

**Dilution.** In unmixed water `P = 1`, and the law is exactly the museum-lab reservoir feed.
Where inked water has mixed with clear water, `P` is the inked fraction, and capping the hold at
`P` rather than at 1 makes every strength scale with it: water holding 2% of fresh ink is 2% as
dark, not full black. (The prototype ages every ink sample exactly and never dilutes, so it has no
such case.)

**The floor.** With `floor_tau: null` (the default) the floor wash stays for the whole film.
A number fades the presence with that e-folding age, and with it (the law caps every strength at
`P`) the whole deposit.

The configuration bounds `hold_fraction / fade_fraction` by `ln(10⁶) ≈ 13.8`, so that flushing
tiny freshness values to zero never cuts visible ink. Every parameter and its default is listed in
[ember-design.md, Appendix A](ember-design.md#appendix-a-default-configuration-emberconfigdefault).

### 5. Paper and optics (`paper.rs`, `optics.rs`)

**The sheet.** The kozo sheet is 1,490 mm wide across the output. Its texture has two parts:

- **Formation.** 1,024 random cosine modes (flocs 1.5–8 mm), averaged exactly over each pixel.
- **Fibres.** Visible fibres at about 1.5 per cm², 35% of them aligned with the machine direction.

Together they give a per-pixel absorption mottle and an ink gain. The sheet is seeded from the
package seed bytes followed by `"\0cosmic-ember/kozo-sheet/v1"`, so every seed has its own sheet.

**Shading.** Each node is shaded over 36 wavelength bands (380–730 nm):

- Kubelka–Munk: the pine-soot load sits in the paper's fibres. The paper's absorption is derived
  from a smooth kozo reflectance spectrum fitted to the paper colour (sRGB 0.935, 0.915, 0.865).
- A Saunderson surface correction and a thin glue (nikawa) film over dense ink.
- Integration under a warm gallery LED (CIE LED-V1), adapted to D65 with CAT16.

**Pixels.** Each pixel is the mean XYZ of its 3 × 3 nodes. It then goes through black-point
compensation, then gamut mapping (only if a colour falls outside sRGB: the OKLab chroma is
reduced by bisection, keeping lightness and hue; neutral soot on warm paper stays inside, so this
is a safeguard), and is encoded as 16-bit sRGB. A pixel whose nine nodes are all bare paper uses a
cached paper value.

### 6. The slow film

`videos/web/ember_slow.mp4` is the same film ten times slower (`EMBER_SLOW_FACTOR` in
`src/app.rs`): between every two frames of `ember.mp4` it shows nine more, at 60 fps.

**Real frames.** No frame is interpolated from its neighbours. Inside every frame interval the
fluid already stops at a number of evenly spaced snapshots that is a multiple of ten, so each
in-between moment is a snapshot of the simulation. An in-between frame is the previous film
frame's ink carried through the flow up to its snapshot, with the bodies at their positions of
that moment and the ink they lay on the way, shaded like any other frame. Every tenth frame of
the slow film is a frame of `ember.mp4`, byte for byte, and its last frame is the still.

**A side product.** The ink of a film frame is never computed from an in-between frame: each
frame of either film is one remap away from a film frame's ink. The in-between frames therefore
blur nothing and the film frames do not stand out among them. The still and `ember.mp4` are the
same whether or not the slow film is rendered; `--image-only` renders the same still without
either film.

**Start and length.** The first frames of the film are bare paper, because nothing inks during
the pre-roll. The slow film skips most of them: it starts six film frames (one second of slow
film) before the frame interval in which the bodies start inking, and runs to the end of the
orbit. From there it has ten frames per film frame, plus the last one: at most 18,011 frames
(300 s) at the default 1,000,000 steps, less ten for every film frame it skips. The certificate
records the film frame it starts at as `derived.slow_first_frame`.

**The look, ten times slower.** The film times of [the look](#4-the-look-lookrs) stretch with
the film: in the slow film the ink stays black for 8 s and fades with an e-folding time of
7.5 s.

The slow film has a web encode only, with the settings of `videos/web/ember.mp4`. Its raw frames
have their own digest in the certificate, so an archival encode can be made from a re-render.
The details are in [ember-design.md §8.3](ember-design.md#83-the-slow-film).

## The determinism contract

The raw frames of both films (`rgb48le`, 16-bit little-endian sRGB samples) and the still are a
pure function of six inputs:

- the selected orbit's initial conditions (masses, positions and velocities), the number of steps,
  the integrator time step and the gravitational constant;
- the main edition's view of that orbit (`inputs.view`): its projection space, viewing rotation,
  drift and frame, recorded as exact bit patterns;
- the output size;
- the paper seed;
- the full `EmberConfig`;
- the frame schedule: `main.mp4`'s checkpoints for `steps`, and the slow factor, recorded as
  `inputs.frames`. The ink is remapped once per frame interval, through a number of snapshots
  that is a multiple of the slow factor, so both shape the still too. Changing the schedule, for
  example through `DEFAULT_TARGET_FRAMES`, or the slow factor changes every digest, the still's
  included.

Rendering them again on any IEEE-754 CPU, with any number of threads, reproduces every bit. The
contract holds because of these rules:

- **Arithmetic.** Only exactly rounded arithmetic is used: `+ − × ÷`, `sqrt`, comparisons,
  conversions. Rust never fuses `a·b + c` into an FMA, and it never reorders floating-point
  operations.
- **Transcendentals.** `exp`, `tanh`, `sin`, `cos`, `pow`, `cbrt`, `ln` and the remainder `fmod`
  come from the pure-Rust [`libm`](https://docs.rs/libm) crate (a port of musl), pinned at
  `=0.2.16`, never from the platform's C library. A unit test in `src/ember/math.rs` rejects any
  other call. Dependabot ignores `libm`; bump it only by hand, together with the golden hashes.
- **No architecture-dependent code paths.** The FFT is the crate's own (no `rustfft`), and there
  are no intrinsics, no `target_feature` dispatch and no `mul_add`.
- **Parallelism over independent outputs only.** Rows, columns, nodes and pixels are split across
  threads, and each output is computed by a fixed sequential recipe. The only reductions across
  threads are integer counts and exact maxima; floating-point sums run in a fixed order within one
  output. Every parallel module has a test that compares 1 and 3 threads bit for bit.
- **The orbit itself.** It comes from the same deterministic integrator. The gravity kernel cubes
  distances by multiplication, not `powi`. The tidal shapes use only `+ − × ÷ √` and comparisons,
  in a fixed order, and their reference is an order statistic.
- **The view.** The main edition computes its view with the platform's math library and chooses
  its viewing angle with a platform-dependent score. The ember edition therefore takes no
  position from it. It takes the view's parameters, recorded in the certificate as bit patterns,
  and applies them to the raw orbit with the arithmetic and the functions above.

The encoded files (MP4, WebP, even the PNG's compressed bytes) are *derived* artefacts. Their
bytes also depend on encoder and library versions, so they are outside the contract. The PNG is
lossless: decoding it to `rgb48le` reproduces the certified still digest.

The contract is keyed on the recorded initial conditions and the recorded view, not on the seed.
Turning a seed into an orbit runs the main generator's orbit search, whose scores use platform
floating point (`rustfft` with runtime SIMD dispatch, the platform's libm). On an exact near-tie
between two candidate orbits, a different machine could in principle select the other one. The
view is resolved by the main pipeline with the platform's libm too, so two machines can record
views that differ in their last bits, or in principle choose another viewing angle. A
cross-machine check therefore compares `inputs.bodies` and `inputs.view` first, or re-renders
from them with the `verify` tool below.

### The certificate: `metadata/ember.json`

| Field | Contents |
|-------|----------|
| `schema_version`, `edition`, `algorithm` | Layout version (4), `"ember"`, and the rendering algorithm version (`ember-v3`, bumped whenever rendered bits change; see [Looks](#looks)). |
| `contract` | The statement being certified. |
| `inputs.seed`, `inputs.steps`, `inputs.dt`, `inputs.gravitational_constant`, `inputs.integrator` | The simulation inputs. |
| `inputs.bodies` | The selected orbit's initial masses, positions and velocities, as decimals (for people) **and** as exact IEEE-754 bit patterns (`bits`, `0x` followed by 16 hex digits; authoritative). |
| `inputs.view` | The main edition's view of the orbit, which the bodies follow: `projection` (`position`, `phase_portrait`, `cross_braid` or `hodograph`), `rotation` (the viewing rotation, a 3 × 3 matrix), `drift` (`mode` `none`, `linear` with its `velocity`, or `elliptical` with its `rotation`, `mean_anomaly`, `mean_motion`, `eccentricity`, `semi_major` and `semi_minor`) and `frame` (`min_x`, `min_y`, `width`, `height`, `scale`). Every number is an exact bit pattern, like `inputs.bodies[*].bits`, with no decimal copy. |
| `inputs.width`, `inputs.height`, `inputs.paper_seed_sha256` | Output size and the SHA-256 of the paper seed. |
| `inputs.frames` | Frame count, first and last step, frame rate, a SHA-256 of the schedule (little-endian `u64` step indices), and `slow_factor` (10): how many times slower the slow film is. |
| `config` | The full `EmberConfig`: `fluid`, `contact`, `tidal` (`max_aspect`, `stretch_quantile`, `softening`), `look` (`floor`, `fade_fraction`, `hold_fraction`, `floor_tau`), `paper` and `raster`. |
| `derived` | The orbit duration and valve time in fluid units, the look's `hold_time` and `fade_time` in fluid units, the orbit's `tidal_reference` anisotropy, the fluid and ink grids (`fluid_grid`, `fluid_dx`, `ink_grid`), and `slow_first_frame`, the frame of `ember.mp4` at which the slow film starts. |
| `outputs.frames_rgb48le_sha256` | SHA-256 of all frames of `ember.mp4` concatenated as `rgb48le`; `null` for `--image-only`. |
| `outputs.slow_frames_rgb48le_sha256` | SHA-256 of all frames of the slow film concatenated as `rgb48le`; `null` when the slow film was not rendered (`--image-only`). |
| `outputs.still_rgb48le_sha256` | SHA-256 of the still as `rgb48le`. |
| `outputs.frames_emitted`, `outputs.slow_frames_emitted`, `outputs.encoding` | The frame counts of the two films and the colour encoding of the digested pixels. |
| `stats` | Deterministic render statistics: fluid steps (`fluid_steps`), time-step range (`min_dt`, `max_dt`), peak flow speed (`max_flow_speed`), `snapshots`, `contact_events`, the still's ink coverage (`still_ink_fraction`, the fraction of its visible ink nodes carrying any ink) and gamut-mapped pixels (`still_gamut_mapped_pixels`). |
| `build` | Informational only: crate version (1.1.0 for this release), CPU architecture, OS and thread count of the producing machine. |
| `timings_seconds` | Informational only: wall-clock seconds of the `fluid`, `ink`, `shade` and `sink` stages and the `total`. |

Layout 4 came with `ember-v3`. It added `inputs.view`, `inputs.frames.slow_factor`,
`derived.slow_first_frame`, `outputs.slow_frames_rgb48le_sha256` and
`outputs.slow_frames_emitted`, and dropped `config.projection` and `derived.projection`, the
principal-plane projection of `ember-v2`. Layout 3 came with `ember-v2`. It dropped
`ember-v1`'s cinnabar statistics (`stats.frames_with_cinnabar`,
`stats.peak_frame_cinnabar_fraction`, `stats.still_cinnabar_fraction`) and vermilion settings,
and added `config.tidal`, `derived.hold_time`, `derived.fade_time` and
`derived.tidal_reference`. The published `ember-v1` certificates have layout 2 (layout 1 was
only ever written by test renders).

Everything except `build` and `timings_seconds` must match between two renders of the same
inputs. `ember::EmberCertificate::read_json` reads a certificate back into typed Rust values. It
rejects other layout versions (this build reads only layout 4), unknown or missing fields (a
nullable field such as a frames digest must be present, as `null` or a value), malformed bit
patterns and outputs that disagree about the frames of either film, and rebuilds the initial
conditions and the view bit for bit from their bit patterns. The layout, the digest definitions
and the versioning rules are in
[ember-design.md §8.4](ember-design.md#84-the-certificate-certificaters-metadataemberjson).

### Verifying a package on another machine

`examples/ember_render.rs` re-renders a package from its certificate alone and compares the
digests it records, of the still and of the two films:

```bash
cargo run --release --example ember_render -- verify output/<name>/metadata/ember.json
```

The re-render takes as long as the original, so the tool first checks that this build can
reproduce the certificate at all. It stops with a message naming each field that rules it out:
outputs that disagree about the frames (frames emitted without a frames digest, or a count other
than the film's length), the rendering algorithm version, the integrator, the time step and
the gravitational constant (bit for bit), the paper-seed digest, the frame-schedule digest, and
whether the recorded configuration round-trips through this build's `EmberConfig`. The view and
the slow factor are inputs like the bodies: the re-render follows what the certificate records.
The tool then prints `MATCH` or `MISMATCH` for `still`, `frames` and `slow` (`(not recorded)`
for a film the certificate has no digest of), and for the frame counts (`outputs`), `derived`
and `stats`, which the re-render must reproduce as well. The exit status is 0 when everything
matches, 1 on a mismatch and 2 on an error. A build verifies only certificates of its own look:
an `ember-v1` or `ember-v2` certificate is an error for an `ember-v3` build (its layout cannot
be read).

The tool's `render` subcommand renders an orbit outside the generator, for look development:
`--video` and `--slow-video` encode the two films, `--slow-factor` sets another slow factor, and
`--frontal` draws the orbit as it is (plain positions, no rotation, no drift) instead of
through a certificate's view.

### Verifying on two machines

1. Build the same commit on both machines with `cargo build --release --locked`. The native-CPU
   flags in `.cargo/config.toml` do not affect the ember edition.
2. Render the same seed on each. Use `--image-only` for a faster check of the still alone, but use
   the same mode on both machines, because the frames digests exist only for full renders:

   ```bash
   ./target/release/three_body_problem --seed 0x46205528 --output verify
   ```

3. Compare the certificates without the informational sections:

   ```bash
   jq -S 'del(.build, .timings_seconds)' output/verify/metadata/ember.json > ember-$(uname -m).json
   diff ember-x86_64.json ember-arm64.json    # must print nothing
   ```

   A difference in `inputs.bodies` means that the two machines selected different orbits, and a
   difference in `inputs.view` that their main pipelines resolved different views (see the
   contract above; the view is computed with the platform's math library, so two operating
   systems can record views that differ in their last bits). The two certificates then certify
   different inputs: verify each of them on the other machine with the `verify` tool instead.

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
solid sRGB colours, including a saturated warm red `(177, 34, 16)`, must round-trip within 4
levels (8-bit 4:2:0) or 1 level (10-bit) on an 8-bit scale, a BT.601 decode must be more than 10
levels off (so the check discriminates), and both tag locations must be correct. It passes with
`FFmpeg` 7.1.1 and with the production host's `FFmpeg` 6.1.1. Its worst errors were recorded
only for 7.1.1: 3 levels (8-bit) and 0.25 levels (10-bit), against 22–24 for a BT.601 decode.
Run it on any encoding host:

```bash
cargo test --release --lib srgb_variants_round_trip -- --ignored --nocapture
```

The WebPs are derived from the sRGB PNG with the same recipe as the main images. WebP is sRGB by
definition.

## Runtime

Everything runs on the CPU. The fluid, the ink and the shading use the machine's cores through
rayon (the fluid solver at most 32 of them), and the three encoders are software (`libx264`
for the two web videos, `libx265` for the archival one); nothing uses a GPU or a hardware
encoder.

Every figure in this section was measured or estimated with the `ember-v2` look, which had no
slow film and showed the orbit in its own principal plane. `ember-v3` has not been timed yet.
Two things change its cost: the orbit's duration in fluid time, and with it the number of fluid
steps, now follows from the bodies' speeds in the main edition's view; and the slow film adds
ink and shading work for nine more frames per film frame and a third encoder (see
[How the cost scales](#how-the-cost-scales)). The preflight's `Ember plan:` line gives a lower
estimate of the fluid steps of a seed before anything is rendered.

### Measured cost

The stage times below are from one default `ember-v2` render of seed `0x46205528` (1,000,000
steps, 3456 × 2234): a standalone render with `examples/ember_render`, which runs the generator's
`render_ember` code path, and the cleanest ember measurement available. The last two rows come from
a full package of the same seed on the same machine, `three_body_problem --seed 0x46205528` with
`run.py`'s flags (100,000 sims, all outputs). The certificate's `timings_seconds` holds the fluid,
ink, shade, sink and total times; the generator's log line `Ember timings: … stage total …` adds
the orbit, PNG, WebP and certificate.

Host: Apple M4 Max (16 cores, 128 GB), aarch64 macOS. Other work (test suites, a second session)
shared the CPU during both runs, so every time here is an upper bound for this machine.
Orbit: 19.12 fluid time units, 78,356 fluid steps (58.9 ms each), 9,009 velocity snapshots.

| Stage | Wall time | Share |
|-------|-----------|-------|
| Fluid (`timings_seconds.fluid`) | 4,614 s | 62% |
| Ink (`timings_seconds.ink`) | 2,538 s | 34% |
| Shading (`timings_seconds.shade`) | 283 s | 4% |
| Sink and encoder back-pressure (`timings_seconds.sink`) | 61 s | 1% |
| **Render** (`timings_seconds.total`) | **7,496 s (2 h 05 min)** | 100% |
| **Ember stage** (log, with orbit, PNG, WebPs, certificate and the encoders' tail) | **About 2 h 34 min** in the package, whose ember render took 8,846 s (2 h 27 min) | |
| Whole package (main render, spectral gallery and sweep, ember edition) | 13,186 s (3 h 40 min) | |

Before its ember stage, the package spent 27.5 min on the Borda search, 8.4 min on the histogram
pass, 1.6 min on the levels and 28.2 min on the main video, the spectral gallery and the sweep.

**The same seed with the `ember-v1` look**, from its certificate (production host, x86_64,
128 threads): a render of 1,406 s (23.4 min), of which the fluid took 1,004 s for 45,438 steps
(22.1 ms each on the 1440 × 1024 grid), the ink 267 s on 39.0 million nodes, the shading 36 s and
the sink 97 s, over the same 9,009 snapshots. `ember-v2` takes 1.72 times as many fluid steps
(the finer grid's shorter CFL step), each about 2.2 times as costly on the production host
(48.9 ms with 32 threads, measured on the loaded host), and has 2.25 times as many ink nodes.

**The production host with `ember-v2`: estimates, not measurements.** From these ratios, the ember
stage should take about 75–80 minutes there (23 minutes for the `ember-v1` render), and a whole
package about 2 hours. Sync runs with the `ember-v1` look took about 70 minutes per package end to
end (sync log, 2026-09-30).

The package's peak memory on the M4 Max was 85 GB resident (a 118 GB macOS memory footprint). It
belongs to the main render, not the ember stage. On the production host, sync runs with the
`ember-v1` look peaked at 120 GB of its 503 GB (the sync unit's systemd `MemoryPeak`). The ember
render itself peaks at 4.9 GB (the standalone M4 Max render): the fluid state, the snapshot
window, the paper, one frame, and two ink-field buffers of 87.7 million nodes, four `f32` fields
each (2.8 GB together). The encoders add their own (x265's look-ahead holds several GB at this
size).

**Cross-architecture check at production scale (`ember-v2`).** On the M4 Max, the package's
ember still and frame stream had exactly the standalone render's digests, so the generator's
path reproduced the edition bit for bit:

```text
outputs.still_rgb48le_sha256   b34495d49de0db906dacd7f9b1c47c9aee29bd58439c31f7d23455a8c22608fd
outputs.frames_rgb48le_sha256  80c891251270b0b08aa5f9776fe5b0e5e38f287d8499319d6cbb4945a03f8483
```

The production host (x86_64, a release build for its native CPU with AVX2 and FMA) re-rendered
this edition at full scale with `examples/ember_render` while a sync run shared the host: the still
and all 1,802 frames matched the digests above bit for bit, and the render took 4,673 s (78 min:
fluid 3,839 s at 49.0 ms per step, ink 730 s, shading 63 s, sink 41 s). (For the `ember-v1`
look, an Apple M4 Max re-rendered an x86_64 production package from its certificate: the still
and all 1,802 frames matched bit for bit.) At small scale, the `ember-v2` golden renders of
`tests/ember_determinism.rs` and every ember unit golden were bit-identical on aarch64 macOS,
x86_64 Linux and x86-64-v3 (AVX2); CI runs the golden tests on x86_64 Linux (baseline and
x86-64-v3), aarch64 Linux and aarch64 macOS. The full-scale check is still to be repeated for
`ember-v3`, whose certificate also carries the slow film's digest.

`--image-only` was not timed separately: it skips the per-frame shading and all encoding, so it
costs about the fluid plus the ink.

### How the cost scales

| Stage | Scales with | Notes |
|-------|-------------|-------|
| Orbit re-simulation | steps | Done twice: by the preflight and by the ember stage, each time with the view, the time map and the tidal reference. Not timed on its own: in the M4 Max package (`ember-v2`), the ember stage spent about 7 minutes outside its render, on the orbit, the PNG and WebP writes and the encoders' tail together. |
| Fluid | orbit duration / time step | 26 real 2160 × 1536 FFTs per step: 58.9 ms on the M4 Max, 48.9 ms on the production host. The grid does not depend on the output size; the step count depends on the orbit's duration, the grid spacing and the peak speeds of the flow and the bodies (78,356 steps for 19.12 fluid units). The solver runs in its own pool of at most 32 threads: beyond that its transforms stop scaling. On the loaded production host a step took 55.8 ms with 16 threads, 48.9 ms with 32, 47.9 ms with 48, 50.2 ms with 64 and 63.6 ms with 96. |
| Ink | frames × ink nodes × snapshots per frame | About 87.7 million nodes (11374 × 7708, margin included) at 3456 × 2234, 2.25 times as many as with the 2 × 2 nodes of `ember-v1`. The 0.28-radius travel bound (0.5 for `ember-v1`) sets the snapshots per frame wherever the bodies move fast; since `ember-v3` their number is rounded up to a multiple of ten. Uses every core. |
| Ink of the slow film | film frames × visible ink nodes × snapshots per frame × 4.5 | The nine in-between frames of a frame interval trace through 1/10, 2/10, … 9/10 of its snapshots: 4.5 times the snapshot steps of the film frame's own remap, on the visible nodes only (the margin, about a fifth of the nodes, is skipped). Not rendered under `--image-only`. |
| Shading | frames × pixels × 9 nodes | 36-band Kubelka–Munk for inked nodes; cached paper elsewhere. Uses every core. With the slow film, about ten times as many frames are shaded. |
| Encoding | frames × pixels | Runs concurrently with rendering. The frames go to the encoders through pipes (the film's two and the slow film's one), so an encoder slower than the render would show up as sink back-pressure (`timings_seconds.sink`). In the M4 Max package (`ember-v2`, two encoders), it was 65 s of an 8,846 s render: the encoders kept pace. |

`--image-only` skips the per-frame shading, the slow film and all encoding, but the fluid and the
ink of the film frames still run over the whole orbit.
