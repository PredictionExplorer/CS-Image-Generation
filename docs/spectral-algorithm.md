# Spectral Image and Video Generation Algorithm

This document describes the complete algorithm for generating spectral images and
spectral videos from a finished three-body simulation. It assumes orbit selection
has finished: the Borda search shortlisted its top 24 orbits, and the aesthetic
pass re-simulated each one at the full step count, proxy-rendered it in the seed's
projection space under the seed's layer stack and a few curated fallback stacks,
and kept the best (orbit, stack) pair together with its trajectory. The spectral
renders use that trajectory directly, with no re-simulation (only the trait
analysis and the ember edition re-simulate the raw orbit).

---

## Table of Contents

1. [Inputs](#1-inputs)
2. [The SPD Buffer](#2-the-spd-buffer)
3. [Spectral Accumulation (Layer-Stack Line Splatting)](#3-spectral-accumulation)
4. [Energy-Density Redshift (Removed)](#4-energy-density-redshift-removed)
5. [SPD-to-RGBA Conversion](#5-spd-to-rgba-conversion)
6. [CosmicSignature Finish Pipeline](#6-cosmicsignature-finish-pipeline)
7. [Still Image Output](#7-still-image-output)
8. [Main Video Output](#8-main-video-output)
9. [Spectral Gallery (Per-Bin Images)](#9-spectral-gallery)
10. [Spectral Sweep Video](#10-spectral-sweep-video)
11. [Constants Reference](#11-constants-reference)

---

## 1. Inputs

After orbit selection hands over the winning trajectory (already in the seed's
projection space) and its layer stack, the rendering pipeline receives:

| Input | Type | Description |
|-------|------|-------------|
| `positions` | `&[Vec<Vector3<f64>>]` (3 bodies) | N timesteps per body of (x, y, z) in the seed's projection space, view-rotated and drifted (see below) |
| `colors` | `&[Vec<OklabColor>]` | Per-body OkLab (L, a, b) color at each timestep (Section 3.0) |
| `body_alphas` | `&[f64]` (3) | Per-body opacity, constant across time: `(1 / 15,000,000) x log-uniform[0.55, 1.80]` |
| `width`, `height` | `u32` | Output resolution, carried in `ResolvedEffectConfig` (default 3456x2234, `-r WxH`) |
| `hdr_scale` | `f64` | Global energy multiplier, sampled per seed uniformly in [0.225, 0.295] (`RenderConfig::hdr_scale`) |
| `resolved_config` | `ResolvedEffectConfig` | Finish parameters: halation (DoG) and prism strength/radius, `clip_black`/`clip_white` quantiles |
| `traits` | `SceneTraits` | Layer stack, line weight, age ramp, edge energy, symmetry, spikes, stardust and the per-vocabulary knobs (Section 3.3b) |
| `levels` | `ChannelLevels` | Display levels from pass 1 (the histogram pass), used by pass 2 and the still |

The positions are centre-of-mass simulation positions that have been (1) mapped into
the seed's projection space (`apply_projection`: plain position space, or a
phase-space projection whose velocity axes are rescaled to the position extent),
(2) rotated by the best-composed of 4 seeded Shoemake orientations
(`apply_view_orientation`, `cosmic-view/v1`), and (3) offset by the drift
(`--drift`, default `elliptical`). A `RenderContext` maps (x, y) to pixel
coordinates via a bounding box over all bodies and timesteps. Each axis is padded by
5% of its span on both sides (an axis with zero span first gets +/-0.5 units). Aspect
correction (on by default) then pads one axis symmetrically so the box matches the
output aspect ratio: `px = (x - min_x) / bbox_width * width`,
`py = (y - min_y) / bbox_height * height`. z does not move pixels; it feeds the
depth fade, the 3D segment length and the speed estimate.

---

## 2. The SPD Buffer

The core data structure is a **per-pixel spectral power distribution (SPD)** --
an array of energy values across wavelength bins.

### Layout

```
accum_spd: Vec<[f64; 64]>   // one [f64; 64] per pixel, row-major order
                              // total size: width * height * 64 * 8 bytes
                              // at 1080p: ~0.99 GiB (1920 * 1080 * 64 * 8)
                              // at the default 3456x2234: ~3.68 GiB (logged as the
                              // "Full-frame SPD memory estimate")
```

### Bin Parameters

| Parameter | Value |
|-----------|-------|
| Number of bins (`NUM_BINS`) | 64 |
| Spectrum start (`LAMBDA_START`) | 380 nm (deep violet) |
| Spectrum end (`LAMBDA_END`) | 700 nm (deep red) |
| Spectrum range (`LAMBDA_RANGE`) | 320 nm |
| Bin width (`BIN_WIDTH`) | 5.0 nm |

### Bin-to-Wavelength Mapping

The center wavelength for bin `i` (0-indexed) is:

```
center_wavelength(i) = 380.0 + (i + 0.5) * 5.0
```

This produces centers from 382.5 nm (bin 0) to 697.5 nm (bin 63).

### Wavelength-to-Bin Mapping

Given a wavelength in nanometers:

```
bin_f = clamp((wavelength - 380.0) / 5.0, 0.0, 63.0)
```

This yields a fractional bin index. Energy is not split between two bins: each
emission wavelength is deposited as a Gaussian lobe centred on `bin_f`, with
`sigma_bins` clamped to [0.45, 1.35] and a radius of `min(ceil(2 * sigma_bins), 3)`
bins. The kernel is normalised and scaled by the vertex's lightness energy factor
(Section 3.4). Because `bin_f` is measured from the 380 nm edge, a wavelength equal
to bin i's centre maps to `i + 0.5`. Emission wavelengths come from the
spectral-locus inversion over 405-690 nm, so strokes deposit only into bins 2-63.

---

## 3. Spectral Accumulation

This is the core rendering step. For every simulation timestep, the seed's layer
stack is drawn into the SPD buffer: the primary vocabulary (alpha 1.0) plus an
optional underlay and accent layer (Section 3.3b). Each layer turns the step's three
body vertices into strokes (triangle-web edges, ribbons, spokes, chords, veil fill
lines, weave Beziers, stipple dots or tangent segments). Each stroke is splatted as
an anti-aliased spectral line segment (Section 3.6) and replicated per the seed's
symmetry op, with energy divided by the fold count. Per-step energy is
`hdr_scale x age_factor x glow_pulse x layer.alpha x sheet_taper`, times the
per-stroke velocity multiplier. A seeded stardust field (when the seed has one) is
splatted once per accumulation pass, before the first step.

### 3.0 Procedural OkLab Palette Synthesis

Before rasterization, each body receives an OKLCh color sequence
(`generate_body_color_sequences`, forked `cosmic-color/v3` stream). There are no
preset palettes: a continuous genome is sampled and must pass a deterministic
beauty gate ("sample wild, gate hard").

```
anchor             = rng * 360
dispersion         = log_lerp(8, 300, rng)       // total hue span, degrees
skew               = rng * 2 - 1
chroma_peak        = log_lerp(0.34, 0.97, rng)   // floor 0.26 without chroma_boost
chroma_floor_ratio = lerp(0.25, 0.75, rng)
lightness_center   = lerp(0.46, 0.80, rng)
lightness_span     = lerp(0.10, 0.40, rng)
journey_scale      = rng
wave_freq          = lerp(0.6, 6.0, rng)
wave_amp           = log_lerp(3, 42, rng)        // degrees
accent_strength    = lerp(4, 40, rng)            // degrees
lightness_wave     = lerp(0.030, 0.160, rng)
chroma_wave        = lerp(0.020, 0.100, rng)
offsets            = [-0.5*dispersion, 0.35*skew*dispersion, +0.5*dispersion]
base_hue[body]     = anchor + offsets[hue_rank] + uniform(-10, 10)
```

Hue, lightness and chroma ranks are shuffled independently across bodies, and the
most chromatic (dominant) body is lifted to lightness >= 0.62. The gate rejects a
candidate when:

- two bodies' mean colors are closer than 0.085 in OKLab;
- mean chroma is outside [0.045, 0.30] (floor 0.070 when dispersion < 40 deg);
- the lightness ladder spans less than 0.04 (0.10 when dispersion < 40 deg);
- a red hue (< 35 or > 335 deg) meets a green one (92-152 deg).

Up to 24 genomes are drawn (`MAX_PALETTE_ATTEMPTS`). If all fail, the last one is
repaired deterministically: if a red hue meets a green one, the green-sector hues
rotate +55 deg; then, until the distance floor holds, hues spread by 0/+24/-24 deg, lightness moves halfway toward
0.78/0.62/0.44 and chroma fraction rises to at least 0.60; finally every chroma
fraction is raised to at least 0.45. Dispersion moves continuously through
monochrome, analogous, split and triadic spans with no fixed angle privileged, but
red/green opposition is excluded.

The resolved `palette_phase` (uniform [0, 1] from the `CosmicSignature` profile)
shifts the phase of both hue waves and raises their frequency:

```
t = step / N
f = wave_freq + 0.45 * palette_phase
hue = base_hue
    + (smoothstep(t) - 0.5) * hue_journey               // per body; |hue_journey| <= lerp(12, 110, journey_scale) * max(dispersion / 300, 0.25)
    + BASE_HUE_DRIFT * HUE_DRIFT_SCALE * (1 + ln(step)) // 1.4 * 1.15, ~24 deg by step 10^6
    + sin(TAU * (0.33 * body + phase_jitter + body_phase + t * f)) * wave_amp
    + sin(TAU * (palette_phase + 0.618 * body + body_phase + t * (0.37 * f + 0.71))) * accent_strength
    +/- 0.1                                             // random sign per step
phase_jitter = 0.1 * rng + palette_phase + 0.07 * body_phase
```

Lightness is `target + noise * range + wave * lightness_wave + 0.018 * accent`.
It is floored at 0.62 for the dominant body and clamped to [0.34, 0.94]. Chroma is
a fraction of the Display-P3 maximum chroma at that (L, h):
`fraction = plan + noise + wave * chroma_wave + 0.035 * |accent|`, floored at 0.62
for the dominant body and clamped to [0.10, 0.985]. The resulting chroma has an
absolute floor of 0.055 (dominant) or 0.026 (others) and a cap of 0.995 x max.
The gate and repair above enforce the palette's quality; the per-body
lightness/chroma ranks and gamut-relative chroma keep tight or unusual hue
relationships tasteful while seed-to-seed variety stays high.

### 3.1 Overview

```
if step_start == 0: splat the seed-gated stardust field (once per pass)
for step in step_start..step_end:
    vertices      = prepare_triangle_vertices(step)
    next_vertices = prepare_triangle_vertices(step + 1)   // while the simulation continues
    age_hdr = hdr_scale * age_factor(step) * glow_pulse(step)
    for layer in stack.layers():        // primary (alpha 1), optional underlay, accent
        step_hdr = age_hdr * layer.alpha * sheet_taper(layer.vocabulary, step / total)
        match layer.vocabulary:
            triangle_web | duet -> 3 (or 2) edges with per-edge velocity dynamics,
                                   interpolated toward step + 1 inside the chunk (§3.5)
            spokes              -> body-to-centroid lines, interpolated likewise
            orbit_ribbons       -> each body's trail step -> step + 1, plus echo bands
            time_chords         -> faint ribbon underlay (0.30) + chord step -> step + lag
            nebula_veil         -> fill lines across the pivot body's corner, every 2nd step
            harmonic_weave      -> 3 bowed Bezier chords (6 segments each), every 2nd step
            stipple_constellation -> one dot per body at the time pitch
            tangent_caustics    -> a velocity tangent per body, direction from step + 1
        every stroke -> symmetry copies (energy / fold count) -> SDF splat (§3.6)
```

### 3.2 Triangle Vertex Preparation

For each timestep `step`, build 3 vertices:

```
for body in 0..3:
    (pixel_x, pixel_y) = render_context.to_pixel(positions[body][step].x,
                                                   positions[body][step].y)
    vertex[body] = LineVertex {
        x: pixel_x,        // f32, pixel coordinates
        y: pixel_y,        // f32, pixel coordinates
        z: positions[body][step].z as f32,  // world-space depth
        color: colors[body][step],          // (L, a, b) in OkLab
        alpha: body_alphas[body],           // f64
    }
```

### 3.3 Orbit-Relative Velocity Dynamics

Each stroke receives brightness and width modulation derived from how fast the
relevant bodies move **relative to this orbit's own speed distribution**. (An
older build compared speeds against a fixed absolute threshold; bound orbits
move so much faster than that threshold that every segment saturated to the
maximum boost and the contrast was lost.)

At calculator construction, forward-difference speeds `|p[s+1] - p[s]| / dt` are
sampled with a deterministic stride of max(1, ⌊(steps − 1) / 4096⌋). Runs of up to
8,192 steps sample every step (up to 8,191 samples per body); longer runs get
between about 4,096 and 8,191 samples per body (4,099 at the default 1,000,000
steps). The samples are pooled across the three bodies and sorted. The trajectory sampled is the one being
rendered, after projection, view rotation and drift, so drift motion counts toward
speed. The low/high quantiles (index `round(q * (n - 1))`) define the
normalization window:

```
v_low  = quantile(VELOCITY_NORM_LOW_QUANTILE)    // 0.15
v_high = quantile(VELOCITY_NORM_HIGH_QUANTILE)   // 0.97
norm(v) = clamp((v - v_low) / (v_high - v_low), 0, 1)
```

If the window collapses, every moving body gets norm 0.6, and the final step
always gets neutral dynamics (1x energy, thickness 1).

Per segment: triangle-web edges and harmonic-weave chords use the mean of their
two endpoint body norms; ribbons, chords and echo bands, spokes and tangents use
the body's own norm; stipple dots use it for energy only (their width is 1.6, or
2.6 for pearls, × line weight); nebula-veil fill lines use the norm of the step's
pivot body. For every stroke except stipple dots, the resulting `thickness_factor`
is multiplied by the seed's line weight (with its width pulse) before it reaches
the splatter (§3.6):

```
s = smoothstep(norm)
hdr_multiplier   = 1 + s^VELOCITY_FLARE_GAMMA * (VELOCITY_HDR_BOOST_FACTOR - 1)
thickness_factor = lerp(VELOCITY_THICKNESS_SLOW, VELOCITY_THICKNESS_FAST, s)
```

| Constant | Value |
|----------|-------|
| `VELOCITY_HDR_BOOST_FACTOR` | 8.0 |
| `VELOCITY_NORM_LOW_QUANTILE` / `HIGH` | 0.15 / 0.97 |
| `VELOCITY_FLARE_GAMMA` | 1.35 |
| `VELOCITY_THICKNESS_SLOW` / `FAST` | 1.90 / 0.72 |
| `dt` (simulation timestep) | 0.001 |

Slow apoapsis arcs render bold and quiet near 1x energy; periapsis whips render
as thin flares approaching 8x. Because the window adapts per orbit, every seed
exhibits the full dynamic range.

### 3.3b Scene Traits (Layer Stack, Projection, Symmetry)

The `CosmicSignature` profile resolves seed-varying scene traits consumed by
the accumulator. Structure is a composed **layer stack** rather than a single
mode. The stack the seed rolls is its *preferred* stack. During orbit selection,
the top 24 Borda candidates are proxy-rendered under that stack and a few curated
fallbacks (the preferred primary alone, orbit_ribbons + harmonic_weave@0.29,
time_chords + harmonic_weave@0.24, solo orbit_ribbons, solo time_chords). A prior
bonus protects seed identity, and the best-scoring (orbit, stack) pair is
rendered, with up to 3 re-seeded retries while the current winner's raw proxy
score (its aesthetic total, without the prior bonus) is below the 0.82 quality
floor. The profile is then re-resolved for the chosen primary. The line-weight
range and halation gate follow that primary. The age-ramp range and echo bands
follow the stack the seed's rolls give that primary (`resolve_stack`), and only
then is the stack replaced by the chosen one:

- **Primary vocabulary** (seeded weighted choice over nine atoms):
  `triangle_web`, `orbit_ribbons` (with optional decaying echo bands), `duet`
  (one edge omitted), `spokes`, `time_chords` (string-art chords over a faint
  ribbon underlay), `nebula_veil` (the triangle interior swept as translucent
  gauze), `harmonic_weave` (bowed Bezier chords), `stipple_constellation`
  (time-pitched pointillist dots and pearls), `tangent_caustics` (velocity
  tangent envelopes).
- **Underlay layer** (~58% of seeds): a second vocabulary from a family other
  than the primary's, drawn from a weighted table (harmonic_weave 0.22,
  nebula_veil 0.18, time_chords 0.18, orbit_ribbons 0.16, triangle_web 0.08,
  stipple_constellation 0.08, spokes 0.05, tangent_caustics 0.05; duet never),
  at a log-uniform alpha in [0.12, 0.54] ([0.12, 0.70] for wildcards).
  **Accent layer** (~10%, independent of the underlay): another vocabulary from
  a family used by neither the primary nor the underlay (if any), at a
  log-uniform alpha in [0.05, 0.18]. Additive accumulation makes layer order
  irrelevant.
- **Projection axis** (~18% of seeds): instead of position space, the
  trajectory is rendered in a phase-space projection: `phase_portrait` (25% of
  these; (x, v_x, y)), `cross_braid` (50%; body i takes its y from body i+1) or
  `hodograph` (25%; (v_x, v_y, z)), with velocities rescaled to the position
  extent. This produces Lissajous-like curve families. The projection is applied
  during orbit selection, so candidates are scored in the space they are
  rendered in.
- **Symmetry op**: `none` (~85%), `mirror_x` (~3%), `rot{k}` k∈2..6 (~7%),
  `dih{k}` k∈{3,4,6} (~5%) — every stroke is replicated about the frame
  center with per-copy energy divided by the fold count (mandala/rosette
  compositions with stable exposure). Rotational and dihedral copies are also
  shrunk uniformly by min(width, height) / diagonal (≈0.54 at 3456×2234) so the
  rotated composition stays inside the frame. The stardust field is not
  replicated.
- **Wildcard seeds** (~7%): extended parameter ranges (bolder line weights,
  deeper veils, stronger halation); the aesthetic quality floor keeps the
  extremes presentable.
- **Line weight** (mode-aware range per primary vocabulary, maximum ×1.35 for
  wildcards), **age ramp** in [-0.6, 0.6] (±0.85 for wildcards; narrowed to
  [-0.6, 0.27], or [-0.85, 0.38] for wildcards, when the stack re-rolled for
  the chosen primary contains orbit_ribbons. That is the chosen stack unless a
  fallback won; a fallback without orbit_ribbons, such as solo time_chords, can
  still get the narrowed ramp), applied as `1 + age_ramp · (2t − 1)`, and **exposure key** in
  [0.94, 1.16] tune stroke rendering and display exposure per seed.
- **Width pulse** (~98% of seeds): a slow sinusoid along the timeline
  multiplies the line weight, `line_weight · (1 + amp · sin(2π(freq · t + phase)))`,
  with amp in [0.32, 0.60] ([0.32, 0.70] for wildcards), freq 2–8 cycles over
  the run and a uniform phase. Its brightness companion,
  `1 + 0.45 · amp · sin(2π(freq · t + phase) + π/3)`, scales step energy
  slightly out of phase, so strokes swell and glow independently of orbital
  speed.
- **Per-vocabulary tuning**: whenever the stack contains orbit_ribbons, ~94% of
  seeds add time-lagged echo bands: 2 (18%) or 3 bands at 1×, 2× and 3× the
  chord lag (0.6–2.6% of total steps), at alpha 0.30–0.56 times decays
  1.0 / 0.55 / 0.30. Web and weave layers weight their edges (0-1, 1-2, 2-0) by
  [e, 1, 2 − e] with the per-seed edge energy e ∈ [0.92, 1.18]; duet gives its
  dropped edge 0. Sheet-prone vocabularies (web, duet, spokes, time chords,
  veil, weave) are tapered along the timeline by `0.58 + 0.42 · sin(πt)^0.65`,
  so the first and last fans do not burn in as hard edges.
- **Finishing traits**: halation (mode-gated tight DoG bloom), prism (~5%,
  chromatic bloom), diffraction spikes (~4%, astrophoto star crosses marched
  from the brightest cores), stardust (~10%, a faint seeded micro-dot field
  splatted once per accumulation pass). All other legacy effects stay
  disabled.

A seeded 3D viewing orientation is applied to the trajectory before rendering.
Four Shoemake-uniform rotations are drawn from the forked RNG domain
`cosmic-view/v1`, each is scored on the projected 2D occupancy of a strided
sample (coverage/balance proxy for the chosen stack), and the best-composed one
rotates the whole trajectory. The same orbit family is therefore photographed
from a well-composed, seed-specific angle. The order is projection (applied
during selection), then view rotation, then drift. The layer composition
(underlay, accent), symmetry, projection, wildcard gate and the extra
per-vocabulary and finishing rolls (echo-band count, veil lines, weave bow,
stipple, tangent, spikes, stardust, width pulse) come from the forked
`cosmic-structure/v2` domain. The primary vocabulary, hdr_scale, clip points,
palette_phase, edge energy, exposure key, line weight, age ramp, chord lag, echo
gate/alpha, halation and prism come from the main seed stream.
Palette genomes come from `cosmic-color/v3` and must pass a deterministic
perceptual beauty gate: minimum OKLab distance 0.085 between mean body colours,
mean chroma between 0.045 (0.070 for near-monochrome palettes, dispersion < 40°)
and 0.30, a lightness-ladder span of at least 0.04 (0.10 near-monochrome), and no
red/green opposition. Up to 24 genomes are sampled; if none passes, the last is
deterministically repaired (Section 3.0).

### 3.4 OkLab Hue to Spectral Emission Lobes

Each vertex has an OkLab color (L, a, b). The hue chooses the dominant
wavelength region, while chroma controls the width of a Gaussian spectral
emission lobe. This means saturated colors emit narrow, pure spectral bands;
pastel colors emit broader bands that mix more naturally toward white.

```
hue_rad = atan2(b, a)
hue_deg = hue_rad.to_degrees()
if hue_deg < 0: hue_deg += 360
chroma = sqrt(a*a + b*b)
sigma_nm = 8 - 5.5 * clamp(chroma / 0.34, 0, 1)
sigma_bins = clamp(sigma_nm / 5.0, 0.45, 1.35)
```

The hue-to-wavelength map is the inverse of the spectral locus as this renderer
draws it. 257 wavelengths from 405 to 690 nm are converted through the CIE-derived
`wavelength_to_rgb` tint to OKLab hue, sorted by hue, and interpolated in reverse,
so equal hue steps map to equal perceptual steps with no breakpoints. A hue on that
arc emits a single lobe at the matching wavelength. A hue on the line of purples
(between the violet and red ends of the arc, across 0°) emits a pair of lobes at
the two arc ends (≈405 nm and ≈690 nm) whose weights crossfade with a smoothstep
across the gap, so magenta emerges without a special-cased wrap region.

For each lobe, energy is spread across at most ±3 nearby bins, and the kernel is
normalized to a lightness-dependent total:
```
center_bin = wavelength_to_bin(lobe_wavelength)          // (λ - 380) / 5
radius     = min(ceil(2 * sigma_bins), 3)
for bin in round(center_bin) - radius ..= round(center_bin) + radius:
    w = lobe_weight * exp(-(bin - center_bin)^2 / (2 * sigma_bins^2))
    keep if w > 1e-6                                      // pair lobes add into shared bins
scale all weights so they sum to lightness_energy(L)
lightness_energy(L) = 0.30 + 1.10 * clamp((L - 0.30) / 0.64, 0, 1)^1.6   // 0.30..1.40
```
The kernel encodes only hue and chroma, so lightness scales the deposited energy
instead: bright bodies radiate and dark bodies recede.

### 3.5 High-Resolution Triangle Interpolation

The renderer draws one sample per simulation step. Only the
instantaneous-geometry vocabularies, triangle_web/duet edges and spokes, insert
extra interpolated triangle samples, and only for high-resolution outputs or
unusually large default-resolution jumps. Ribbons, chords, echo bands and tangents
never interpolate. Veils and weaves draw every 2nd step and stipple dots draw at
their time pitch, with compensated energy. This keeps runtime and accumulated
energy stable while still helping extreme renders.

```
resolution_scale = clamp(min(width, height) / 2234, 0.55, 32.0)
max_motion_px = max distance any body moves between step and step + 1

if max_motion_px <= 1.0:
    substeps = 1
else if resolution_scale < 1.5 and max_motion_px < 12.0:
    substeps = 1
else:
    scale_factor = clamp(sqrt(resolution_scale), 1, 2) if resolution_scale >= 1.5 else 1
    substeps = clamp(ceil(max_motion_px / (2.5 / scale_factor)), 1, 8)

for substep in 0..substeps:
    sample = exact current-step triangle if substep == 0
             else lerp(triangle[step], triangle[step + 1], substep / substeps)
    sample_hdr_scale = step_hdr_scale / substeps
    draw sample
```

Here `step_hdr_scale` is the step's full energy (`hdr_scale × age × glow pulse ×
layer alpha × sheet taper`). The `1 / substeps` energy compensation is important:
interpolation increases spatial sampling density, not exposure. The first substep
preserves the exact simulation knot, so low-motion intervals do not smear to their
midpoint. Checkpointed video frames never interpolate past their current
checkpoint (web and spoke samples interpolate toward step + 1 only while it lies
inside the chunk). Trail-type strokes do cross the checkpoint on purpose. Ribbons
always draw step → step + 1, because otherwise every checkpoint would leave a
one-step gap. Time chords and echo bands reach step + lag (up to 3 × lag), and
tangents take their direction from step + 1. A frame can therefore show trail and
chord geometry slightly ahead of its checkpoint, and the accumulated still is
seamless across checkpoints.

### 3.6 Crisp SDF Line Segment Splatting

Every stroke of every vocabulary (web edges, trails, chords, spokes, veil lines,
weave segments, tangents, and the zero-length stipple and stardust dots, for which
h is fixed at 0.5) is rasterized as an anti-aliased line segment with a steep
super-Gaussian falloff. This is the innermost loop and deposits energy into the
SPD buffer. The splatter itself adds no depth blur and no broad line halo. Any
glow comes from the seed-gated halation (tight DoG bloom) and prism finishing
traits applied afterwards (Section 6.1).

**Setup per segment:**

Given start vertex `v0` and end vertex `v1`:

```
dx = v1.x - v0.x
dy = v1.y - v0.y
dz = v1.z - v0.z
len_sq = dx*dx + dy*dy           // 2D length squared (pixel space)
len_3d = sqrt(dx*dx + dy*dy + dz*dz)

// Dynamic line width: short segments draw bold, long spans fine; seed line
// weight and velocity dynamics arrive via thickness_factor; all constants
// scale with the output short edge.
resolution_scale  = clamp(min(width, height) / 2234, 0.55, 32.0)
normalized_len_3d = len_3d / resolution_scale
proximity = 1 / (0.55 + normalized_len_3d * 0.008)
thickness = clamp(0.95 * resolution_scale * thickness_factor * proximity,
                  0.30 * resolution_scale,
                  7.00 * resolution_scale)
// thickness_factor = velocity thickness (§3.3) × line weight × width pulse
// (stipple dots: 1.6, pearls 2.6, × line weight; stardust: dot size 0.45–1.60)

// Crisp production mode disables depth-of-field broadening
avg_z = (v0.z + v1.z) * 0.5
coc = |avg_z * 0.0|
effective_thickness = thickness + coc

// Bounding box padding: exp(-2 (d/t)^4) falls below the 0.004 cutoff at d ≈ 1.29·t
pad = ceil(effective_thickness * 1.5) + 1
```

**Spectral kernels for endpoints:**

```
kernel0 = spectral_kernel_for_oklab(v0.color)
kernel1 = spectral_kernel_for_oklab(v1.color)
```

**Energy conservation and depth fade:**

```
energy_conservation = thickness / effective_thickness     // 1.0: no depth broadening
depth_fade          = clamp(exp(-|avg_z| * 0.0007), 0.18, 1.0)
base_energy_mult    = segment.hdr_scale * depth_fade * energy_conservation
// segment.hdr_scale = hdr_scale × age factor × glow pulse × layer alpha × sheet taper
//                     × velocity hdr_multiplier × edge weight / substeps / fold count
//                     (× per-vocabulary factors: veil stride ÷ fill count, weave stride
//                     ÷ 6 segments, stipple pitch × 0.55 (× 2 for pearls), tangent length
//                     norm, echo alpha × decay, 0.30 for time_chords' ribbon underlay)
```

Stardust dots are the exception: they use `segment.hdr_scale = 1` and carry their
energy in the vertex alpha, `brightness × mean body alpha × N × hdr_scale × 0.0012
× (0.4 + 0.6 · glow) × twinkle`, with no age, glow pulse, layer alpha, taper,
velocity or symmetry factor.

**Per-pixel loop** (over all pixels in the padded bounding box):

```
for py in min_y..=max_y:
    for px in min_x..=max_x:
        coverage_sum = start_energy_sum = end_energy_sum = 0
        for each 2x2 subpixel sample at (px + (sx + 0.5) / 2, py + (sy + 0.5) / 2):
            h = projection onto the segment clamped to [0, 1] (0.5 for a zero-length segment)
            dist_sq = squared distance from sample to segment
            normalized = dist_sq / (effective_thickness * effective_thickness)
            energy = exp(-(normalized * normalized) * 2.0)
            alpha = v0.alpha * (1 - h) + v1.alpha * h
            coverage_sum     += energy                  // geometric, not alpha-weighted
            start_energy_sum += energy * alpha * (1 - h)
            end_energy_sum   += energy * alpha * h
        if coverage_sum / 4 < 0.004: continue           // tiny alphas never culled
        for (bin, weight) in kernel0: accum_spd[pixel][bin] += base_energy_mult * start_energy_sum * weight / 4
        for (bin, weight) in kernel1: accum_spd[pixel][bin] += base_energy_mult * end_energy_sum * weight / 4
```

The production path keeps this fixed 2x2 coverage grid. Higher-order AA can be
revisited later behind an explicit quality mode, but it is not part of the
default crisp profile.

### 3.7 Parallelization

**Scanline Bands** (every accumulation except tiled stills): the image is divided into
`min(height, rayon threads)` horizontal bands of `ceil(height / bands)` rows. Each
band processes every simulation step of the current chunk, including the stardust
field clipped to its rows, but writes only its own rows. No synchronization is
needed since bands don't overlap. A test-only serial reference runs the same
routine over the whole frame, and tests assert the parallel output is
bit-identical.

**Tiled Stills:** in `--image-only` runs, a final still above 32,000,000 pixels
with neither halation nor prism enabled is rendered in 384-row stripes, each with
guard rows above and below (`max(4, ceil(7.00 · scale · 1.5) + 1 + 2)`). Each
stripe, with its guard rows, re-runs every step on a single thread
(`accumulate_spectral_steps_into_rows`), and stripes are processed sequentially.
Stripes are accumulated, finished, tone-mapped and quantised one at a time, so
the still itself never allocates a full-frame SPD buffer (pass 1, the histogram
pass, still allocates one, so the run's peak memory does not drop); spikes run
stripe-locally there (Section 7).

---

## 4. Energy-Density Redshift (Removed)

Earlier builds warmed high-energy pixels by shifting their spectral power toward
longer (redder) wavelengths to simulate a "heat" effect. This biased bright
cores, overlaps, and fast-trail flares toward orange/red and reduced color
variety, so it has been removed from the production pipeline. Accumulated SPD
energy now flows directly into SPD-to-RGBA conversion with no wavelength shift.

---

## 5. SPD-to-RGBA Conversion

This stage converts the per-pixel 64-bin SPD into linear-space premultiplied
RGBA in a single parallel pass over the buffer (`convert_spd_buffer_to_rgba` in
`src/render/effects.rs`). Each pixel goes through two sub-steps: the radial
dispersion gather (5.1, a no-op in production) and the SPD-to-RGBA conversion
(5.3). The conversion runs on AVX2, NEON or a portable scalar kernel
(`src/spectrum_simd.rs`).

### 5.1 Radial Spectral Dispersion (Disabled in Crisp Mode)

The renderer still contains a radial spectral-dispersion path for experiments,
but production CosmicSignature output sets the active dispersion strength to
zero. This prevents wavelength-dependent spatial shifts from becoming chromatic
blur at high resolution.

**Setup:**

```
cx = width / 2.0
cy = height / 2.0
max_r = sqrt(cx*cx + cy*cy)
dispersion_strength = 0.0   // SPECTRAL_DISPERSION_STRENGTH (or _BOOSTED); both alias CRISP_DISPERSION_STRENGTH
```

**Per pixel:**

```
dx = pixel_x - cx
dy = pixel_y - cy
r = sqrt(dx*dx + dy*dy)
dir_x = dx / r    // radial unit vector (0 at the exact centre)
dir_y = dy / r
r_norm = r / max_r  // normalized distance from center [0, 1]

local_spd = [0.0; 64]

for bin in 0..64:
    // Bin offset: normalized position within the spectrum, centered at 0
    bin_offset = (bin - 31.5) / 31.5     // range [-1, +1]

    // Spatial shift: blue bins (bin_offset < 0) gather from farther out, so blue content
    // moves inward; red bins gather from nearer the centre, so red content moves outward
    shift = bin_offset * dispersion_strength * r_norm * 50.0

    // Sample from shifted source position
    sx = round(pixel_x - dir_x * shift)
    sy = round(pixel_y - dir_y * shift)

    if (sx, sy) is within image bounds:
        local_spd[bin] = src_spd[sy * width + sx][bin]
    else:
        local_spd[bin] = 0.0
```

In production, the branch is skipped and `local_spd = src_spd[pixel]`.

### 5.2 Energy-Density Redshift (Removed)

The energy-density redshift described in Section 4 has been removed, so no
per-pixel wavelength shift is applied here. `local_spd` passes directly to the
SPD-to-linear-RGBA step below.

### 5.3 SPD to Linear RGBA

The final conversion maps the 64-bin spectrum to CIE XYZ, then to a linear
Rec.2020 working RGB tuple. This replaced the older Dan Bruton wavelength
approximation with CIE 1931 2-degree color matching functions.

#### The CIE XYZ LUT

A 64-entry LUT (`BIN_XYZ_LUT`) is built lazily on first use. Each entry stores
`(X, Y, Z, k)`:

- **(X, Y, Z):** Simpson-integrated CIE 1931 2° color matching values over the
  bin's 5 nm span (bins cover 380-700 nm; the 5 nm CIE table is linearly
  interpolated). Each column is scaled so the 64 bins sum to D65 white
  (0.95047, 1.0, 1.08883). The conversion takes an energy-weighted average of the
  entries, so a spectrum with equal mapped energy in every bin lands on D65's
  chromaticity at 1/64 of its level.
- **k (tone steepness):** `k = 1.78 - 0.18 * (λ - 380) / 320` with λ the
  bin-centre wavelength, so k falls linearly from ≈1.779 (bin 0) to ≈1.601
  (bin 63) and highlights stay colorful without one spectral edge dominating.

#### Conversion Algorithm

```
X_sum = 0, Y_sum = 0, Z_sum = 0, total = 0

for i in 0..64:
    e = local_spd[i]
    if e <= 1e-10: continue

    (lut_X, lut_Y, lut_Z, k) = BIN_XYZ_LUT[i]

    // Per-bin tone mapping: soft saturation curve
    e_mapped = 1.0 - exp(-k * e)

    total += e_mapped
    X_sum += e_mapped * lut_X
    Y_sum += e_mapped * lut_Y
    Z_sum += e_mapped * lut_Z

if total < 1e-10:
    return (0, 0, 0, 0)    // black pixel

X = X_sum / total
Y = Y_sum / total
Z = Z_sum / total

// Perceptual vibrance in OkLab/OkLCh, not linear RGB buckets
(base_R, base_G, base_B) = xyz_to_linear_rec2020(X, Y, Z)
(L, a, b) = linear_rec2020_to_oklab(base_R, base_G, base_B)
C = sqrt(a*a + b*b)
amount = 0.48                 // SAT_BOOST_ENABLED (default on); 0.24 when off
knee = 0.11
highlight_guard = 1 / (1 + 0.035 * total)
boost = 1 + amount * knee / (C + knee) * highlight_guard
(R, G, B) = oklab_to_linear_rec2020(L, a * boost, b * boost)

// Brightness from total accumulated energy
brightness = 1.0 - exp(-total)

// Bright cores drift back toward the un-boosted hue
core_hue_mix = min(0.18 * brightness^2, 0.18)
(R, G, B) = lerp((R, G, B), (base_R, base_G, base_B), core_hue_mix)
(R, G, B) = preserve_hue_gamut_map(R, G, B)   // scale chroma toward Rec.709 luma until in [0, 1]

// Premultiplied alpha output
output = (R * brightness, G * brightness, B * brightness, brightness)
```

The output is a premultiplied-alpha linear Rec.2020 tuple in [0, 1]. The alpha
channel (brightness) represents how much light the pixel received overall.
After tone mapping (Section 6), the display values are matrixed from Rec.2020 to
Display P3 primaries and quantised to 16 bits. The master PNG is labelled with P3
`cHRM` chromaticities and `gAMA` 1/2.2. Video frames go to FFmpeg as 16-bit
`rgb48le`. The high-quality HEVC copy (or the H.264 that replaces it under
`--fast-encode`) keeps the P3 frames and is tagged with P3 primaries (`smpte432`).
The web H.264 copy and the WebPs get the frames converted to sRGB first (Section
6.4).

---

## 6. CosmicSignature Finish Pipeline

After SPD-to-RGBA conversion produces a linear RGBA buffer, the CosmicSignature
profile runs a short finish path. Its only effects are a few seed-gated finishing
traits: halation, prism and diffraction spikes. The legacy Gaussian bloom and the
post-tone-map image effects are never switched on, so the image is still driven
by spectral geometry, transparent overlap, thin luminous edges, clean tonemapping,
and black negative space.

### 6.1 Trajectory Effects (`process_trajectory`) and Spikes

The trajectory chain (`FinishEffectPipeline::build_trajectory_chain`) holds:

- **Halation (DoG bloom)** when the seed's halation gate passes. The gate
  probability depends on the primary vocabulary: 1.0 for orbit ribbons and
  stipple constellations, 0.82 for time chords, 0.74 for harmonic weave, 0.60 for
  tangent caustics, 0.50 for spokes, 0.40 for triangle web/duet and 0.35 for
  nebula veil. Inner sigma is 0.0030-0.0050 × min(width, height) for orbit
  ribbons and stipple constellations, and 0.0026-0.0046 × min(width, height)
  for the other vocabularies; the
  outer/inner ratio is 2.8-4.0, strength comes from the vocabulary's range
  (wildcards raise the cap by 1.6×, to at most 0.30), and the threshold is 0.012
  (0.016 with prism).
- **Prism (chromatic bloom)** for 5% of seeds: strength 0.16-0.30, radius
  0.0035-0.0050 × min dim, channel separation 0.0008-0.0014 × min dim, threshold
  0.20-0.26.

A seed that rolls neither has an empty chain, and its buffer passes through
unchanged. Next, `apply_spike_finish` adds the rare diffraction-spike trait to 4%
of seeds: 4 arms (62%) or 6, strength 0.10-0.32, length scale 0.020-0.048,
threshold fraction 0.65. It runs in every render path (histogram, video frames,
stills), so exposure analysis sees the spiked image.

### 6.2 Tone Mapping

A custom tone mapper uses histogram-derived exposure levels:

1. **Pass 1 (histogram):** Accumulate the trajectory to about 240 evenly spaced
   checkpoints (`DEFAULT_HISTOGRAM_SAMPLE_FRAMES`; interval = steps / 240, plus
   the final step). At each checkpoint, convert SPD to RGBA, run the trajectory
   chain and the spike finish, then push every pixel's alpha-premultiplied
   `(r·a, g·a, b·a)` into the histogram. `analyze_tonemapping` picks per-channel
   black and white points at the seed's `clip_black` (0.0045-0.0085) and
   `clip_white` (0.9935-0.9985) quantiles (the 0.45th-0.85th and
   99.35th-99.85th percentiles; index `round(q · N)`). It sets
   `exposure_scale = min(0.88 / L_white, 1)` with a floor of 0.35, where
   `L_white` is the `clip_white` quantile of normalised Rec.709 luminance. If
   more than 0.25% of samples would exceed 1.10, a governor darkens exposure by
   `1 + 1.5·(ratio / 0.0025 − 1)`. The result is multiplied by the seed's
   exposure key (0.94-1.16). `ChannelLevels` stores `black`, `range`
   (white − black), `exposure_scale`, `paper_white` (0.92) and
   `highlight_rolloff` (2.25). No gamma is computed.

2. **Apply tone map (`tonemap_core`):** For each pixel, multiply RGB by alpha
   (the same premultiplication the histogram sampled), then compute
   `max(v − black, 0) / range · exposure_scale` per channel. Apply the AgX inset
   matrix, log2-allocate over [−10, +2.5] EV and apply a 6th-order polynomial
   sigmoid. Then apply the AgX "punchy" outset matrix (`ACES_TWEAK_ENABLED`, on
   by default; otherwise the default AgX outset). Finally, compress Rec.709
   luminance above `paper_white` with the shoulder
   `pw + (1 − pw)·(1 − exp(−(L − pw)·rolloff / (1 − pw)))`, scaling RGB
   uniformly, and clamp to [0, 1].

### 6.3 Image Effects (`process_image`)

Default production chain: empty. The display buffer is quantized without a grain
or texture overlay.

### 6.4 Display P3 Quantization

The tone-mapped display buffer (AgX output, not scene-linear; no further transfer
curve is applied) is matrixed from Rec.2020 to Display P3 primaries and quantised
to 16-bit RGB (`quantize_display_buffer_to_16bit`). The PNG writer sets `gAMA`
1/2.2 and a `cHRM` chunk with the P3 primaries and D65 white. It also fills the
cICP field (primaries 12, transfer 13, matrix 0, full range), but the `png` 0.18
encoder ignores that field, so the file has no `cICP` chunk. Colour-managed viewers
therefore decode the PNG with a pure 2.2 power curve, slightly darker in the deepest
shadows than the sRGB curve the code values follow.

The archival video (the HEVC copy, or its `--fast-encode` H.264 replacement) keeps
the P3 frames: `-color_primaries smpte432 -color_trc iec61966-2-1`, plus
`colorprim=smpte432` in the x265 VUI. Everything a browser shows directly is sRGB:
the web H.264 copies of the main and sweep videos and the two WebPs get the P3
frames converted by `render::display_p3::to_srgb_samples` (decode the sRGB curve,
linear P3 → linear BT.709 primaries with the exact 3×3 matrix of the two primary
sets, clip each channel to [0, 1], encode the sRGB curve; neutral pixels are copied
unchanged), and the web videos are tagged BT.709 primaries, `iec61966-2-1`
transfer. Every encode converts RGB to Y′CbCr with an explicit
`scale=out_color_matrix=bt709:out_range=tv` and stamps its tags on the frames with
`setparams`, so FFmpeg 6.1 (which otherwise converts with BT.601 while tagging
BT.709) and 7.1 (which otherwise drops the primaries and transfer options) write
the same, correctly tagged streams.

```
for each pixel of the tone-mapped buffer:
    (p3_r, p3_g, p3_b) = linear_rec2020_to_display_p3(r, g, b)
    // Quantize to [0, 65535]
    output_u16 = round(clamp(p3_channel, 0, 1) * 65535)
```

---

## 7. Still Image Output

To produce a single spectral image (16-bit PNG):

```
1. Initialize accum_spd to zeros: Vec<[f64; 64]> of size width * height
2. Accumulate all simulation steps (0..total_steps) into accum_spd
3. convert_spd_buffer_to_rgba(accum_spd) -> linear RGBA buffer
4. process_trajectory(rgba_buffer)  -> DoG halation if the seed's halation gate passed,
                                       chromatic bloom for prism seeds (unchanged when neither)
5. apply_spike_finish(rgba_buffer)  -> diffraction spikes for spike seeds (4%)
6. tonemap(rgba_buffer, channel_levels) -> display-space RGBA
7. process_image(rgba_buffer)       -> unchanged display RGBA (image chain is always empty)
8. quantize to 16-bit Display P3 (Rec.2020 -> P3 matrix on the tone-mapped values)
9. Save as images/source/master.png (16-bit RGB, P3-tagged), then convert the
   frame to sRGB (Section 6.4) and pipe it to FFmpeg/libwebp for
   images/web/full.webp (full size) and images/web/preview.webp (at most 640 px
   wide; preset picture, quality 82, compression level 6, metadata stripped)
```

In a default run the still is the last frame of pass 2 (checkpoint
`total_steps − 1`, so every step is accumulated), taken from the same render that
feeds the videos. Only `--image-only` renders the still on its own with
`render_final_frame_spectral`.

In `--image-only` runs, a still larger than 32,000,000 pixels
(`HIGH_RES_TILED_PIXEL_THRESHOLD`) whose seed has neither halation nor prism
switches to a striped renderer (`render_final_frame_spectral_tiled`). It renders
384-row bands (`HIGH_RES_TILE_ROWS`) with `max(4, ceil(1.5 · 7.00 · scale) + 1 + 2)`
guard rows above and below, so cross-band splats are not clipped. Each band is
tone-mapped, quantised to Display P3 and stitched into the 16-bit image, so the
still itself never allocates a full-frame `width * height * 64 * f64` SPD buffer
(pass 1, the histogram pass, still allocates one, so the run's peak memory does
not drop). Spikes are applied
band-locally. Default runs, which render the still as the last video frame,
always hold the full-frame SPD buffer. The generator logs a full-frame SPD memory
estimate (STAGE 4) before rendering, so extreme-resolution jobs are visible up
front.

---

## 8. Main Video Output

The main video progressively reveals the orbit over time. Each frame shows the
accumulated trajectory up to a specific simulation checkpoint.

### Frame Scheduling

```
total_steps    = number of recorded orbit steps (positions per body)
target_frames  = DEFAULT_TARGET_FRAMES = 1800   // nominal 30 s at 60 fps
fps            = DEFAULT_VIDEO_FPS = 60
frame_interval = max(total_steps / target_frames, 1)   // integer division

// Checkpoints (render::main_video_checkpoints): every frame_interval-th
// step below total_steps, then the final step total_steps - 1 unless it
// is already the last checkpoint
checkpoints = [frame_interval, 2*frame_interval, ..., k*frame_interval, total_steps - 1]
```

The last frame therefore always shows the whole trajectory. At the default
1,000,000 steps this is 1,802 frames (about 30.03 s): every 555th step up to
999,555, then step 999,999. The ember edition renders exactly these
checkpoints, so frame `i` of both videos shows the same moment.

### Per-Frame Pipeline

For each checkpoint (frame) in sequence:

```
1. Accumulate the new steps into accum_spd:
   accumulate_spectral_steps(accum_spd, step_start, checkpoint + 1)
   (step_start = previous checkpoint + 1, or 0 for the first frame, so
   accumulation is incremental and no step is drawn twice)

2. Convert: convert_spd_buffer_to_rgba(accum_spd) -> linear Rec.2020 RGBA

3. Trajectory finish: process_trajectory(RGBA) -> DoG halation when the
   seed's halation gate passed, then chromatic bloom for prism seeds
   (unchanged for seeds with neither)

4. Diffraction spikes: apply_spike_finish for the rare spike seeds

5. Tone map with the ChannelLevels from Pass 1 (AgX curve, then the
   luminance shoulder above paper white)

6. Image finish: process_image(display) -> empty chain, unchanged

7. Quantize: linear_rec2020_to_display_p3, clamp to [0, 1],
   round(x * 65535) -> packed 16-bit RGB (rgb48le, 6 bytes per pixel)

8. Write the P3 frame bytes to the HQ encoder's stdin pipe, then the frame
   converted to sRGB (Section 6.4) to the web encoder's
```

The frame of the final step (`total_steps - 1`) is also kept. After
encoding it is saved as the master still `images/source/master.png`, with
its WebP derivatives (from the sRGB conversion of that frame). The fully accumulated `accum_spd` is returned for the
spectral gallery (Section 9) and the sweep video (Section 10).

### Video Encoding

Frames are streamed as raw `rgb48le` data (16-bit RGB, little-endian, no
alpha) to two FFmpeg processes started up front, as two streams
(`create_video_groups_from_frames`): the sRGB conversion of every frame to the
web encoder and the Display P3 frames to the HQ encoder:

| Parameter | `videos/web/main.mp4` (`web_compatible_srgb`) | `videos/hq/main.mp4` (`high_quality`) | HQ with `--fast-encode` (`fast_encode`) |
|-----------|------|------|------|
| Codec | libx264 | libx265 (Main 4:2:2 10, `hvc1` tag) | libx264 |
| Preset | medium | slower | fast |
| CRF | 18 | 17 | 21 |
| Pixel format | yuv420p (8-bit 4:2:0) | yuv422p10le (10-bit 4:2:2) | yuv420p10le (10-bit 4:2:0) |
| Tuning | - | `-tune grain`, `-profile:v main422-10`; `-x265-params bframes=8:ref=6:rc-lookahead=250:aq-mode=3:aq-strength=1.0:psy-rd=2.5:psy-rdoq=1.5:deblock=-1,-1:no-sao=0:colorprim=smpte432:transfer=iec61966-2-1:colormatrix=bt709:qg-size=8:rdoq-level=2` | `-tune film` |
| Frames | sRGB (converted) | Display P3 | Display P3 |
| Colour | explicit BT.709 conversion; BT.709 primaries, sRGB transfer (`iec61966-2-1`), BT.709 matrix, tv range | explicit BT.709 conversion; Display P3 primaries (`smpte432`), sRGB transfer, BT.709 matrix, tv range | as HQ |
| Input / FPS | rgb48le / 60 | rgb48le / 60 | rgb48le / 60 |

All three use `+faststart`. The render loop writes each frame to the HQ
encoder's pipe, then its sRGB conversion to the web encoder's. The writes block, and there is no
intermediate queue or writer thread, so rendering runs at the pace of the
slower encoder. If the render or a write fails, every encoder is killed
before its pipe closes and no video is finalised.

---

## 9. Spectral Gallery

The spectral gallery produces 64 individual 16-bit PNGs, one per wavelength
bin. These reveal which parts of the image contain energy at each wavelength.

### Per-Bin Image Algorithm

After full accumulation:

```
for bin in 0..64:
    wavelength = 380.0 + (bin + 0.5) * 5.0
    (tint_r, tint_g, tint_b) = wavelength_to_rgb(wavelength)

    // Find the maximum energy for this bin across all pixels
    max_val = max(accum_spd[pixel][bin] for all pixels)
    max_val = max(max_val, 1e-10)   // prevent division by zero

    for each pixel:
        normalized = clamp(accum_spd[pixel][bin] / max_val, 0, 1)

        // Tint by wavelength colour and apply display gamma
        R = (normalized * tint_r) ^ (1/2.2)
        G = (normalized * tint_g) ^ (1/2.2)
        B = (normalized * tint_b) ^ (1/2.2)

        // Convert to Display P3 and quantize to 16-bit (truncating)
        (p3_r, p3_g, p3_b) = linear_srgb_to_display_p3(R, G, B)
        pixel_out = floor(clamp(p3_channel, 0, 1) * 65535)

    save as "spectral/{bin:02}_{wavelength:.0}nm.png"   // tagged Display P3
```

The gallery is written only by video runs, from the `accum_spd` that pass 2
returns. `--image-only` skips it.

### Wavelength-to-RGB (CIE-derived tint)

Used for tinting bin images. Returns a display-safe linear sRGB colour for
a given wavelength:

```
(X, Y, Z) = CIE 1931 2-degree CMFs at lambda   // 5 nm table, linear interpolation;
                                               // 0 outside 380-700 nm
(R, G, B) = xyz_to_linear_srgb(X, Y, Z) / max(X, Y, Z)
if min(R, G, B) < 0:  subtract min(R, G, B) from every channel   // lift into gamut
if max(R, G, B) > 1:  divide every channel by max(R, G, B)
clamp each channel to [0, 1]
```

An intensity falloff factor is applied near the edges of the visible range:

```
if 380 <= lambda < 420:  factor = 0.35 + 0.65 * (lambda - 380) / 40
if 420 <= lambda < 645:  factor = 1.0
if 645 <= lambda <= 700: factor = 0.35 + 0.65 * (700 - lambda) / 55
otherwise:               factor = 0
```

Final: `(R * factor, G * factor, B * factor)`

---

## 10. Spectral Sweep Video

After the main still image and gallery, the pipeline encodes web and HQ
spectral sweep videos (`videos/web/spectral_sweep.mp4` and
`videos/hq/spectral_sweep.mp4`). They animate a smooth sweep through
wavelength bins using precomputed bin images, **Gaussian blending** across bins
(not simple two-bin linear interpolation), a **constant-speed ping-pong sweep**
with a gentle wavelength vibrato, and a dynamic **active bin range** so
mostly-empty bins at the spectrum edges can be skipped.

### 10.1 Shared setup: `BinBuffers`

The same per-bin float RGB images as in Section 9 are built from the fully
accumulated SPD buffer: for each bin, normalize
that bin's energy across all pixels, tint by `wavelength_to_rgb`, and apply
display gamma. The result is 64 parallel buffers of `[f32; 3]` per pixel.

### 10.2 Active bin range

Before choosing the sweep endpoints, the implementation scans the bin buffers to
find the first and last bins with visible RGB energy (above a small threshold),
pads by two bins on each side, and clamps to `[0, NUM_BINS - 1]`. If no bin
looks active, it falls back to fixed defaults (`SWEEP_BIN_START` ..
`SWEEP_BIN_END`, currently 4..=59). The sweep only traverses this inclusive
range `[active_start, active_end]`, not necessarily the full 0..63 span.

### 10.3 Sweep timing and centre bin

Let `total_frames` be `CYCLE_TOTAL_FRAMES` (600), `frame` in `0..total_frames`,
and `leg_t` be the normalized position within the current outbound or return
leg (each `total_frames / 2` frames long). The first leg sweeps from violet
to red, and the second sweeps back from red to violet, so the first and last
frames share the same centre bin.

The sweep runs at constant speed (no easing):

```
bin_f = active_start + leg_t * (active_end - active_start)
```

A small vibrato is then added, faded out at both ends of each leg:

```
phase  = frame / (total_frames - 1)
gate   = clamp(4 * leg_t * (1 - leg_t), 0, 1)
centre = clamp(bin_f + sin(2*pi * 7 * phase) * 0.16 * gate, active_start, active_end)
```

### 10.4 Spectral blend and composition per frame

Each frame mixes three normalized Gaussians over bin index. The main lobe
sits at the centre bin with standard deviation `SWEEP_GAUSSIAN_SIGMA`, widened
by up to 38% at the red-end turnaround. Two afterglow lobes trail behind the
moving centre: at 2.15 bins with sigma x1.25 and strength 0.55, and at 4.30
bins with sigma x1.65 and strength 0.24. Each bin image is sampled bilinearly
with a small per-wavelength prism displacement of (bin/63 − 0.5) × prism_scale ×
gain px along a hue-rotating axis (plus a 0.32 × (nx, ny) radial term), where
prism_scale = `SWEEP_PRISM_DISPLACEMENT_PX` (4.8) × clamp(sqrt(short_edge/1080),
0.65, 2.2) and gain = 1 + 0.85·radius + 0.55·min(luma, 1.5) + 0.55·flare. The
displacement therefore grows toward the edges, in bright areas and at the
turnaround.

Beneath the moving spectrum, every frame adds the full-spectrum composite
(the mean of all 64 bin images, x0.32), a soft blurred halo of that composite
(x1.65, radius 14 px at 1080), and a wavelength-tinted atmosphere. Pixels
are lifted to a minimum luminance of 0.022, so no frame collapses to black.
A turnaround flare, a Gaussian in time of width 0.055 centred on the midpoint,
brightens the halo and atmosphere as the sweep reverses at red.

The result goes through a colour grade (vignette 0.35, softness 2.6,
vibrance 1.08, no clarity, tone curve or tints). The 0.022 luminance floor
is enforced again, and the frame is converted to Display P3 and quantized
to 16-bit. Gaussian bloom is not used.

### 10.5 Video parameters

| Parameter | Value | Source constant |
|-----------|-------|------------------|
| Duration | 10.0 s | `CYCLE_DURATION_SECONDS` |
| FPS | 60 | `DEFAULT_VIDEO_FPS` |
| Total frames | 600 | `CYCLE_TOTAL_FRAMES` |
| Gaussian sigma (bins) | 0.42 | `SWEEP_GAUSSIAN_SIGMA` |
| Pixel format | rgb48le input | same as main video path |
| Codec | web: H.264 (libx264, CRF 18, yuv420p); HQ: HEVC (libx265, CRF 17, yuv422p10le), or libx264 CRF 21 yuv420p10le with `--fast-encode` | same pair as the main video |

### 10.6 Encoding

Frames are written as raw `rgb48le` bytes via
`create_video_groups_from_frames`, in two streams with the same profiles as the
main trajectory video: the sRGB conversion of each frame (Section 6.4) to the web
H.264 copy, and the Display P3 frames to the HQ HEVC copy (libx264 10-bit under
`--fast-encode`).

---

## 11. Constants Reference

### Spectral Parameters

| Constant | Value | Description |
|----------|-------|-------------|
| `NUM_BINS` | 64 | Number of wavelength bins in the SPD |
| `LAMBDA_START` | 380.0 nm | Start of visible spectrum |
| `LAMBDA_END` | 700.0 nm | End of visible spectrum |
| `LAMBDA_RANGE` | 320.0 nm | Total spectral range |
| `BIN_WIDTH` | 5.0 nm | Width of each bin |

### Rendering Parameters

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_HDR_SCALE` | 1.0 | `RenderConfig::default()` HDR scale; production overrides it with the seed's `hdr_scale`, uniform in [0.225, 0.295] |
| `VELOCITY_HDR_BOOST_FACTOR` | 8.0 | Maximum velocity brightness multiplier |
| `VELOCITY_NORM_LOW_QUANTILE` | 0.15 | Orbit speed quantile mapped to "slow" |
| `VELOCITY_NORM_HIGH_QUANTILE` | 0.97 | Orbit speed quantile mapped to "fast" |
| `VELOCITY_FLARE_GAMMA` | 1.35 | Flare response exponent |
| `VELOCITY_THICKNESS_SLOW` / `FAST` | 1.90 / 0.72 | Width multipliers across the speed range (~2.6x calligraphic swell) |
| `LIGHTNESS_ENERGY_FLOOR` / `SPAN` / `GAMMA` | 0.30 / 1.10 / 1.6 | OKLab lightness to deposited-energy response |
| `CRISP_DISPERSION_STRENGTH` | 0.0 | Production chromatic dispersion strength (disabled) |
| `SPECTRAL_DISPERSION_STRENGTH` | 0.0 | Base chromatic aberration strength in crisp mode |
| `SPECTRAL_DISPERSION_STRENGTH_BOOSTED` | 0.0 | Boosted chromatic aberration in crisp mode |
| `CRISP_LINE_FALLOFF_EXPONENT` | 2.0 | Super-Gaussian line profile `exp(-2 (d/t)^4)` |
| `CRISP_SPECTRAL_SIGMA_MIN_BINS` / `MAX_BINS` | 0.45 / 1.35 | Width range of the Gaussian spectral lobes deposited per colour |
| `CRISP_SPECTRAL_KERNEL_RADIUS_BINS` | 3 | Max lobe radius in bins (`min(ceil(2 sigma), 3)`) |
| `DEFAULT_TONEMAP_PAPER_WHITE` / `HIGHLIGHT_ROLLOFF` | 0.92 / 2.25 | Luminance shoulder applied after the AgX curve |
| `DEFAULT_HISTOGRAM_SAMPLE_FRAMES` | 240 | Frames sampled by the Pass 1 histogram |

### Video Parameters

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_VIDEO_FPS` | 60 | Frames per second |
| `DEFAULT_TARGET_FRAMES` | 1800 | Target frame count (30 seconds) |
| `DEFAULT_DT` | 0.001 | Simulation timestep |
| `CYCLE_DURATION_SECONDS` | 10.0 | Spectral sweep video duration |
| `CYCLE_TOTAL_FRAMES` | 600 | Spectral sweep frame count (`duration * fps`) |
| `SWEEP_BIN_START` / `SWEEP_BIN_END` | 4 / 59 | Fallback active bin range when energy detection finds nothing |
| `SWEEP_GAUSSIAN_SIGMA` | 0.42 | Narrow bin-domain Gaussian width for sweep frame blending |
| `DISPLAY_GAMMA` | 2.2 | Gamma for spectral gallery/bin images |
| `SWEEP_AFTERGLOW_STRENGTH` / `SECONDARY_STRENGTH` / `OFFSET_BINS` | 0.55 / 0.24 / 2.15 | Trailing spectral echoes behind the sweep centre |
| `SWEEP_CENTER_VIBRATO_BINS` / `CYCLES` | 0.16 / 7 | Wavelength-centre vibrato |
| `SWEEP_PRISM_DISPLACEMENT_PX` | 4.8 | Prism displacement scale at a 1080 px short edge (edge bins shift by up to half of it, times gain) |
| `SWEEP_RADIAL_PRISM_BURST` / `SWEEP_LUMINANCE_DISPERSION` | 0.85 / 0.55 | Prism displacement gain from frame radius and source luminance |
| `SWEEP_AMBIENT_COMPOSITE_STRENGTH` / `HALO_STRENGTH` / `HALO_RADIUS_PX` | 0.32 / 1.65 / 14 | Full-spectrum structure and halo beneath every sweep frame |
| `SWEEP_BACKGROUND_LUMINANCE_FLOOR` / `SWEEP_BACKGROUND_AURA_STRENGTH` | 0.018 / 0.095 | Atmosphere mask = floor + aura × field |
| `SWEEP_MIN_FRAME_LUMINANCE` | 0.022 | Luminance floor before and after grading |
| `SWEEP_TURNAROUND_FLARE_STRENGTH` / `WIDTH` | 0.85 / 0.055 | Flare at the red-end turnaround |
| `SWEEP_VIGNETTE_STRENGTH` / `SOFTNESS` / `SWEEP_VIBRANCE` | 0.35 / 2.6 / 1.08 | Sweep colour grade |

### Line Splatting Parameters

| Constant/Expression | Value | Description |
|---------------------|-------|-------------|
| Reference short edge | 2234 | Default-size anchor for crisp line scaling |
| Resolution scale | `clamp(min_dim / 2234, 0.55, 32.0)` | Multiplier for crisp line thickness |
| Base thickness | `0.95 * scale * thickness_factor * proximity` | Line width before clamping; `thickness_factor` = velocity thickness × width-pulsed line weight for line strokes; dot/pearl thickness × line weight for stipple; the dot size for stardust |
| Proximity response | `1 / (0.55 + 0.008 * len_3d / scale)` | Short segments draw bold, long spans fine (`CRISP_LINE_PROXIMITY_OFFSET` / `SLOPE`) |
| Thickness range | `[0.30, 7.00] * scale` | Clamped dynamic thickness |
| Interpolation start | motion > 1 px and (scale >= 1.5 or motion >= 12 px) | When render-time substeps activate (web/duet and spokes only) |
| Interpolation target motion | `2.5 / clamp(sqrt(scale), 1, 2)` px at scale >= 1.5, else 2.5 px | Desired max body motion per substep; substeps = `ceil(motion / target)` |
| Max interpolation substeps | 8 | Upper bound per simulation interval |
| Line subpixel coverage grid | 2x2 | Production per-pixel line coverage samples |
| CoC factor | 0.0 | Circle of confusion disabled in crisp mode |
| Bounding box pad | `ceil(effective_thickness * 1.5) + 1` | Pixel padding around segment (the profile falls below the energy cutoff at ~1.29x thickness) |
| Energy cutoff | 0.004 | Minimum averaged super-Gaussian coverage to deposit |
| Depth fade rate | 0.0007 | Exponential depth separation coefficient |
| Depth fade range | [0.18, 1.0] | Clamped depth visibility range |
