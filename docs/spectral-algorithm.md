# Spectral Image and Video Generation Algorithm

This document describes the complete algorithm for generating spectral images and
spectral videos from a finished three-body simulation. It assumes the Borda search
has already selected an optimal trajectory and the simulation has been re-run at
full resolution.

---

## Table of Contents

1. [Inputs](#1-inputs)
2. [The SPD Buffer](#2-the-spd-buffer)
3. [Spectral Accumulation (Triangle Rasterization)](#3-spectral-accumulation)
4. [Energy-Density Redshift (Removed)](#4-energy-density-redshift-removed)
5. [SPD-to-RGBA Conversion](#5-spd-to-rgba-conversion)
6. [Post-Processing Pipeline](#6-post-processing-pipeline)
7. [Still Image Output](#7-still-image-output)
8. [Main Video Output](#8-main-video-output)
9. [Spectral Gallery (Per-Bin Images)](#9-spectral-gallery)
10. [Spectral Sweep Video](#10-spectral-sweep-video)
11. [Constants Reference](#11-constants-reference)

---

## 1. Inputs

After the Borda search selects the best initial conditions and the winning orbit
is re-simulated at full resolution, the rendering pipeline receives:

| Input | Type | Description |
|-------|------|-------------|
| `positions` | `[Vec<Vector3<f64>>; 3]` | 3 bodies, each with N timesteps of (x, y, z) |
| `colors` | `[Vec<(f64, f64, f64)>; 3]` | Per-body OkLab (L, a, b) color at each timestep |
| `body_alphas` | `[f64; 3]` | Per-body opacity (constant across time) |
| `width`, `height` | `u32` | Output resolution (default 3456x2234) |
| `hdr_scale` | `f64` | Global energy multiplier (typically 3.0) |
| `effect_config` | struct | Post-processing effect parameters |

The positions are in world-space coordinates. A `RenderContext` maps world (x, y)
to pixel coordinates via a bounding box computed from all positions across all
timesteps, with 0.5 units of padding on each side.

---

## 2. The SPD Buffer

The core data structure is a **per-pixel spectral power distribution (SPD)** --
an array of energy values across wavelength bins.

### Layout

```
accum_spd: Vec<[f64; 64]>   // one [f64; 64] per pixel, row-major order
                              // total size: width * height * 64 * 8 bytes
                              // at 1080p: ~1.0 GiB (1920 * 1080 * 64 * 8)
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

This yields a fractional bin index. Energy is split between the two neighboring
integer bins using linear interpolation (see Section 3).

---

## 3. Spectral Accumulation

This is the core rendering step: for every simulation timestep, rasterize the
triangle formed by the three bodies into the SPD buffer.

### 3.0 Procedural OkLab Palette Synthesis

Before rasterization, each body receives an OKLCh color sequence. The generator
is deterministic for a seed and is built from a few continuous, uniformly drawn
knobs - no modes, scoring, or rejection sampling. A uniform `anchor` hue sets the
palette center, and a `spread` scalar scales two independently drawn inter-body
hue gaps. Because the gaps are randomized rather than pinned to a fixed angle,
palettes range continuously and without bias from near-monochrome through
analogous, split, complementary, and widely separated relationships. No fixed
angular structure (such as a 120-degree triad) and no warm/cool axis is
privileged.

```
anchor = rng * 360
spread = rng
gap1   = lerp(MIN_BODY_HUE_GAP_DEG, MAX_BODY_HUE_GAP_DEG, rng) * spread
gap2   = lerp(MIN_BODY_HUE_GAP_DEG, MAX_BODY_HUE_GAP_DEG, rng) * spread
base_hues = [anchor, anchor + gap1, anchor + gap1 + gap2]
```

The resolved `palette_phase` from the `CosmicSignature` visual profile modulates
the per-trail hue rhythm and secondary accent wave:

```
hue = base_hue
    + logarithmic_drift
    + primary_wave * HUE_WAVE_AMPLITUDE
    + accent_wave * hue_accent_strength
```

Chroma and lightness are wave-modulated and then clamped to curated bounds, and
each body keeps a distinct lightness/chroma rank. Beauty is guaranteed by
construction via that hierarchy plus Display-P3 gamut-relative chroma, so even
tight or unusual hue relationships stay tasteful while seed-to-seed variety
stays high.

### 3.1 Overview

```
for step in 0..total_steps:
    form triangle from positions[0][step], positions[1][step], positions[2][step]
    if motion interpolation is active:
        draw interpolated triangle samples toward step + 1 with compensated energy
    for each of 3 edges (0-1, 1-2, 2-0):
        compute velocity HDR multiplier for this edge
        rasterize edge as anti-aliased spectral line segment into SPD buffer
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

At calculator construction, body speeds are sampled (strided, deterministic)
across the whole trajectory and sorted. The low/high quantiles define the
normalization window:

```
v_low  = quantile(VELOCITY_NORM_LOW_QUANTILE)    // 0.15
v_high = quantile(VELOCITY_NORM_HIGH_QUANTILE)   // 0.97
norm(v) = clamp((v - v_low) / (v_high - v_low), 0, 1)
```

Per segment (edge uses the mean of its endpoint body norms; ribbons and spokes
use the body's own norm):

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
| `VELOCITY_THICKNESS_SLOW` / `FAST` | 1.30 / 0.62 |
| `dt` (simulation timestep) | 0.001 |

Slow apoapsis arcs render bold and quiet near 1x energy; periapsis whips render
as thin flares approaching 8x. Because the window adapts per orbit, every seed
exhibits the full dynamic range.

### 3.3b Scene Traits (Structure Mode, Line Weight, Age Ramp)

The `CosmicSignature` profile resolves seed-varying scene traits consumed by
the accumulator:

- **Structure mode** (seeded weighted choice): `triangle_web` (40%),
  `orbit_ribbons` (20%, each body paints its own trajectory),
  `web_ribbon_hybrid` (16%, faint web + full ribbons), `duet` (14%, one edge
  omitted), `spokes` (10%, body-to-centroid lines).
- **Line weight** in [0.85, 1.45]: global stroke width multiplier.
- **Age ramp** in [-0.35, 0.35]: linear exposure ramp across simulation time,
  encoding the arrow of time into the accumulated image.
- **Exposure key** in [0.85, 1.12]: multiplies the histogram-derived exposure
  for darker/ember or brighter/airier seeds.
- **Halation** (~30% of seeds): subtle tight DoG bloom (strength 0.05-0.14)
  as a film-style highlight halo; all other legacy effects stay disabled.

A seeded uniform 3D rotation (Shoemake quaternion method, forked RNG domain
`cosmic-view/v1`) is applied to the trajectory before rendering, so the same
orbit family is photographed from a different angle every seed.

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

The hue-to-wavelength map is piecewise linear, with a smoothed violet-to-red
bridge at the hue-wheel wrap. Hue values in the wrap region emit a dual lobe
near violet and deep red instead of jumping through the whole spectrum.

| Hue Range (degrees) | Color Region | Wavelength Range (nm) |
|---------------------|-------------|----------------------|
| 0 - 30 | Red to red-orange | 700 - 650 |
| 30 - 60 | Red-orange to orange | 650 - 620 |
| 60 - 90 | Orange to yellow | 620 - 570 |
| 90 - 150 | Yellow to green | 570 - 510 |
| 150 - 210 | Green to cyan | 510 - 485 |
| 210 - 270 | Cyan to blue | 485 - 450 |
| 270 - 330 | Blue to violet | 450 - 405 |
| 330 - 360 | Violet/red wrap | dual lobe: 405 and 690 |

For each lobe, energy is spread across nearby bins and normalized:
```
center_bin = wavelength_to_bin(center_wavelength)
for bin near center_bin:
    weight = exp(-distance_bins^2 / (2 * sigma_bins^2))
normalize(weights)
```

### 3.5 High-Resolution Triangle Interpolation

For normal projected motion, the renderer draws one triangle sample per
simulation step. It inserts extra interpolated triangle samples only for
high-resolution outputs or unusually large default-resolution jumps. This keeps
runtime and accumulated energy stable while still helping extreme renders.

```
resolution_scale = clamp(min(width, height) / 2234, 0.55, 32.0)
max_motion_px = max distance any body moves between step and step + 1

if max_motion_px <= 1.0:
    substeps = 1
else if resolution_scale < 1.5 and max_motion_px < 12.0:
    substeps = 1
else:
    substeps = clamp(ceil(max_motion_px / 2.5), 1, 8)

for substep in 0..substeps:
    if substep == 0:
        sample = exact current-step triangle
    else:
        t = substep / substeps
        sample = lerp(triangle[step], triangle[step + 1], t)
    sample_hdr_scale = hdr_scale / substeps
    draw sample triangle
```

The `1 / substeps` energy compensation is important: interpolation increases
spatial sampling density, not exposure. The first substep preserves the exact
simulation knot, so low-motion intervals do not smear to their midpoint.
Checkpointed video frames do not interpolate beyond their current checkpoint,
so frames do not leak future motion.

### 3.6 Crisp SDF Line Segment Splatting

Each edge is rasterized as an anti-aliased line segment with a steep
super-Gaussian falloff. This is the innermost loop and deposits energy into the
SPD buffer. Production CosmicSignature output intentionally avoids depth blur,
bloom haze, or broad line halos.

**Setup per segment:**

Given start vertex `v0` and end vertex `v1`:

```
dx = v1.x - v0.x
dy = v1.y - v0.y
dz = v1.z - v0.z
len_sq = dx*dx + dy*dy           // 2D length squared (pixel space)
len_3d = sqrt(dx*dx + dy*dy + dz*dz)

// Dynamic line width: faster segments are thinner, but all crisp thickness
// constants scale with the output short edge.
resolution_scale = clamp(min(width, height) / 2234, 0.55, 32.0)
normalized_len_3d = len_3d / resolution_scale
base_thickness = 0.82 * resolution_scale
min_thickness = 0.30 * resolution_scale
max_thickness = 1.55 * resolution_scale
thickness = clamp(base_thickness / (0.1 + normalized_len_3d * 0.5),
                  min_thickness,
                  max_thickness)

// Crisp production mode disables depth-of-field broadening
avg_z = (v0.z + v1.z) * 0.5
coc = |avg_z * 0.0|
effective_thickness = thickness + coc

// Bounding box padding
pad = ceil(effective_thickness * 3.0)
```

**Spectral kernels for endpoints:**

```
kernel0 = spectral_kernel_for_oklab(v0.color)
kernel1 = spectral_kernel_for_oklab(v1.color)
```

**Energy conservation and depth fade:**

```
energy_conservation = thickness / effective_thickness
depth_fade = clamp(exp(-|avg_z| * 0.002), 0.05, 1.0)
base_energy_mult = hdr_scale * edge_hdr_multiplier * depth_fade * energy_conservation
```

**Per-pixel loop** (over all pixels in the padded bounding box):

```
for py in min_y..=max_y:
    for px in min_x..=max_x:
        energy_sum = 0
        start_energy_sum = 0
        end_energy_sum = 0

        for each 2x2 subpixel sample:
            pax = sample_x - v0.x
            pay = sample_y - v0.y
            h = project sample onto segment, clamped to [0, 1]
            dist_sq = squared distance from sample to segment
            normalized = dist_sq / (effective_thickness * effective_thickness)
            energy = exp(-(normalized * normalized) * 2.0)
            alpha = v0.alpha * (1 - h) + v1.alpha * h

            weighted_energy = energy * alpha
            energy_sum += weighted_energy
            start_energy_sum += weighted_energy * (1 - h)
            end_energy_sum += weighted_energy * h

        coverage = energy_sum / 4
        if coverage < 0.004: continue

        for (bin, weight) in kernel0:
            accum_spd[pixel_index][bin] += base_energy_mult * start_energy_sum * weight / 4
        for (bin, weight) in kernel1:
            accum_spd[pixel_index][bin] += base_energy_mult * end_energy_sum * weight / 4
```

The production path keeps this fixed 2x2 coverage grid. Higher-order AA can be
revisited later behind an explicit quality mode, but it is not part of the
default crisp profile.

### 3.7 Parallelization

The accumulation supports two parallelization strategies:

**Scanline Bands:** The image is divided into horizontal bands (one per CPU
thread). Each band processes all simulation steps but only writes to pixels
within its row range. No synchronization is needed since bands don't overlap.

**Step Chunking:** When the step count is large (>= a threshold) and multiple
threads are available, the step range is divided into chunks. Each chunk
accumulates into its own full-resolution SPD buffer. After all chunks complete,
the partial buffers are merged by element-wise addition into the main buffer.

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
RGBA. It is a fused operation that combines three sub-steps in a single pass
over the buffer.

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
dispersion_strength = 0.0   // CRISP_DISPERSION_STRENGTH
```

**Per pixel:**

```
dx = pixel_x - cx
dy = pixel_y - cy
r = sqrt(dx*dx + dy*dy)
dir_x = dx / r    // radial direction (unit vector)
dir_y = dy / r
r_norm = r / max_r  // normalized distance from center [0, 1]

local_spd = [0.0; 64]

for bin in 0..64:
    // Bin offset: normalized position within the spectrum, centered at 0
    bin_offset = (bin - 31.5) / 31.5     // range [-1, +1]

    // Spatial shift: blue bins shift outward, red bins shift inward
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

A 64-entry LUT is precomputed at startup. Each entry stores `(X, Y, Z, k)`:

- **(X, Y, Z):** Simpson-integrated CIE 1931 color matching values across the
  5 nm bin span. The table is normalized so a flat SPD reads as D65 white.
- **k (tone steepness):** A smooth per-bin compression value retuned for CIE
  energy. It gently decreases from violet/blue toward red to keep highlights
  colorful without letting any one spectral edge dominate.

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
(L, a, b) = linear_rec2020_to_oklab(xyz_to_linear_rec2020(X, Y, Z))
C = sqrt(a*a + b*b)
boost = 1 + vibrance_amount * knee / (C + knee)
(R, G, B) = oklab_to_linear_rec2020(L, a * boost, b * boost)
(R, G, B) = preserve_hue_gamut_map(R, G, B)

// Brightness from total accumulated energy
brightness = 1.0 - exp(-total)

// Premultiplied alpha output
output = (R * brightness, G * brightness, B * brightness, brightness)
```

The output is a premultiplied-alpha linear Rec.2020 tuple in [0, 1]. The alpha
channel (brightness) represents how much light the pixel received overall.
During final quantization, Rec.2020 display values are converted to Display P3
and written as 16-bit PNG/video frames with explicit P3 metadata.

---

## 6. CosmicSignature Finish Pipeline

After SPD-to-RGBA conversion produces a linear RGBA buffer, the default
CosmicSignature profile keeps the finish path intentionally minimal. Legacy
post-effects still exist as optional modules, but the production profile leaves
them disabled so the final image is driven by spectral geometry, transparent
overlap, thin luminous edges, clean tonemapping, and black negative space.

### 6.1 Trajectory Effects (`process_trajectory`)

Default production chain: empty. The trajectory buffer is passed through
unchanged after spectral conversion.

### 6.2 Tone Mapping

A custom tone mapper uses histogram-derived exposure levels:

1. **Pass 1 (histogram):** Sample a subset of frames, convert SPD to RGBA,
   run trajectory effects, then collect R/G/B histograms to determine
   `ChannelLevels` (black point, white point, gamma per channel).

2. **Apply tone map:** Per-pixel, scale by exposure, apply per-channel
   levels, compress highlights with a shoulder curve.

### 6.3 Image Effects (`process_image`)

Default production chain: empty. The display buffer is quantized without a grain
or texture overlay.

### 6.4 Display P3 Quantization

The final linear Rec.2020 RGBA buffer is converted to linear Display P3 and
then quantized to 16-bit RGB. PNG outputs include Display P3 chromaticities and
cICP metadata (`color_primaries = 12`, `transfer_function = 13`,
`matrix_coefficients = 0`). Video encodes use Display P3 primaries in FFmpeg
metadata.

```
for each pixel:
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
4. process_trajectory(rgba_buffer)         -> unchanged RGBA by default
5. tonemap(rgba_buffer, channel_levels)    -> display-space RGBA
6. process_image(rgba_buffer)              -> unchanged display RGBA by default
7. quantize to 16-bit sRGB
8. Save as PNG
```

For very large stills, the final accumulated frame can switch to a striped
renderer. The striped path renders row bands with guard rows, quantizes each
band to Display P3, and stitches the final 16-bit image without allocating a
full-frame `width * height * 64 * f64` SPD buffer. Guard rows scale with the
maximum crisp footprint at the output resolution so cross-tile splats are not
clipped. The generator logs a
full-frame SPD memory estimate before rendering so extreme-resolution jobs are
visible up front.

---

## 8. Main Video Output

The main video progressively reveals the orbit over time. Each frame shows the
accumulated trajectory up to a specific simulation checkpoint.

### Frame Scheduling

```
total_steps   = number of simulation timesteps
target_frames = 1800          // 30 seconds at 60 fps
fps           = 60
frame_interval = max(total_steps / target_frames, 1)

// Checkpoints: evenly spaced simulation steps
checkpoints = [frame_interval, 2*frame_interval, ..., total_steps]
```

### Per-Frame Pipeline

For each checkpoint (frame) in sequence:

```
1. Accumulate new steps into accum_spd:
   accumulate_spectral_steps(accum_spd, step_start, checkpoint)
   (step_start = previous checkpoint, so accumulation is incremental)

2. Convert: convert_spd_buffer_to_rgba(accum_spd) -> RGBA

3. Trajectory finish: process_trajectory(RGBA), unchanged by default

4. Tone map using pre-computed ChannelLevels from Pass 1

5. Temporal smoothing (optional, default off in production):
   display[i] = display[i] * 0.10 + previous_display[i] * 0.90
   (blends 10% new frame with 90% previous for smooth transitions)

6. Image finish: process_image(display), unchanged by default

7. Quantize to 16-bit: rgb48le format (6 bytes per pixel)

8. Write raw frame bytes to FFmpeg stdin
```

### Video Encoding

Frames are streamed to FFmpeg via stdin as raw `rgb48le` data (16-bit RGB,
little-endian, no alpha). FFmpeg encodes to H.265/HEVC:

| Parameter | Default Value |
|-----------|---------------|
| Codec | libx265 |
| CRF | 17 |
| Pixel format | yuv422p10le (10-bit 4:2:2) |
| Preset | slower |
| Input format | rgb48le |
| FPS | 60 |

A bounded channel (capacity 32 frames) buffers frames between the renderer
and the FFmpeg writer thread to keep CPU cores busy while the encoder drains.

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

        // Tint by wavelength color and apply display gamma
        R = (normalized * tint_r) ^ (1/2.2)
        G = (normalized * tint_g) ^ (1/2.2)
        B = (normalized * tint_b) ^ (1/2.2)

        // Quantize to 16-bit
        pixel_out = (round(R * 65535), round(G * 65535), round(B * 65535))

    save as "{bin:02}_{wavelength:.0}nm.png"
```

### Wavelength-to-RGB (Dan Bruton's formula)

Used for tinting bin images. Returns linear sRGB for a given wavelength:

| Wavelength Range | R | G | B |
|-----------------|---|---|---|
| 380-440 nm | -(lambda-440)/(440-380) | 0 | 1 |
| 440-490 nm | 0 | (lambda-440)/(490-440) | 1 |
| 490-510 nm | 0 | 1 | -(lambda-510)/(510-490) |
| 510-580 nm | (lambda-510)/(580-510) | 1 | 0 |
| 580-645 nm | 1 | -(lambda-645)/(645-580) | 0 |
| 645-700 nm | 1 | 0 | 0 |

An intensity falloff factor is applied near the edges of the visible range:

```
if 380 <= lambda < 420:  factor = 0.3 + 0.7 * (lambda - 380) / 40
if 420 <= lambda < 645:  factor = 1.0
if 645 <= lambda <= 700: factor = 0.3 + 0.7 * (700 - lambda) / 55
```

Final: `(R * factor, G * factor, B * factor)`

---

## 10. Spectral Sweep Video

After the main still image and gallery, the pipeline can encode **one**
spectral sweep video (`spectral_sweep.mp4`). It animates a smooth sweep through
wavelength bins using precomputed bin images, **Gaussian blending** across bins
(not simple two-bin linear interpolation), **cosine easing** over time, and a
dynamic **active bin range** so mostly-empty bins at the spectrum edges can be
skipped.

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

### 10.3 Time easing and centre bin

Let `total_frames` be `CYCLE_TOTAL_FRAMES` (1440), `frame` in `0..total_frames`,
and `leg_t` be the normalized position within the current outbound or return
leg. The first leg sweeps from violet to red; the second sweeps back from red
to violet, so the first and last frames share the same centre bin.

Cosine easing (slow at both ends of each leg):

```
t_eased = (1.0 - cos(leg_t * pi)) * 0.5
```

Fractional centre bin (within the active range):

```
bin_f = active_start + t_eased * (active_end - active_start)
```

### 10.4 Gaussian blend per frame

For each frame, every pixel gets a weighted sum of nearby bin images. Weights
are a **normalized Gaussian** over bin index centred at `bin_f` with standard
deviation `SWEEP_GAUSSIAN_SIGMA` (bins within about `3 * sigma` of the centre
contribute). This produces smooth transitions between wavelength-dominated looks
without hard banding.

The blended linear RGB is then run through a crisp colour-grade pass. Gaussian
bloom is disabled in production sweep output (`SWEEP_BLOOM_RADIUS = 0`,
`SWEEP_BLOOM_STRENGTH = 0.0`) so auxiliary outputs do not introduce haze.

### 10.5 Video parameters

| Parameter | Value | Source constant |
|-----------|-------|------------------|
| Duration | 24.0 s | `CYCLE_DURATION_SECONDS` |
| FPS | 60 | `DEFAULT_VIDEO_FPS` |
| Total frames | 1440 | `CYCLE_TOTAL_FRAMES` |
| Gaussian sigma (bins) | 0.55 | `SWEEP_GAUSSIAN_SIGMA` |
| Pixel format | rgb48le | same as main video path |
| Codec | HEVC (libx265) | same options as main encode / fast-encode mode |

### 10.6 Encoding

Frames are written as raw `rgb48le` bytes to FFmpeg stdin via
`create_video_from_frames_singlepass`, reusing the same encoding profile as the
main trajectory video (default quality vs `--fast-encode`).

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
| `DEFAULT_HDR_SCALE` | 1.0 | Base HDR scale (usually overridden to ~3.0) |
| `VELOCITY_HDR_BOOST_FACTOR` | 8.0 | Maximum velocity brightness multiplier |
| `VELOCITY_NORM_LOW_QUANTILE` | 0.15 | Orbit speed quantile mapped to "slow" |
| `VELOCITY_NORM_HIGH_QUANTILE` | 0.97 | Orbit speed quantile mapped to "fast" |
| `VELOCITY_FLARE_GAMMA` | 1.35 | Flare response exponent |
| `VELOCITY_THICKNESS_SLOW` / `FAST` | 1.30 / 0.62 | Width multipliers across the speed range |
| `LIGHTNESS_ENERGY_FLOOR` / `SPAN` / `GAMMA` | 0.30 / 1.10 / 1.6 | OKLab lightness to deposited-energy response |
| `CRISP_DISPERSION_STRENGTH` | 0.0 | Production chromatic dispersion strength (disabled) |
| `SPECTRAL_DISPERSION_STRENGTH` | 0.0 | Base chromatic aberration strength in crisp mode |
| `SPECTRAL_DISPERSION_STRENGTH_BOOSTED` | 0.0 | Boosted chromatic aberration in crisp mode |

### Video Parameters

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_VIDEO_FPS` | 60 | Frames per second |
| `DEFAULT_TARGET_FRAMES` | 1800 | Target frame count (30 seconds) |
| `DEFAULT_DT` | 0.001 | Simulation timestep |
| `CYCLE_DURATION_SECONDS` | 24.0 | Spectral sweep video duration |
| `CYCLE_TOTAL_FRAMES` | 1440 | Spectral sweep frame count (`duration * fps`) |
| `SWEEP_BIN_START` / `SWEEP_BIN_END` | 4 / 59 | Fallback active bin range when energy detection finds nothing |
| `SWEEP_GAUSSIAN_SIGMA` | 0.55 | Narrow bin-domain Gaussian width for sweep frame blending |
| `DISPLAY_GAMMA` | 2.2 | Gamma for spectral gallery/bin images |

### Line Splatting Parameters

| Constant/Expression | Value | Description |
|---------------------|-------|-------------|
| Reference short edge | 2234 | Default-size anchor for crisp line scaling |
| Resolution scale | `clamp(min_dim / 2234, 0.55, 32.0)` | Multiplier for crisp line thickness |
| Base thickness | `0.82 * scale` | Starting line width in pixels |
| Thickness range | `[0.30, 1.55] * scale` | Clamped dynamic thickness |
| Interpolation start | scale >= 1.5 or motion >= 12 px | When render-time substeps activate |
| Interpolation target motion | 2.5 px | Desired max body motion per render-time substep |
| Max interpolation substeps | 8 | Upper bound per simulation interval |
| Line subpixel coverage grid | 2x2 | Production per-pixel line coverage samples |
| CoC factor | 0.0 | Circle of confusion disabled in crisp mode |
| Bounding box pad | `ceil(effective_thickness * 3.0)` | Pixel padding around segment |
| Energy cutoff | 0.004 | Minimum averaged super-Gaussian coverage to deposit |
| Depth fade rate | 0.0007 | Exponential depth separation coefficient |
| Depth fade range | [0.18, 1.0] | Clamped depth visibility range |
