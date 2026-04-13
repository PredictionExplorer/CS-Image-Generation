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
4. [Energy-Density Redshift](#4-energy-density-redshift)
5. [SPD-to-RGBA Conversion](#5-spd-to-rgba-conversion)
6. [Post-Processing Pipeline](#6-post-processing-pipeline)
7. [Still Image Output](#7-still-image-output)
8. [Main Video Output](#8-main-video-output)
9. [Spectral Gallery (Per-Bin Images)](#9-spectral-gallery)
10. [Spectral Cycle Videos (6 Variants)](#10-spectral-cycle-videos)
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
| `width`, `height` | `u32` | Output resolution (default 1920x1080) |
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
accum_spd: Vec<[f32; 64]>   // one [f32; 64] per pixel, row-major order
                              // total size: width * height * 64 * 4 bytes
                              // at 1080p: ~500 MB
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

### 3.1 Overview

```
for step in 0..total_steps:
    form triangle from positions[0][step], positions[1][step], positions[2][step]
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

### 3.3 Velocity HDR Multiplier

Each triangle edge receives a brightness boost proportional to how fast the
bodies at its endpoints are moving. This makes fast-moving regions flare
dramatically.

For a single body at timestep `step`:

```
velocity = |positions[body][step+1] - positions[body][step]| / dt
normalized_velocity = min(velocity / VELOCITY_HDR_BOOST_THRESHOLD, 1.0)
multiplier = 1.0 + normalized_velocity * (VELOCITY_HDR_BOOST_FACTOR - 1.0)
```

For an edge between body A and body B:

```
edge_multiplier = (multiplier_A + multiplier_B) / 2.0
```

| Constant | Value |
|----------|-------|
| `VELOCITY_HDR_BOOST_FACTOR` | 8.0 |
| `VELOCITY_HDR_BOOST_THRESHOLD` | 0.15 |
| `dt` (simulation timestep) | 0.001 |

A stationary body gets multiplier 1.0. A body at or above the threshold velocity
gets multiplier 8.0.

### 3.4 OkLab Hue to Wavelength

Each vertex has an OkLab color (L, a, b). The hue determines which wavelength
bin receives energy. The mapping uses only the `a` and `b` components:

```
hue_rad = atan2(b, a)
hue_deg = hue_rad.to_degrees()
if hue_deg < 0: hue_deg += 360
```

Then a piecewise linear mapping converts hue degrees to wavelength:

| Hue Range (degrees) | Color Region | Wavelength Range (nm) |
|---------------------|-------------|----------------------|
| 0 - 30 | Red to red-orange | 700 - 650 |
| 30 - 60 | Red-orange to orange | 650 - 620 |
| 60 - 90 | Orange to yellow | 620 - 570 |
| 90 - 150 | Yellow to green | 570 - 510 |
| 150 - 210 | Green to cyan | 510 - 485 |
| 210 - 270 | Cyan to blue | 485 - 450 |
| 270 - 330 | Blue to violet | 450 - 380 |
| 330 - 360 | Violet to red (wrap) | 380 - 700 |

Each sub-range uses linear interpolation. The result is clamped to [380, 700].

Example for hue in [0, 30):
```
wavelength = 700.0 - (hue_deg / 30.0) * 50.0
```

### 3.5 Gaussian SDF Line Segment Splatting

Each edge is rasterized as an anti-aliased line segment with Gaussian falloff.
This is the innermost loop and deposits energy into the SPD buffer.

**Setup per segment:**

Given start vertex `v0` and end vertex `v1`:

```
dx = v1.x - v0.x
dy = v1.y - v0.y
dz = v1.z - v0.z
len_sq = dx*dx + dy*dy           // 2D length squared (pixel space)
len_3d = sqrt(dx*dx + dy*dy + dz*dz)

// Dynamic line width: faster segments are thinner
base_thickness = 1.2
thickness = clamp(base_thickness / (0.1 + len_3d * 0.5), 0.2, 4.0)

// Depth of field: Circle of Confusion from average z-depth
avg_z = (v0.z + v1.z) * 0.5
coc = |avg_z * 0.05|
effective_thickness = thickness + coc

// Bounding box padding
pad = ceil(effective_thickness * 2.5)
```

**Wavelength bins for endpoints:**

```
wavelength0 = oklab_hue_to_wavelength(v0.color.a, v0.color.b)
wavelength1 = oklab_hue_to_wavelength(v1.color.a, v1.color.b)
bin0_f = wavelength_to_bin(wavelength0)   // fractional bin [0, 63]
bin1_f = wavelength_to_bin(wavelength1)   // fractional bin [0, 63]
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
        // Vector from v0 to pixel
        pax = px - v0.x
        pay = py - v0.y

        // Project pixel onto segment, get parameter h in [0, 1]
        if len_sq > 1e-6:
            h = clamp((pax*dx + pay*dy) / len_sq, 0.0, 1.0)
        else:
            h = 0.5   // degenerate (zero-length) segment

        // Perpendicular distance from pixel to segment
        proj_x = pax - dx * h
        proj_y = pay - dy * h
        dist_sq = proj_x*proj_x + proj_y*proj_y

        // Gaussian falloff
        energy = exp(-dist_sq / (effective_thickness * effective_thickness))
        if energy < 0.01: continue    // skip negligible contributions

        // Interpolate alpha along segment
        alpha = v0.alpha * (1 - h) + v1.alpha * h

        // Final energy for this pixel
        final_energy = energy * alpha * base_energy_mult

        // Interpolate fractional bin along segment
        bin_f = bin0_f * (1 - h) + bin1_f * h

        // Split energy between two adjacent bins
        bin_left  = floor(bin_f) as usize, clamped to [0, 63]
        bin_right = min(bin_left + 1, 63)
        w_right   = fract(bin_f)

        // Deposit into SPD buffer
        if bin_left == bin_right:
            accum_spd[pixel_index][bin_left] += final_energy
        else:
            accum_spd[pixel_index][bin_left]  += final_energy * (1 - w_right)
            accum_spd[pixel_index][bin_right] += final_energy * w_right
```

### 3.6 Parallelization

The accumulation supports two parallelization strategies:

**Scanline Bands:** The image is divided into horizontal bands (one per CPU
thread). Each band processes all simulation steps but only writes to pixels
within its row range. No synchronization is needed since bands don't overlap.

**Step Chunking:** When the step count is large (>= a threshold) and multiple
threads are available, the step range is divided into chunks. Each chunk
accumulates into its own full-resolution SPD buffer. After all chunks complete,
the partial buffers are merged by element-wise addition into the main buffer.

---

## 4. Energy-Density Redshift

After accumulation, high-energy pixels undergo a spectral shift toward red
(longer wavelengths). This simulates a "heat" effect where intense regions
appear warmer.

### Algorithm

For each pixel (parallelized):

```
total_energy = sum(spd[0..64])

if total_energy < ENERGY_DENSITY_SHIFT_THRESHOLD:
    return    // no shift for dim pixels

excess = total_energy - ENERGY_DENSITY_SHIFT_THRESHOLD
shift_amount = min(excess * ENERGY_DENSITY_SHIFT_STRENGTH, 1.0)

// Shift spectrum toward red (higher bin indices)
shifted_spd = copy of spd
for i in (1..64).rev():
    shifted_spd[i] = spd[i] * (1 - shift_amount) + spd[i-1] * shift_amount
shifted_spd[0] = spd[0] * (1 - shift_amount)

spd = shifted_spd
```

| Constant | Value |
|----------|-------|
| `ENERGY_DENSITY_SHIFT_THRESHOLD` | 0.08 |
| `ENERGY_DENSITY_SHIFT_STRENGTH` | 0.75 |

Each bin blends with its blue-side neighbor proportional to `shift_amount`.
Bin 0 (most violet) loses energy with no replacement. The net effect pushes the
spectral distribution toward the red end.

---

## 5. SPD-to-RGBA Conversion

This stage converts the per-pixel 64-bin SPD into linear-space premultiplied
RGBA. It is a fused operation that combines three sub-steps in a single pass
over the buffer.

### 5.1 Radial Spectral Dispersion (Chromatic Aberration)

Before converting each pixel, bins are sampled from spatially shifted positions
to simulate optical dispersion (prismatic color separation radiating from the
image center).

**Setup:**

```
cx = width / 2.0
cy = height / 2.0
max_r = sqrt(cx*cx + cy*cy)
dispersion_strength = SPECTRAL_DISPERSION_STRENGTH_BOOSTED * 3.0   // default: 1.1 * 3.0 = 3.3
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

At the image center (`r = 0`), there is no shift -- the dispersion is purely
zero there. The effect increases toward the edges of the image.

### 5.2 Fused Energy-Density Redshift

The same redshift algorithm from Section 4 is applied per-pixel to `local_spd`
after dispersion sampling. This ensures the shift operates on the
dispersion-adjusted spectrum.

```
total_energy = sum(local_spd)
if total_energy >= 0.08:
    // Apply the same shift as Section 4
    ...
```

### 5.3 SPD to Linear RGBA

The final conversion maps the 64-bin spectrum to a linear-sRGB (R, G, B, A)
tuple using a precomputed lookup table.

#### The Combined LUT

A 64-entry LUT is precomputed at startup. Each entry stores `(R, G, B, k)`:

- **(R, G, B):** The linear-sRGB color of that bin's center wavelength, computed
  from Dan Bruton's spectrum-to-RGB formula. This assigns each bin a "basis
  color" -- e.g., bin 0 is deep violet, bin 32 is green, bin 63 is deep red.

- **k (tone steepness):** A per-bin tone-mapping strength that controls how
  quickly energy saturates. Blue wavelengths have higher k (compress faster),
  red wavelengths have lower k (stay linear longer):

| Wavelength Range | k Value |
|-----------------|---------|
| < 450 nm (violet) | 2.2 + 0.3 * (450 - lambda) / 70 |
| 450-490 nm (blue) | 2.0 |
| 490-550 nm (cyan-green) | 1.8 |
| 550-590 nm (green-yellow) | 1.6 |
| 590-650 nm (orange) | 1.4 - 0.2 * (lambda - 590) / 60 |
| 650-700 nm (red) | 1.2 - 0.2 * (lambda - 650) / 50 |

#### Conversion Algorithm

```
R_sum = 0, G_sum = 0, B_sum = 0, total = 0

for i in 0..64:
    e = local_spd[i]
    if e <= 1e-10: continue

    (lut_R, lut_G, lut_B, k) = BIN_COMBINED_LUT[i]

    // Per-bin tone mapping: soft saturation curve
    e_mapped = 1.0 - exp(-k * e)

    total += e_mapped
    R_sum += e_mapped * lut_R
    G_sum += e_mapped * lut_G
    B_sum += e_mapped * lut_B

if total < 1e-10:
    return (0, 0, 0, 0)    // black pixel

// Normalize to get the "chromaticity" (hue/saturation)
R = R_sum / total
G = G_sum / total
B = B_sum / total

// Saturation boost
mean = (R + G + B) / 3.0
color_range = max(R, G, B) - min(R, G, B)

if color_range < 0.1:
    sat_boost = 3.0      // low-saturation: strong boost
elif color_range < 0.3:
    sat_boost = 2.6      // medium: moderate boost
else:
    sat_boost = 2.2      // high-saturation: gentle boost

R = mean + (R - mean) * sat_boost
G = mean + (G - mean) * sat_boost
B = mean + (B - mean) * sat_boost

// Rescale if any channel exceeds 1.0
max_val = max(R, G, B)
if max_val > 1.0:
    scale = 1.0 / max_val
    R *= scale; G *= scale; B *= scale

R = clamp(R, 0, 1)
G = clamp(G, 0, 1)
B = clamp(B, 0, 1)

// Brightness from total accumulated energy
brightness = 1.0 - exp(-total)

// Premultiplied alpha output
output = (R * brightness, G * brightness, B * brightness, brightness)
```

The output is a premultiplied-alpha linear-sRGB tuple in [0, 1]. The alpha
channel (brightness) represents how much light the pixel received overall,
while the RGB channels encode the spectral color.

---

## 6. Post-Processing Pipeline

After SPD-to-RGBA conversion produces a linear RGBA buffer, the image passes
through a multi-stage post-processing pipeline.

### 6.1 Trajectory Effects (`process_trajectory`)

Applied in order:

1. **Bloom** -- Gaussian blur blended with the original for diffuse glow
2. **DoG Bloom** -- Difference-of-Gaussians for edge-detected glow
3. **Glow Enhancement** -- Tight sparkle on very bright areas
4. **Chromatic Bloom** -- Prismatic color separation in bloom
5. **Perceptual Blur** -- OkLab-space smoothing
6. **Micro-Contrast** -- Local contrast enhancement
7. **Gradient Map** -- Artistic color palette mapping
8. **Cinematic Color Grade** -- Film-like color grading with split toning
9. **Opalescence** -- Gem-like shimmer
10. **Champlevé** -- Voronoi cell structure with metallic rims
11. **Aether** -- Woven filament and volumetric scattering
12. **Edge Luminance** -- Selective edge brightening
13. **Atmospheric Depth** -- Spatial perspective and fog

### 6.2 Tone Mapping

A custom tone mapper uses histogram-derived exposure levels:

1. **Pass 1 (histogram):** Sample a subset of frames, convert SPD to RGBA,
   run trajectory effects, then collect R/G/B histograms to determine
   `ChannelLevels` (black point, white point, gamma per channel).

2. **Apply tone map:** Per-pixel, scale by exposure, apply per-channel
   levels, compress highlights with a shoulder curve.

### 6.3 Image Effects (`process_image`)

Applied after tone mapping to the composited display image:

1. **Fine Texture** -- Subtle surface grain

### 6.4 Quantization

The final linear RGBA buffer is converted to 16-bit sRGB:

```
for each pixel:
    // Apply sRGB transfer function (gamma ~2.2)
    // Quantize to [0, 65535]
    output_u16 = round(srgb_transfer(linear_value) * 65535)
```

---

## 7. Still Image Output

To produce a single spectral image (16-bit PNG):

```
1. Initialize accum_spd to zeros: Vec<[f32; 64]> of size width * height
2. Accumulate all simulation steps (0..total_steps) into accum_spd
3. convert_spd_buffer_to_rgba(accum_spd) -> linear RGBA buffer
4. process_trajectory(rgba_buffer)         -> post-processed RGBA
5. tonemap(rgba_buffer, channel_levels)    -> display-space RGBA
6. process_image(rgba_buffer)              -> final RGBA
7. quantize to 16-bit sRGB
8. Save as PNG
```

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

3. Trajectory effects: process_trajectory(RGBA)

4. Tone map using pre-computed ChannelLevels from Pass 1

5. Temporal smoothing (optional):
   display[i] = display[i] * 0.10 + previous_display[i] * 0.90
   (blends 10% new frame with 90% previous for smooth transitions)

6. Image effects: process_image(display)

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
bin, plus a dominant-wavelength heatmap. These reveal which parts of the image
contain energy at each wavelength.

### Per-Bin Image Algorithm

After full accumulation and energy-density redshift:

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

### Dominant-Wavelength Heatmap

An additional image where each pixel is colored by its dominant (highest-energy)
bin, with brightness scaled logarithmically:

```
max_total = max(sum(spd) for all pixels)
log_denom = ln(1 + max_total)

for each pixel:
    dominant_bin = argmax(spd[0..64])
    t = dominant_bin / 63.0
    (r, g, b) = heatmap_color(t)    // violet-blue-green-red gradient
    brightness = ln(1 + sum(spd)) / log_denom

    output = ((r * brightness)^(1/2.2), (g * brightness)^(1/2.2), (b * brightness)^(1/2.2))
```

---

## 10. Spectral Cycle Videos

Six video variants animate a continuous sweep through the 64 wavelength bins.
They share a common setup and differ only in how the bin index evolves over time.

### 10.1 Shared Setup: BinBuffers

Before generating any cycle video, precompute 64 "bin images" -- one per
wavelength bin. Each bin image shows that bin's energy distribution, tinted by
its wavelength color.

```
1. Start with the fully accumulated SPD buffer (all timesteps)
2. Apply energy-density redshift

3. For each bin in 0..64 (in parallel):
    wavelength = 380.0 + (bin + 0.5) * 5.0
    (tint_r, tint_g, tint_b) = wavelength_to_rgb(wavelength)

    max_val = max(accum_spd[pixel][bin] for all pixels)
    max_val = max(max_val, 1e-10)

    for each pixel:
        normalized = clamp(accum_spd[pixel][bin] / max_val, 0, 1)
        buffer[bin][pixel] = [
            (normalized * tint_r) ^ (1/2.2),
            (normalized * tint_g) ^ (1/2.2),
            (normalized * tint_b) ^ (1/2.2),
        ]
```

This produces `BinBuffers`: 64 float RGB images stored in memory.

### 10.2 Frame Interpolation (lerp_bins)

Given a fractional bin index `bin_f`, produce a frame by linearly interpolating
between the two nearest bin images:

```
bin_f = bin_f mod 64              // wrapping (allows negative and > 64)
lo = floor(bin_f) mod 64
hi = (lo + 1) mod 64
t  = fract(bin_f)

for each pixel:
    R = buffers[lo][pixel].R * (1 - t) + buffers[hi][pixel].R * t
    G = buffers[lo][pixel].G * (1 - t) + buffers[hi][pixel].G * t
    B = buffers[lo][pixel].B * (1 - t) + buffers[hi][pixel].B * t

    // Quantize to 16-bit
    output = (clamp(R, 0, 1) * 65535, clamp(G, 0, 1) * 65535, clamp(B, 0, 1) * 65535)
```

### 10.3 Video Parameters

| Parameter | Value |
|-----------|-------|
| Duration | 12.0 seconds |
| FPS | 60 |
| Total frames | 720 |
| Pixel format | rgb48le (16-bit RGB) |
| Codec | HEVC (same as main video) |

### 10.4 The Six Variants

All variants compute a fractional bin index `bin_f` for each frame, then call
`lerp_bins(bin_f)` (or `per_pixel_bin_select` for radial) to produce the frame.

Let `frame` be the current frame index (0 to 719) and
`t = frame / total_frames` (normalized time, 0 to ~1).

---

#### Variant 1: Forward Cycle

A linear sweep from violet (bin 0) to red (bin 63):

```
bin_f = frame * 64 / 720
```

Produces a smooth, constant-speed scan through the entire spectrum.

---

#### Variant 2: Reverse Cycle

A linear sweep from red to violet:

```
bin_f = 64 - (frame * 64 / 720)
```

---

#### Variant 3: Ping-Pong Cycle

Sweeps from violet to red in the first half, then back to violet:

```
max_bin = 63.0
if t < 0.5:
    bin_f = t * 2.0 * max_bin          // 0 -> 63 in first 6 seconds
else:
    bin_f = (1.0 - (t - 0.5) * 2.0) * max_bin   // 63 -> 0 in last 6 seconds
```

---

#### Variant 4: Ease Cycle

Uses cosine easing so the sweep lingers at the violet and red extremes:

```
bin_f = (1.0 - cos(t * 2 * pi)) * 0.5 * 64
```

This is a single full cosine period, producing slow-fast-slow motion. The sweep
moves slowest near bin 0 and bin 64, fastest through the middle greens.

---

#### Variant 5: Radial Cycle

Creates concentric rings of different wavelengths radiating from the image
center. Each pixel gets its own bin index based on distance from center.

**Precompute distance map:**

```
cx = (width - 1) / 2.0
cy = (height - 1) / 2.0
max_dist = sqrt(cx*cx + cy*cy)

for each pixel (x, y):
    dist_map[pixel] = sqrt((x-cx)^2 + (y-cy)^2) / max_dist   // [0, 1]
```

**Per frame:**

```
RADIAL_SPREAD = 24.0
base_bin = frame * 64 / 720

for each pixel:
    bin_map[pixel] = base_bin - dist_map[pixel] * RADIAL_SPREAD
```

Then call `per_pixel_bin_select(bin_map)` which applies `lerp_bins` per pixel
using each pixel's individual fractional bin index (with wrapping via
`rem_euclid`).

This creates a rainbow ring pattern that sweeps outward through the spectrum.

---

#### Variant 6: Complementary Cycle

Two cursors sweep simultaneously, half a spectrum apart (32 bins), and their
results are averaged per pixel:

```
half = 32.0
bin_f = frame * 64 / 720

frame_A = lerp_bins(bin_f)          // primary cursor
frame_B = lerp_bins(bin_f + half)   // complementary cursor

for each channel value:
    output = (frame_A + frame_B) / 2     // u16 average, clamped to 65535
```

This produces color combinations from opposite sides of the spectrum
(e.g., violet + yellow-green, blue + orange).

---

### 10.5 Video Encoding

Each variant's frames are written as raw `rgb48le` bytes to FFmpeg stdin via
`create_video_from_frames_singlepass`, using the same encoding options as the
main video (HEVC, CRF 17, or fast hardware-accelerated mode). All six variants
can be encoded in parallel since they only read from the shared `BinBuffers`.

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
| `VELOCITY_HDR_BOOST_THRESHOLD` | 0.15 | Velocity at which max boost is reached |
| `ENERGY_DENSITY_SHIFT_THRESHOLD` | 0.08 | Minimum energy for redshift |
| `ENERGY_DENSITY_SHIFT_STRENGTH` | 0.75 | How strongly high energy shifts to red |
| `SPECTRAL_DISPERSION_STRENGTH` | 0.8 | Base chromatic aberration strength |
| `SPECTRAL_DISPERSION_STRENGTH_BOOSTED` | 1.1 | Boosted chromatic aberration |

### Video Parameters

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_VIDEO_FPS` | 60 | Frames per second |
| `DEFAULT_TARGET_FRAMES` | 1800 | Target frame count (30 seconds) |
| `DEFAULT_DT` | 0.001 | Simulation timestep |
| `CYCLE_DURATION_SECONDS` | 12.0 | Spectral cycle video duration |
| `RADIAL_SPREAD` | 24.0 | Bin offset range for radial variant |
| `DISPLAY_GAMMA` | 2.2 | Gamma for spectral gallery/bin images |

### Line Splatting Parameters

| Constant/Expression | Value | Description |
|---------------------|-------|-------------|
| Base thickness | 1.2 | Starting line width in pixels |
| Thickness range | [0.2, 4.0] | Clamped dynamic thickness |
| CoC factor | 0.05 | Circle of confusion from z-depth |
| Bounding box pad | `ceil(effective_thickness * 2.5)` | Pixel padding around segment |
| Energy cutoff | 0.01 | Minimum Gaussian energy to deposit |
| Depth fade rate | 0.002 | Exponential fog coefficient |
| Depth fade range | [0.05, 1.0] | Clamped atmospheric fade |
