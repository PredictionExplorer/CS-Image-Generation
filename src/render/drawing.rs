//! Line drawing, plot functions, and primitive rendering

use super::color::OklabColor;
use super::constants::{
    CRISP_DEPTH_BROADENING_FACTOR, CRISP_LINE_BASE_THICKNESS, CRISP_LINE_ENERGY_CUTOFF,
    CRISP_LINE_FALLOFF_EXPONENT, CRISP_LINE_MAX_THICKNESS, CRISP_LINE_MIN_THICKNESS,
    CRISP_LINE_PROXIMITY_OFFSET, CRISP_LINE_PROXIMITY_SLOPE, CRISP_LINE_SUBPIXEL_GRID,
    CRISP_SPECTRAL_KERNEL_RADIUS_BINS, CRISP_SPECTRAL_SIGMA_MAX_BINS,
    CRISP_SPECTRAL_SIGMA_MIN_BINS, LIGHTNESS_ENERGY_FLOOR, LIGHTNESS_ENERGY_GAMMA,
    LIGHTNESS_ENERGY_SPAN, crisp_line_resolution_scale,
};
use crate::{spectral_constants, spectrum::NUM_BINS, utils::build_gaussian_kernel};
use rayon::prelude::*;
use smallvec::SmallVec;
use spectral_constants::BIN_WIDTH;

/// Runtime toggle: when true, non-production spectral dispersion is applied in the render path.
pub static DISPERSION_BOOST_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// One endpoint of a line in pixel space with `OkLab` color and coverage.
#[derive(Clone, Copy, Debug)]
pub struct LineVertex {
    /// Horizontal pixel coordinate.
    pub x: f32,
    /// Vertical pixel coordinate.
    pub y: f32,
    /// Depth coordinate used for thickness and depth-of-field.
    pub z: f32,
    /// `OkLab` color at this vertex.
    pub color: OklabColor,
    /// Opacity or stroke weight multiplier in \[0, 1\] (or beyond for HDR).
    pub alpha: f64,
}

/// Anti-aliased spectral line segment between two vertices with HDR energy scale.
#[derive(Clone, Copy, Debug)]
pub struct SpectralLineSegment {
    /// Start vertex (position, color, alpha).
    pub start: LineVertex,
    /// End vertex (position, color, alpha).
    pub end: LineVertex,
    /// Multiplier for deposited spectral energy (HDR / velocity boost).
    pub hdr_scale: f64,
    /// Stroke width multiplier (seed line weight x velocity dynamics).
    pub thickness_factor: f64,
}

/// `OKLab` hue angle in degrees, normalized to `[0, 360)`.
#[inline]
fn oklab_hue_degrees(a: f64, b: f64) -> f64 {
    b.atan2(a).to_degrees().rem_euclid(360.0)
}

#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

// ---- Spectral locus inversion -------------------------------------------------
//
// Rather than a hand-tuned per-region hue->wavelength ladder, the mapping is the
// inverse of the CIE-derived spectral locus, measured once: for sampled
// wavelengths we read back the rendered OKLab hue, then interpolate the inverse.
// Equal hue steps therefore map to equal perceptual steps (no breakpoints, no
// green-heavy band), and the "line of purples" between the red and violet ends
// of the locus becomes one continuous red/violet crossfade instead of a
// special-cased magenta wrap.

const LOCUS_SAMPLES: usize = 256;
const LOCUS_LAMBDA_MIN: f64 = 405.0;
const LOCUS_LAMBDA_MAX: f64 = 690.0;

struct LocusTable {
    /// `(hue_deg, wavelength_nm)` sorted by ascending hue across the spectral arc.
    points: Vec<(f64, f64)>,
}

static SPECTRAL_LOCUS: std::sync::LazyLock<LocusTable> =
    std::sync::LazyLock::new(LocusTable::build);

#[inline]
fn locus_hue_for_wavelength(lambda: f64) -> f64 {
    let (r, g, b) = crate::spectrum::wavelength_to_rgb(lambda);
    let (_, a, b_lab) = crate::oklab::linear_srgb_to_oklab(r, g, b);
    oklab_hue_degrees(a, b_lab)
}

impl LocusTable {
    fn build() -> Self {
        let mut points: Vec<(f64, f64)> = (0..=LOCUS_SAMPLES)
            .map(|i| {
                let lambda =
                    lerp(LOCUS_LAMBDA_MIN, LOCUS_LAMBDA_MAX, i as f64 / LOCUS_SAMPLES as f64);
                (locus_hue_for_wavelength(lambda), lambda)
            })
            .collect();
        points.sort_by(|lhs, rhs| lhs.0.total_cmp(&rhs.0));
        points.dedup_by(|lhs, rhs| (lhs.0 - rhs.0).abs() < 1e-6);
        Self { points }
    }

    #[inline]
    fn arc_min_hue(&self) -> f64 {
        self.points.first().map_or(0.0, |p| p.0)
    }

    #[inline]
    fn arc_max_hue(&self) -> f64 {
        self.points.last().map_or(360.0, |p| p.0)
    }

    /// Wavelength whose rendered hue matches `hue_deg` on the spectral arc.
    fn wavelength_for_arc_hue(&self, hue_deg: f64) -> f64 {
        let hue = hue_deg.clamp(self.arc_min_hue(), self.arc_max_hue());
        let idx = self.points.partition_point(|p| p.0 < hue).clamp(1, self.points.len() - 1);
        let (h0, l0) = self.points[idx - 1];
        let (h1, l1) = self.points[idx];
        let t = if (h1 - h0).abs() < 1e-9 { 0.0 } else { (hue - h0) / (h1 - h0) };
        lerp(l0, l1, t)
    }
}

/// Spectral emission lobes `(wavelength_nm, weight)` for an `OKLab` hue.
///
/// On the locus arc this is a single lobe; across the line of purples it is a
/// red/violet pair whose balance slides smoothly, so magenta emerges naturally
/// instead of from a hard-coded wrap region.
fn hue_to_spectral_lobes(hue_deg: f64) -> SmallVec<[(f64, f64); 2]> {
    let locus = &*SPECTRAL_LOCUS;
    let (arc_min, arc_max) = (locus.arc_min_hue(), locus.arc_max_hue());
    let mut lobes = SmallVec::new();

    if (arc_min..=arc_max).contains(&hue_deg) {
        lobes.push((locus.wavelength_for_arc_hue(hue_deg), 1.0));
        return lobes;
    }

    let gap_width = 360.0 - arc_max + arc_min;
    let red_weight = smoothstep((hue_deg - arc_max).rem_euclid(360.0) / gap_width);
    lobes.push((locus.wavelength_for_arc_hue(arc_max), 1.0 - red_weight));
    lobes.push((locus.wavelength_for_arc_hue(arc_min), red_weight));
    lobes
}

#[inline]
fn smoothstep(t: f64) -> f64 {
    let x = t.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

type SpectralKernel = SmallVec<[(usize, f64); 16]>;

fn add_gaussian_lobe(
    kernel: &mut SpectralKernel,
    center_wavelength: f64,
    sigma_bins: f64,
    weight: f64,
) {
    if weight <= 0.0 {
        return;
    }

    let center_bin = spectral_constants::wavelength_to_bin(center_wavelength);
    let radius = (sigma_bins * 2.0).ceil().min(CRISP_SPECTRAL_KERNEL_RADIUS_BINS as f64) as isize;
    let base = center_bin.round() as isize;

    for bin_i in (base - radius)..=(base + radius) {
        if !(0..NUM_BINS as isize).contains(&bin_i) {
            continue;
        }
        let bin = bin_i as usize;
        let d = bin as f64 - center_bin;
        let value = weight * (-(d * d) / (2.0 * sigma_bins * sigma_bins)).exp();
        if value <= 1e-6 {
            continue;
        }

        if let Some((_, existing)) = kernel.iter_mut().find(|(idx, _)| *idx == bin) {
            *existing += value;
        } else {
            kernel.push((bin, value));
        }
    }
}

/// Energy multiplier derived from `OKLab` lightness.
///
/// The palette engine assigns each body a deliberate lightness rank, but the
/// spectral kernel itself only encodes hue and chroma. Scaling deposited
/// energy by lightness makes that hierarchy (and the per-step lightness
/// waves) actually visible: bright bodies radiate, dark bodies recede.
#[inline]
fn lightness_energy_factor(lightness: f64) -> f64 {
    let t = ((lightness - 0.30) / 0.64).clamp(0.0, 1.0);
    LIGHTNESS_ENERGY_FLOOR + LIGHTNESS_ENERGY_SPAN * t.powf(LIGHTNESS_ENERGY_GAMMA)
}

fn spectral_kernel_for_oklab(color: OklabColor) -> SpectralKernel {
    let (l, a, b) = color;
    let chroma = (a * a + b * b).sqrt();
    let purity = (chroma / 0.34).clamp(0.0, 1.0);
    let sigma_nm = 8.0 - 5.5 * purity;
    let sigma_bins =
        (sigma_nm / BIN_WIDTH).clamp(CRISP_SPECTRAL_SIGMA_MIN_BINS, CRISP_SPECTRAL_SIGMA_MAX_BINS);

    let mut kernel = SpectralKernel::new();
    for (lambda, weight) in hue_to_spectral_lobes(oklab_hue_degrees(a, b)) {
        add_gaussian_lobe(&mut kernel, lambda, sigma_bins, weight);
    }

    let sum: f64 = kernel.iter().map(|(_, w)| *w).sum();
    if sum > 0.0 {
        let energy = lightness_energy_factor(l);
        for (_, weight) in &mut kernel {
            *weight *= energy / sum;
        }
    }
    kernel
}

/// Gaussian blur context for efficient blurring with reusable temp buffer
pub(crate) struct GaussianBlurContext {
    kernel: SmallVec<[f64; 32]>,
    radius: usize,
    temp_buffer: Vec<(f64, f64, f64, f64)>,
}

impl GaussianBlurContext {
    fn new(radius: usize, buffer_size: usize) -> Self {
        let kernel = build_gaussian_kernel(radius);
        let kernel_len = kernel.len();
        let mut small_kernel = SmallVec::with_capacity(kernel_len);
        small_kernel.extend_from_slice(&kernel);
        Self { kernel: small_kernel, radius, temp_buffer: vec![(0.0, 0.0, 0.0, 0.0); buffer_size] }
    }

    /// Ensure temp buffer has correct capacity
    fn ensure_capacity(&mut self, size: usize) {
        if self.temp_buffer.len() != size {
            self.temp_buffer.resize(size, (0.0, 0.0, 0.0, 0.0));
        }
    }
}

/// Apply 2D Gaussian blur to RGBA buffer in parallel
pub fn parallel_blur_2d_rgba(
    buffer: &mut [(f64, f64, f64, f64)],
    width: usize,
    height: usize,
    radius: usize,
) {
    if radius == 0 {
        return;
    }

    let mut blur_ctx = GaussianBlurContext::new(radius, buffer.len());

    // Horizontal pass (reuse pre-allocated temp buffer)
    blur_ctx.ensure_capacity(buffer.len());
    blur_ctx.temp_buffer.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        for (x, pixel_out) in row.iter_mut().enumerate() {
            let mut sum = (0.0, 0.0, 0.0, 0.0);

            for (i, &k) in blur_ctx.kernel.iter().enumerate() {
                let src_x = (x as i32 + i as i32 - blur_ctx.radius as i32)
                    .clamp(0, width as i32 - 1) as usize;
                let pixel = buffer[y * width + src_x];
                sum.0 += pixel.0 * k;
                sum.1 += pixel.1 * k;
                sum.2 += pixel.2 * k;
                sum.3 += pixel.3 * k;
            }

            *pixel_out = sum;
        }
    });

    // Vertical pass
    buffer.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        for (x, pixel_out) in row.iter_mut().enumerate() {
            let mut sum = (0.0, 0.0, 0.0, 0.0);

            for (i, &k) in blur_ctx.kernel.iter().enumerate() {
                let src_y = (y as i32 + i as i32 - blur_ctx.radius as i32)
                    .clamp(0, height as i32 - 1) as usize;
                let pixel = blur_ctx.temp_buffer[src_y * width + x];
                sum.0 += pixel.0 * k;
                sum.1 += pixel.1 * k;
                sum.2 += pixel.2 * k;
                sum.3 += pixel.3 * k;
            }

            *pixel_out = sum;
        }
    });
}

/// Draw anti-aliased line segment for spectral rendering using Z-depth aware SDF Splatting
pub fn draw_line_segment_aa_spectral(
    accum: &mut [[f64; NUM_BINS]],
    width: u32,
    height: u32,
    segment: SpectralLineSegment,
) {
    draw_line_segment_aa_spectral_rows(accum, width, height, 0, height as usize, segment);
}

/// Draw anti-aliased line segment into an owned row band of the destination buffer.
pub(crate) fn draw_line_segment_aa_spectral_rows(
    accum: &mut [[f64; NUM_BINS]],
    width: u32,
    height: u32,
    row_start: usize,
    row_end: usize,
    segment: SpectralLineSegment,
) {
    let row_end = row_end.min(height as usize);
    if row_start >= row_end || width == 0 || height == 0 {
        return;
    }

    let LineVertex { x: x0, y: y0, z: z0, color: col0, alpha: alpha0 } = segment.start;
    let LineVertex { x: x1, y: y1, z: z1, color: col1, alpha: alpha1 } = segment.end;
    let hdr_scale = segment.hdr_scale;
    let dx = x1 - x0;
    let dy = y1 - y0;
    let dz = z1 - z0;

    let len_sq = dx * dx + dy * dy;
    let len_3d = (dx * dx + dy * dy + dz * dz).sqrt();

    // Dynamic line width. Three responses combine, all resolution-aware:
    //   1. Proximity: short segments (close encounters, per-step ribbon strokes)
    //      draw bold, long inter-body spans draw fine.
    //   2. Seed line weight + velocity dynamics, via `segment.thickness_factor`.
    //   3. Clamp to the crisp production range.
    let resolution_scale = crisp_line_resolution_scale(width, height);
    let normalized_len_3d = len_3d / resolution_scale;
    let proximity =
        1.0 / (CRISP_LINE_PROXIMITY_OFFSET + normalized_len_3d * CRISP_LINE_PROXIMITY_SLOPE);
    let thickness = (CRISP_LINE_BASE_THICKNESS
        * resolution_scale
        * (segment.thickness_factor as f32)
        * proximity)
        .clamp(
            CRISP_LINE_MIN_THICKNESS * resolution_scale,
            CRISP_LINE_MAX_THICKNESS * resolution_scale,
        );

    // Z-depth calculation (center of segment)
    let avg_z = (z0 + z1) * 0.5;

    // Production crisp mode uses no depth-of-field broadening.
    let coc = (avg_z * CRISP_DEPTH_BROADENING_FACTOR).abs();
    let effective_thickness = thickness + coc;

    // Maximum extent of the SDF bounding box
    let pad = (effective_thickness * 3.0).ceil() as i32;

    let min_x = (x0.min(x1) as i32 - pad).max(0);
    let max_x = (x0.max(x1) as i32 + pad).min(width as i32 - 1);
    let min_y = (y0.min(y1) as i32 - pad).max(row_start as i32);
    let max_y = (y0.max(y1) as i32 + pad).min(row_end as i32 - 1);

    if min_x > max_x || min_y > max_y {
        return;
    }

    let kernel0 = spectral_kernel_for_oklab(col0);
    let kernel1 = spectral_kernel_for_oklab(col1);

    // Energy conservation: wider lines due to DOF should distribute same total energy
    let energy_conservation = thickness / effective_thickness;

    // Keep distant geometry visible; the black field provides separation without fog.
    let depth_fade = (-avg_z.abs() * 0.0007).exp().clamp(0.18, 1.0);
    let base_energy_mult = hdr_scale * f64::from(depth_fade) * f64::from(energy_conservation);
    let subpixel_grid = CRISP_LINE_SUBPIXEL_GRID.max(1);
    let subpixel_count = (subpixel_grid * subpixel_grid) as f64;

    for py in min_y..=max_y {
        for px in min_x..=max_x {
            let mut coverage_sum = 0.0_f64;
            let mut start_energy_sum = 0.0_f64;
            let mut end_energy_sum = 0.0_f64;

            for sy in 0..subpixel_grid {
                for sx in 0..subpixel_grid {
                    let sample_x = px as f32 + (sx as f32 + 0.5) / subpixel_grid as f32;
                    let sample_y = py as f32 + (sy as f32 + 0.5) / subpixel_grid as f32;
                    let pax = sample_x - x0;
                    let pay = sample_y - y0;

                    let h = if len_sq > 1e-6 {
                        ((pax * dx + pay * dy) / len_sq).clamp(0.0, 1.0)
                    } else {
                        0.5
                    };

                    let proj_x = pax - dx * h;
                    let proj_y = pay - dy * h;
                    let dist_sq = proj_x * proj_x + proj_y * proj_y;

                    // Super-Gaussian SDF coverage integrated over subpixel samples.
                    let normalized_dist_sq = dist_sq / (effective_thickness * effective_thickness);
                    let energy = f64::from(
                        (-(normalized_dist_sq * normalized_dist_sq) * CRISP_LINE_FALLOFF_EXPONENT)
                            .exp(),
                    );
                    let h64 = f64::from(h);
                    let alpha = alpha0 * (1.0 - h64) + alpha1 * h64;
                    let weighted_energy = energy * alpha;

                    coverage_sum += energy;
                    start_energy_sum += weighted_energy * (1.0 - h64);
                    end_energy_sum += weighted_energy * h64;
                }
            }

            let coverage = coverage_sum / subpixel_count;
            if coverage < f64::from(CRISP_LINE_ENERGY_CUTOFF) {
                continue;
            }

            let idx = (py as usize - row_start) * width as usize + px as usize;
            let energy_scale = base_energy_mult / subpixel_count;

            for &(bin, weight) in &kernel0 {
                accum[idx][bin] += energy_scale * start_energy_sum * weight;
            }
            for &(bin, weight) in &kernel1 {
                accum[idx][bin] += energy_scale * end_energy_sum * weight;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oklab::linear_srgb_to_oklab;
    use crate::spectrum::BIN_COMBINED_LUT;
    use std::sync::atomic::Ordering;

    fn wavelength_to_oklab(wavelength_nm: f64, intensity: f64) -> (f64, f64, f64) {
        let bin_f = spectral_constants::wavelength_to_bin(wavelength_nm);
        let left = bin_f.floor() as usize;
        let right = (left + 1).min(NUM_BINS - 1);
        let mix = bin_f.fract();
        let (lr, lg, lb, _) = BIN_COMBINED_LUT[left];
        let (rr, rg, rb, _) = BIN_COMBINED_LUT[right];
        let r = (lr * (1.0 - mix) + rr * mix) * intensity;
        let g = (lg * (1.0 - mix) + rg * mix) * intensity;
        let b = (lb * (1.0 - mix) + rb * mix) * intensity;
        linear_srgb_to_oklab(r, g, b)
    }

    fn make_segment(
        start: (f32, f32, f32),
        end: (f32, f32, f32),
        hdr_scale: f64,
    ) -> SpectralLineSegment {
        SpectralLineSegment {
            start: LineVertex {
                x: start.0,
                y: start.1,
                z: start.2,
                color: wavelength_to_oklab(620.0, 0.9),
                alpha: 0.85,
            },
            end: LineVertex {
                x: end.0,
                y: end.1,
                z: end.2,
                color: wavelength_to_oklab(470.0, 0.7),
                alpha: 0.55,
            },
            hdr_scale,
            thickness_factor: 1.0,
        }
    }

    fn draw_with_row_bands(
        segment: SpectralLineSegment,
        width: usize,
        height: usize,
        band_count: usize,
    ) -> Vec<[f64; NUM_BINS]> {
        let mut accum = vec![[0.0; NUM_BINS]; width * height];
        let band_count = band_count.max(1).min(height.max(1));
        let rows_per_band = height.div_ceil(band_count);

        for band_idx in 0..band_count {
            let row_start = band_idx * rows_per_band;
            let row_end = (row_start + rows_per_band).min(height);
            if row_start >= row_end {
                break;
            }

            let start = row_start * width;
            let end = row_end * width;
            draw_line_segment_aa_spectral_rows(
                &mut accum[start..end],
                width as u32,
                height as u32,
                row_start,
                row_end,
                segment,
            );
        }

        accum
    }

    fn assert_spd_buffers_bits_eq(
        actual: &[[f64; NUM_BINS]],
        expected: &[[f64; NUM_BINS]],
        label: &str,
    ) {
        assert_eq!(actual.len(), expected.len(), "{label}: buffer lengths differ");
        for (pixel_idx, (lhs, rhs)) in actual.iter().zip(expected).enumerate() {
            for (bin_idx, (&lhs_bin, &rhs_bin)) in lhs.iter().zip(rhs.iter()).enumerate() {
                assert_eq!(
                    lhs_bin.to_bits(),
                    rhs_bin.to_bits(),
                    "{label}: pixel {pixel_idx} bin {bin_idx} diverged ({lhs_bin} vs {rhs_bin})"
                );
            }
        }
    }

    #[test]
    fn test_dispersion_boost_default_disabled() {
        assert!(
            !DISPERSION_BOOST_ENABLED.load(Ordering::Relaxed),
            "dispersion boost should be disabled by default for crisp production output"
        );
    }

    #[test]
    fn test_hue_to_spectral_lobes_stay_in_visible_range() {
        for deg in (0..360).step_by(5) {
            let lobes = hue_to_spectral_lobes(f64::from(deg));
            let weight: f64 = lobes.iter().map(|(_, w)| *w).sum();
            assert!(weight > 0.0, "hue {deg}\u{00b0} produced no spectral energy");
            for (lambda, _) in &lobes {
                assert!(
                    (380.0..=700.0).contains(lambda),
                    "hue {deg}\u{00b0} -> wavelength {lambda} out of visible range"
                );
            }
        }
    }

    /// Energy-weighted histogram of rendered hues across many seeds, for the
    /// single-colour spectral round-trip. The diagnostic + regression signal for
    /// even hue coverage after the palette/round-trip redesign.
    fn rendered_hue_histogram(seeds: u32, bins: usize) -> Vec<f64> {
        use crate::render::color::generate_body_color_sequences;
        use crate::spectrum::spd_to_rgba;

        let mut hist = vec![0.0f64; bins];
        for s in 0..seeds {
            let seed = [(s & 0xff) as u8, (s >> 8) as u8, 0x5a, 0xa5];
            let mut rng = crate::sim::Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
            let phase = f64::from(s % 97) / 97.0;
            let (colors, alphas) =
                generate_body_color_sequences(&mut rng, 48, 15_000_000, true, true, phase);
            for (body, sequence) in colors.iter().enumerate() {
                for color in sequence.iter().step_by(4) {
                    let mut spd = [0.0f64; NUM_BINS];
                    for (bin, weight) in spectral_kernel_for_oklab(*color) {
                        spd[bin] += weight;
                    }
                    let (r, g, b, alpha) = spd_to_rgba(&spd);
                    if alpha <= 1e-6 {
                        continue;
                    }
                    let (_, la, lb) = crate::oklab::linear_rec2020_to_oklab(r, g, b);
                    let hue = lb.atan2(la).to_degrees().rem_euclid(360.0);
                    let idx = ((hue / 360.0 * bins as f64) as usize).min(bins - 1);
                    hist[idx] += alphas[body];
                }
            }
        }
        let sum: f64 = hist.iter().sum();
        if sum > 0.0 {
            for value in &mut hist {
                *value /= sum;
            }
        }
        hist
    }

    fn normalized_entropy(hist: &[f64]) -> f64 {
        let entropy: f64 = hist.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
        entropy / (hist.len() as f64).ln()
    }

    #[test]
    fn rendered_hue_distribution_is_broadly_uniform() {
        let bins = 12;
        let hist = rendered_hue_histogram(192, bins);
        let occupied = hist.iter().filter(|&&p| p > 0.0).count();
        let max_share = hist.iter().copied().fold(0.0, f64::max);
        let entropy = normalized_entropy(&hist);
        assert_eq!(occupied, bins, "every hue bin should receive energy: {hist:?}");
        assert!(
            entropy > 0.9,
            "rendered hue coverage should be near-uniform: entropy={entropy:.3} {hist:?}"
        );
        assert!(max_share < 0.25, "no hue bin should dominate: max_share={max_share:.3} {hist:?}");
    }

    #[test]
    #[ignore = "diagnostic: run with --ignored --nocapture to print the hue histogram"]
    fn report_rendered_hue_distribution() {
        let bins = 12;
        let hist = rendered_hue_histogram(1024, bins);
        eprintln!("rendered hue distribution ({bins} bins, normalized):");
        for (i, share) in hist.iter().enumerate() {
            let lo = i * 360 / bins;
            let hi = (i + 1) * 360 / bins;
            eprintln!("  [{lo:>3}..{hi:>3}) {share:.4} {}", "#".repeat((share * 200.0) as usize));
        }
        eprintln!(
            "normalized entropy = {:.4} (1.0 = perfectly uniform)",
            normalized_entropy(&hist)
        );
    }

    #[test]
    fn test_lightness_scales_deposited_energy() {
        let bright = wavelength_to_oklab(550.0, 0.9);
        let dark = (0.40, bright.1, bright.2);
        let bright_kernel = spectral_kernel_for_oklab((0.90, bright.1, bright.2));
        let dark_kernel = spectral_kernel_for_oklab(dark);

        let bright_sum: f64 = bright_kernel.iter().map(|(_, w)| *w).sum();
        let dark_sum: f64 = dark_kernel.iter().map(|(_, w)| *w).sum();

        assert!(
            bright_sum > dark_sum * 1.5,
            "high-lightness colors should deposit more energy: bright={bright_sum} dark={dark_sum}"
        );
        assert!(dark_sum > 0.0, "dark colors must still deposit energy (floor)");
    }

    #[test]
    fn test_thickness_factor_widens_stroke_footprint() {
        let width = 32usize;
        let height = 32usize;
        let thin_segment = SpectralLineSegment {
            thickness_factor: 0.6,
            ..make_segment((4.0, 16.0, 0.0), (28.0, 16.0, 0.0), 1.0)
        };
        let thick_segment = SpectralLineSegment { thickness_factor: 2.0, ..thin_segment };

        let mut thin = vec![[0.0; NUM_BINS]; width * height];
        let mut thick = vec![[0.0; NUM_BINS]; width * height];
        draw_line_segment_aa_spectral(&mut thin, width as u32, height as u32, thin_segment);
        draw_line_segment_aa_spectral(&mut thick, width as u32, height as u32, thick_segment);

        let active = |buf: &[[f64; NUM_BINS]]| {
            buf.iter().filter(|bins| bins.iter().sum::<f64>() > 1e-12).count()
        };
        assert!(
            active(&thick) > active(&thin),
            "higher thickness_factor should cover more pixels: thick={} thin={}",
            active(&thick),
            active(&thin)
        );
    }

    #[test]
    fn test_wavelength_to_oklab_roundtrip() {
        let wl = 620.0;
        let (l, a, b) = wavelength_to_oklab(wl, 0.7);
        assert!(l > 0.0, "lightness should be positive");
        let chroma = (a * a + b * b).sqrt();
        assert!(chroma > 0.0, "wavelength {wl} should produce colored output");
    }

    #[test]
    fn test_wavelength_to_oklab_endpoints() {
        let blue = wavelength_to_oklab(420.0, 0.7);
        let red = wavelength_to_oklab(650.0, 0.7);
        assert!(blue.0 > 0.0, "blue lightness should be positive");
        assert!(red.0 > 0.0, "red lightness should be positive");
    }

    #[test]
    fn test_row_banded_line_draw_matches_full_frame_bits() {
        let width = 24usize;
        let height = 18usize;
        let segments = [
            ("single_band", make_segment((3.0, 2.0, 0.1), (20.0, 5.0, -0.2), 1.4)),
            ("cross_one_boundary", make_segment((2.0, 4.0, -0.5), (18.0, 10.0, 0.7), 0.9)),
            ("cross_many_bands", make_segment((1.0, 0.0, 0.4), (22.0, 17.0, -0.8), 1.8)),
            ("clips_edges", make_segment((-3.0, -4.0, 0.2), (26.0, 19.0, -0.3), 1.1)),
            ("near_zero_length", make_segment((12.2, 7.8, 0.0), (12.25, 7.85, 0.0), 0.75)),
        ];

        for (label, segment) in segments {
            let mut full = vec![[0.0; NUM_BINS]; width * height];
            draw_line_segment_aa_spectral(&mut full, width as u32, height as u32, segment);

            for band_count in [1usize, 2, 3, height] {
                let banded = draw_with_row_bands(segment, width, height, band_count);
                assert_spd_buffers_bits_eq(&banded, &full, &format!("{label}/bands={band_count}"));
            }
        }
    }

    #[test]
    fn test_center_sampled_line_splat_spreads_subpixel_diagonal_coverage() {
        let width = 18usize;
        let height = 10usize;
        let segment = make_segment((1.25, 3.35, 0.0), (16.25, 5.15, 0.0), 1.0);
        let mut accum = vec![[0.0; NUM_BINS]; width * height];

        draw_line_segment_aa_spectral(&mut accum, width as u32, height as u32, segment);

        let row_energy = |row: usize| -> f64 {
            accum[row * width..(row + 1) * width].iter().flat_map(|bins| bins.iter()).sum::<f64>()
        };
        let active_rows = (0..height).filter(|&row| row_energy(row) > f64::EPSILON).count();

        assert!(active_rows >= 2, "subpixel diagonal should retain anti-aliased row coverage");
        assert!(row_energy(4) > row_energy(2), "main diagonal band should dominate distant rows");
    }

    #[test]
    fn test_production_alpha_line_survives_coverage_cutoff() {
        let width = 2234usize;
        let height = 2234usize;
        let row_start = 100usize;
        let row_end = row_start + 1;
        let mut accum = vec![[0.0; NUM_BINS]; width * (row_end - row_start)];
        let mut segment =
            make_segment((100.0, row_start as f32, 0.0), (300.0, row_start as f32, 0.0), 1.0);
        segment.start.alpha = 1.0 / 15_000_000.0;
        segment.end.alpha = 1.0 / 15_000_000.0;

        draw_line_segment_aa_spectral_rows(
            &mut accum,
            width as u32,
            height as u32,
            row_start,
            row_end,
            segment,
        );

        let total_energy: f64 = accum.iter().flat_map(|bins| bins.iter()).sum();
        assert!(
            total_energy > 0.0,
            "production-scale alpha should not be rejected by the geometric coverage cutoff"
        );
    }

    #[test]
    fn test_crisp_line_resolution_scale_tracks_output_size() {
        let default_scale = crate::render::constants::crisp_line_resolution_scale(3456, 2234);
        let preview_scale = crate::render::constants::crisp_line_resolution_scale(640, 360);
        let large_scale = crate::render::constants::crisp_line_resolution_scale(10_000, 6_460);
        let huge_scale = crate::render::constants::crisp_line_resolution_scale(200_000, 100_000);

        assert!((default_scale - 1.0).abs() < 0.001);
        assert_eq!(preview_scale, crate::render::constants::CRISP_LINE_RESOLUTION_SCALE_MIN);
        assert!(large_scale > default_scale);
        assert_eq!(huge_scale, crate::render::constants::CRISP_LINE_RESOLUTION_SCALE_MAX);
    }

    #[test]
    fn test_crisp_line_interpolation_is_conservative_at_default_resolution() {
        let moderate_default_substeps =
            crate::render::constants::crisp_line_interpolation_substeps(3456, 2234, 8.0);
        let extreme_default_substeps =
            crate::render::constants::crisp_line_interpolation_substeps(3456, 2234, 20.0);
        let large_substeps =
            crate::render::constants::crisp_line_interpolation_substeps(10_000, 6_460, 20.0);
        let subpixel_motion =
            crate::render::constants::crisp_line_interpolation_substeps(3456, 2234, 0.75);
        let capped_substeps =
            crate::render::constants::crisp_line_interpolation_substeps(100_000, 64_640, 10_000.0);

        assert_eq!(subpixel_motion, 1);
        assert_eq!(moderate_default_substeps, 1);
        assert!(extreme_default_substeps > 1);
        assert!(large_substeps >= extreme_default_substeps);
        assert_eq!(capped_substeps, crate::render::constants::CRISP_INTERPOLATION_MAX_SUBSTEPS);
    }
}
