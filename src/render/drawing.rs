//! Line drawing, plot functions, and primitive rendering

use super::color::OklabColor;
use crate::{spectral_constants, spectrum::NUM_BINS, utils::build_gaussian_kernel};
use rayon::prelude::*;
use smallvec::SmallVec;
use spectral_constants::{LAMBDA_END, LAMBDA_START};

pub static DISPERSION_BOOST_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

#[derive(Clone, Copy, Debug)]
pub struct LineVertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub color: OklabColor,
    pub alpha: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct SpectralLineSegment {
    pub start: LineVertex,
    pub end: LineVertex,
    pub hdr_scale: f64,
}

/// Convert OkLab hue to wavelength with perceptually uniform distribution.
///
/// This mapping ensures that the full visible spectrum (380-700nm) is utilized,
/// providing rich color diversity across blues, greens, yellows, oranges, and reds.
///
/// The mapping is designed to align with perceptual color relationships:
/// - Red hues (around 0°) map to long wavelengths (650-700nm)
/// - Yellow hues (around 60°) map to yellow wavelengths (570-590nm)
/// - Green hues (around 120°) map to green wavelengths (510-550nm)
/// - Cyan hues (around 180°) map to cyan wavelengths (485-510nm)
/// - Blue hues (around 240°) map to blue wavelengths (450-485nm)
/// - Violet hues (around 300°) map to violet wavelengths (380-450nm)
#[inline]
pub(crate) fn oklab_hue_to_wavelength(a: f64, b: f64) -> f64 {
    let hue_rad = b.atan2(a);
    let mut hue_deg = hue_rad.to_degrees();
    if hue_deg < 0.0 {
        hue_deg += 360.0;
    }

    // Map hue to wavelength using a perceptually uniform distribution
    // This mapping is designed to maximize color variety and align with
    // the natural color spectrum while accounting for OkLab's hue distribution
    let wavelength = if hue_deg < 30.0 {
        // Red to red-orange (0-30°) -> 700-650nm
        700.0 - (hue_deg / 30.0) * 50.0
    } else if hue_deg < 60.0 {
        // Red-orange to orange (30-60°) -> 650-620nm
        650.0 - ((hue_deg - 30.0) / 30.0) * 30.0
    } else if hue_deg < 90.0 {
        // Orange to yellow (60-90°) -> 620-570nm
        620.0 - ((hue_deg - 60.0) / 30.0) * 50.0
    } else if hue_deg < 150.0 {
        // Yellow to green (90-150°) -> 570-510nm
        570.0 - ((hue_deg - 90.0) / 60.0) * 60.0
    } else if hue_deg < 210.0 {
        // Green to cyan (150-210°) -> 510-485nm
        510.0 - ((hue_deg - 150.0) / 60.0) * 25.0
    } else if hue_deg < 270.0 {
        // Cyan to blue (210-270°) -> 485-450nm
        485.0 - ((hue_deg - 210.0) / 60.0) * 35.0
    } else if hue_deg < 330.0 {
        // Blue to violet (270-330°) -> 450-380nm
        450.0 - ((hue_deg - 270.0) / 60.0) * 70.0
    } else {
        // Violet to red (330-360°) -> 380-700nm (wrap around)
        // Create smooth transition back to red
        380.0 + ((hue_deg - 330.0) / 30.0) * 320.0
    };

    wavelength.clamp(LAMBDA_START, LAMBDA_END)
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

    // Dynamic line width: faster -> thinner, slower -> thicker
    let base_thickness = 1.2;
    let thickness = (base_thickness / (0.1 + len_3d * 0.5)).clamp(0.2, 4.0);

    // Z-depth calculation (center of segment)
    let avg_z = (z0 + z1) * 0.5;

    // Depth of field (Circle of Confusion)
    // Assuming focal plane is at Z = 0
    let coc = (avg_z * 0.05).abs();
    let effective_thickness = thickness + coc;

    // Maximum extent of the SDF bounding box
    let pad = (effective_thickness * 2.5).ceil() as i32;

    let min_x = (x0.min(x1) as i32 - pad).max(0);
    let max_x = (x0.max(x1) as i32 + pad).min(width as i32 - 1);
    let min_y = (y0.min(y1) as i32 - pad).max(row_start as i32);
    let max_y = (y0.max(y1) as i32 + pad).min(row_end as i32 - 1);

    if min_x > max_x || min_y > max_y {
        return;
    }

    let (_l0, a0, b0) = col0;
    let (_l1, a1, b1) = col1;
    let wavelength0 = oklab_hue_to_wavelength(a0, b0);
    let wavelength1 = oklab_hue_to_wavelength(a1, b1);

    let bin0_f = spectral_constants::wavelength_to_bin(wavelength0);
    let bin1_f = spectral_constants::wavelength_to_bin(wavelength1);

    // Energy conservation: wider lines due to DOF should distribute same total energy
    let energy_conservation = thickness / effective_thickness;

    // Atmospheric attenuation (fog)
    let depth_fade = (-avg_z.abs() * 0.002).exp().clamp(0.05, 1.0);
    let base_energy_mult = hdr_scale * depth_fade as f64 * energy_conservation as f64;

    for py in min_y..=max_y {
        for px in min_x..=max_x {
            let pax = px as f32 - x0;
            let pay = py as f32 - y0;

            let h =
                if len_sq > 1e-6 { ((pax * dx + pay * dy) / len_sq).clamp(0.0, 1.0) } else { 0.5 };

            let proj_x = pax - dx * h;
            let proj_y = pay - dy * h;
            let dist_sq = proj_x * proj_x + proj_y * proj_y;

            // Gaussian SDF falloff
            let energy = (-dist_sq / (effective_thickness * effective_thickness)).exp();
            if energy < 0.01 {
                continue;
            }

            let alpha = alpha0 * (1.0 - h as f64) + alpha1 * h as f64;
            let final_energy = energy as f64 * alpha * base_energy_mult;

            let bin_f = bin0_f * (1.0 - h as f64) + bin1_f * h as f64;
            let bin_left = (bin_f.floor() as usize).min(NUM_BINS - 1);
            let bin_right = (bin_left + 1).min(NUM_BINS - 1);
            let w_right = bin_f.fract();

            let idx = (py as usize - row_start) * width as usize + px as usize;

            if bin_right == bin_left {
                accum[idx][bin_left] += final_energy;
            } else {
                accum[idx][bin_left] += final_energy * (1.0 - w_right);
                accum[idx][bin_right] += final_energy * w_right;
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
    fn test_dispersion_boost_default_enabled() {
        assert!(
            DISPERSION_BOOST_ENABLED.load(Ordering::Relaxed),
            "dispersion boost should be enabled by default"
        );
    }

    #[test]
    fn test_oklab_hue_to_wavelength_range() {
        for deg in (0..360).step_by(10) {
            let rad = (deg as f64).to_radians();
            let a = 0.15 * rad.cos();
            let b = 0.15 * rad.sin();
            let wl = oklab_hue_to_wavelength(a, b);
            assert!(
                (380.0..=700.0).contains(&wl),
                "hue {deg}\u{00b0} -> wavelength {wl} out of visible range"
            );
        }
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
}
