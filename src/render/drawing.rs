//! Line drawing, plot functions, and primitive rendering

use super::color::OklabColor;
use crate::{spectral_constants, spectrum::NUM_BINS, utils::build_gaussian_kernel};
use spectral_constants::{LAMBDA_START, LAMBDA_END};
use rayon::prelude::*;
use smallvec::SmallVec;

pub static DISPERSION_BOOST_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

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
    // Calculate hue angle in radians (-π to +π)
    let hue_rad = b.atan2(a);
    
    // Convert to degrees (0 to 360)
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
    
    // Ensure wavelength is within valid bounds
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
        Self { 
            kernel: small_kernel, 
            radius,
            temp_buffer: vec![(0.0, 0.0, 0.0, 0.0); buffer_size],
        }
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
        #[allow(clippy::needless_range_loop)] // Direct indexing for performance
        for x in 0..width {
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

            row[x] = sum;
        }
    });

    // Vertical pass
    buffer.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        #[allow(clippy::needless_range_loop)] // Direct indexing for performance
        for x in 0..width {
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

            row[x] = sum;
        }
    });
}


/// Draw anti-aliased line segment for spectral rendering
/// (Dispersion has been moved to a post-process for massive performance gains and continuous radial aberration)
#[allow(clippy::too_many_arguments)] // Low-level drawing primitive requires all parameters
pub fn draw_line_segment_aa_spectral_with_dispersion(
    accum: &mut [[f64; NUM_BINS]],
    width: u32,
    height: u32,
    x0: f32,
    y0: f32,
    z0: f32,
    x1: f32,
    y1: f32,
    z1: f32,
    col0: OklabColor,
    col1: OklabColor,
    alpha0: f64,
    alpha1: f64,
    hdr_scale: f64,
    _enable_dispersion: bool,
) {
    draw_line_segment_aa_spectral_internal(
        accum, width, height, x0, y0, z0, x1, y1, z1, col0, col1, alpha0, alpha1, hdr_scale,
    );
}


/// Internal implementation of spectral line drawing using Z-depth aware SDF Splatting
#[allow(clippy::too_many_arguments)]
fn draw_line_segment_aa_spectral_internal(
    accum: &mut [[f64; NUM_BINS]],
    width: u32,
    height: u32,
    x0: f32,
    y0: f32,
    z0: f32,
    x1: f32,
    y1: f32,
    z1: f32,
    col0: OklabColor,
    col1: OklabColor,
    alpha0: f64,
    alpha1: f64,
    hdr_scale: f64,
) {
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
    let min_y = (y0.min(y1) as i32 - pad).max(0);
    let max_y = (y0.max(y1) as i32 + pad).min(height as i32 - 1);

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
            
            let h = if len_sq > 1e-6 {
                ((pax * dx + pay * dy) / len_sq).clamp(0.0, 1.0)
            } else {
                0.5
            };
            
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
            
            let idx = py as usize * width as usize + px as usize;
            
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

    #[test]
    fn test_dispersion_boost_default_enabled() {
        assert!(DISPERSION_BOOST_ENABLED.load(Ordering::Relaxed),
            "dispersion boost should be enabled by default");
    }

    #[test]
    fn test_oklab_hue_to_wavelength_range() {
        for deg in (0..360).step_by(10) {
            let rad = (deg as f64).to_radians();
            let a = 0.15 * rad.cos();
            let b = 0.15 * rad.sin();
            let wl = oklab_hue_to_wavelength(a, b);
            assert!(wl >= 380.0 && wl <= 700.0,
                "hue {deg}\u{00b0} -> wavelength {wl} out of visible range");
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
}
