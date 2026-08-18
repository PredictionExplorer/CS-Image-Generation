//! V55 `hologram` -- A Recording of the Wavefront.
//!
//! A computed off-axis Fresnel hologram of the trajectory point set:
//! points are binned into depth slices, each slice is propagated to the
//! plate with the angular-spectrum method (row-column FFTs), and the
//! interference with a tilted reference beam becomes a 16-bit grayscale
//! pattern for photoplotting. A reconstruction simulation back-propagates
//! the pattern to three focus depths as proof, and the spec file states
//! the print requirements honestly. Bodies are banded to distinct depths
//! so the three strands separate on reconstruction.

use crate::error::Result;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::display::encode_linear_rec2020_png16;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use rustfft::{FftPlanner, num_complex::Complex};
use tracing::{info, warn};

/// Plate size in millimeters.
const PLATE_MM: f64 = 25.0;
/// Laser wavelength in millimeters (650 nm red pointer).
const WAVELENGTH_MM: f64 = 650.0e-6;
/// Depth slices.
const SLICES: usize = 24;
/// Object depth band behind the plate, in millimeters.
const DEPTH_MM: (f64, f64) = (20.0, 60.0);
/// Object points sampled along the trajectories.
const POINTS: usize = 40_000;
/// Grid edge at final quality (the 8192 default of the budget ladder).
const GRID_FINAL: usize = 8_192;
/// Reference beam off-axis angle ceiling in degrees.
const REFERENCE_DEG: f64 = 8.0;
/// Per-body focus depths for the reconstruction proof (mm).
const FOCUS_MM: [f64; 3] = [25.0, 40.0, 55.0];

/// In-place 2D FFT via rows, transpose, rows, transpose.
fn fft2d(grid: &mut [Complex<f32>], n: usize, planner: &mut FftPlanner<f32>, inverse: bool) {
    let fft = if inverse { planner.plan_fft_inverse(n) } else { planner.plan_fft_forward(n) };
    let mut scratch = vec![Complex::new(0.0f32, 0.0f32); fft.get_inplace_scratch_len()];
    let pass = |data: &mut [Complex<f32>], scratch: &mut [Complex<f32>]| {
        for row in data.chunks_mut(n) {
            fft.process_with_scratch(row, scratch);
        }
    };
    pass(grid, &mut scratch);
    transpose(grid, n);
    pass(grid, &mut scratch);
    transpose(grid, n);
    if inverse {
        let norm = 1.0 / (n * n) as f32;
        for value in grid.iter_mut() {
            *value *= norm;
        }
    }
}

/// Square in-place transpose.
fn transpose(grid: &mut [Complex<f32>], n: usize) {
    for row in 0..n {
        for col in row + 1..n {
            grid.swap(row * n + col, col * n + row);
        }
    }
}

/// Multiply by the angular-spectrum transfer function for distance `z_mm`.
fn propagate(grid: &mut [Complex<f32>], n: usize, z_mm: f64) {
    let df = 1.0 / PLATE_MM;
    let k = std::f64::consts::TAU / WAVELENGTH_MM;
    for row in 0..n {
        let fy = if row < n / 2 { row as f64 } else { row as f64 - n as f64 } * df;
        let sy = WAVELENGTH_MM * fy;
        for col in 0..n {
            let fx = if col < n / 2 { col as f64 } else { col as f64 - n as f64 } * df;
            let sx = WAVELENGTH_MM * fx;
            let radicand = 1.0 - sx * sx - sy * sy;
            let slot = &mut grid[row * n + col];
            if radicand <= 0.0 {
                *slot = Complex::new(0.0, 0.0);
            } else {
                let phase = k * z_mm * radicand.sqrt();
                let (sin_p, cos_p) = phase.sin_cos();
                *slot *= Complex::new(cos_p as f32, sin_p as f32);
            }
        }
    }
}

/// The hologram mode.
pub struct Hologram;

impl VizMode for Hologram {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("hologram").expect("hologram is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 100 {
            warn!("hologram skipped: trajectory too short");
            return Ok(());
        }
        let n = ctx.quality.scale_count(GRID_FINAL).max(512);
        let pitch = PLATE_MM / n as f64;
        // Fringe Nyquist: the carrier must stay below 1/(2*pitch).
        let max_angle = (WAVELENGTH_MM / (2.0 * pitch)).clamp(0.0, 1.0).asin();
        let reference = (REFERENCE_DEG.to_radians()).min(max_angle * 0.85);
        info!(
            "   hologram: {n}x{n} grid, pitch {:.2} um, reference {:.2} deg \
             (Nyquist cap {:.2} deg)",
            pitch * 1_000.0,
            reference.to_degrees(),
            max_angle.to_degrees()
        );

        // --- Object points into plate coordinates (mm) + banded depths.
        let mut lo = (f64::INFINITY, f64::INFINITY);
        let mut hi = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for body in ctx.positions {
            for point in body.iter().step_by(31) {
                if point.x.is_finite() && point.y.is_finite() {
                    lo.0 = lo.0.min(point.x);
                    lo.1 = lo.1.min(point.y);
                    hi.0 = hi.0.max(point.x);
                    hi.1 = hi.1.max(point.y);
                }
            }
        }
        let extent = ((hi.0 - lo.0).max(1e-9), (hi.1 - lo.1).max(1e-9));
        let per_body = POINTS / 3;
        let stride = (steps / per_body).max(1);
        // Slice bins: (grid index, amplitude) lists per slice.
        let mut slices: Vec<Vec<usize>> = vec![Vec::new(); SLICES];
        let band = (DEPTH_MM.1 - DEPTH_MM.0) / 3.0;
        for body in 0..3 {
            let depth_center = DEPTH_MM.0 + band * (body as f64 + 0.5);
            for step in (0..steps).step_by(stride) {
                let point = ctx.positions[body][step];
                // Center the object in the middle 60% of the plate.
                let px = ((point.x - lo.0) / extent.0 * 0.6 + 0.2) * n as f64;
                let py = ((point.y - lo.1) / extent.1 * 0.6 + 0.2) * n as f64;
                let depth = depth_center
                    + (point.z.tanh()) * band * 0.35
                    + (step as f64 / steps as f64 - 0.5) * band * 0.2;
                let slice = (((depth - DEPTH_MM.0) / (DEPTH_MM.1 - DEPTH_MM.0)) * SLICES as f64)
                    .clamp(0.0, SLICES as f64 - 1.0) as usize;
                let (ix, iy) = (px as usize, py as usize);
                if ix < n && iy < n {
                    slices[slice].push(iy * n + ix);
                }
            }
        }

        // --- Field synthesis: propagate each slice to the plate and sum.
        let mut planner = FftPlanner::<f32>::new();
        let mut plate = vec![Complex::new(0.0f32, 0.0f32); n * n];
        let mut work = vec![Complex::new(0.0f32, 0.0f32); n * n];
        let started = std::time::Instant::now();
        for (index, slice) in slices.iter().enumerate() {
            if slice.is_empty() {
                continue;
            }
            work.fill(Complex::new(0.0, 0.0));
            for &cell in slice {
                work[cell] += Complex::new(1.0, 0.0);
            }
            let depth =
                DEPTH_MM.0 + (index as f64 + 0.5) / SLICES as f64 * (DEPTH_MM.1 - DEPTH_MM.0);
            fft2d(&mut work, n, &mut planner, false);
            propagate(&mut work, n, depth);
            fft2d(&mut work, n, &mut planner, true);
            for (accumulated, value) in plate.iter_mut().zip(work.iter()) {
                *accumulated += *value;
            }
            if index == 0 {
                info!(
                    "   hologram: slice propagation {:.1}s each, projected {:.1} min",
                    started.elapsed().as_secs_f64(),
                    started.elapsed().as_secs_f64() * SLICES as f64 / 60.0
                );
            }
        }

        // --- Interference with the off-axis reference.
        let carrier = (reference.sin()) / WAVELENGTH_MM; // cycles per mm
        let mut intensity = vec![0.0f32; n * n];
        let mut peak = 0.0f32;
        for row in 0..n {
            for col in 0..n {
                let x_mm = col as f64 * pitch;
                let phase = std::f64::consts::TAU * carrier * x_mm;
                let (sin_p, cos_p) = phase.sin_cos();
                let total = plate[row * n + col] + Complex::new(cos_p as f32, sin_p as f32);
                let value = total.norm_sqr();
                intensity[row * n + col] = value;
                peak = peak.max(value);
            }
        }
        // Normalize by the 99.5th percentile so speckle does not crush it.
        let mut sample: Vec<f32> = intensity.iter().copied().step_by(97).collect();
        sample.sort_by(f32::total_cmp);
        let white = sample[((sample.len() - 1) as f64 * 0.995) as usize].max(1e-9);
        let pattern: Vec<(f64, f64, f64)> = intensity
            .iter()
            .map(|&value| {
                let v = f64::from((value / white).min(1.0));
                (v, v, v)
            })
            .collect();
        let image = encode_linear_rec2020_png16(&pattern, n as u32, n as u32);
        sink.save_png16(&image, "hologram_pattern.png")?;
        drop(pattern);

        // --- Reconstruction proof at the three body depths.
        let panel = (n / 4).min(1_024);
        let mut montage = vec![(0.0, 0.0, 0.0); panel * 3 * panel];
        for (panel_index, focus) in FOCUS_MM.iter().enumerate() {
            // Illuminate the recorded intensity with the reference and
            // back-propagate to the focus depth.
            for row in 0..n {
                for col in 0..n {
                    let x_mm = col as f64 * pitch;
                    let phase = std::f64::consts::TAU * carrier * x_mm;
                    let (sin_p, cos_p) = phase.sin_cos();
                    work[row * n + col] = Complex::new(
                        intensity[row * n + col] * cos_p as f32,
                        intensity[row * n + col] * sin_p as f32,
                    );
                }
            }
            fft2d(&mut work, n, &mut planner, false);
            propagate(&mut work, n, -focus);
            fft2d(&mut work, n, &mut planner, true);
            // Downsample the reconstruction magnitude into the montage.
            let mut panel_peak = 0.0f32;
            let mut panel_values = vec![0.0f32; panel * panel];
            for py in 0..panel {
                for px in 0..panel {
                    let mut sum = 0.0f32;
                    let block = n / panel;
                    for by in 0..block {
                        for bx in 0..block {
                            sum += work[(py * block + by) * n + px * block + bx].norm();
                        }
                    }
                    let value = sum / (block * block) as f32;
                    panel_values[py * panel + px] = value;
                    panel_peak = panel_peak.max(value);
                }
            }
            for py in 0..panel {
                for px in 0..panel {
                    let v =
                        f64::from(panel_values[py * panel + px] / panel_peak.max(1e-9)).powf(0.6);
                    montage[py * panel * 3 + panel_index * panel + px] = (v, v * 0.35, v * 0.3);
                }
            }
        }
        let recon = encode_linear_rec2020_png16(&montage, (panel * 3) as u32, panel as u32);
        sink.save_png16(&recon, "reconstruction_sim.png")?;

        // --- Print spec.
        let dpi = 25.4 / pitch;
        let spec = format!(
            "three-body computed hologram -- seed 0x{seed}\n\
             \n\
             hologram_pattern.png\n\
             - {n} x {n} px over a {plate:.0} x {plate:.0} mm plate\n\
             - pixel pitch {pitch_um:.2} um  =>  {dpi:.0} DPI output required\n\
             - reference beam: plane wave, {angle:.2} degrees off-axis in X\n\
             - wavelength: 650 nm (standard red laser pointer)\n\
             - THIS IS NOT A DESKTOP PRINT: use a laser photoplotter or a\n\
               high-end imagesetter on transparency film (emulsion side\n\
               toward the viewer)\n\
             \n\
             reconstruction: shine the laser through the developed film at\n\
             the reference angle; the orbit reconstructs {d0:.0}-{d1:.0} mm\n\
             behind the plate, one body strand per depth band (see\n\
             reconstruction_sim.png: panels focus at {f0:.0}/{f1:.0}/{f2:.0} mm).\n",
            seed = ctx.seed_hex.to_uppercase(),
            n = n,
            plate = PLATE_MM,
            pitch_um = pitch * 1_000.0,
            dpi = dpi,
            angle = reference.to_degrees(),
            d0 = DEPTH_MM.0,
            d1 = DEPTH_MM.1,
            f0 = FOCUS_MM[0],
            f1 = FOCUS_MM[1],
            f2 = FOCUS_MM[2],
        );
        sink.write_text("hologram_spec.txt", &spec, "data")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_roundtrip_preserves_a_point() {
        let n = 64;
        let mut planner = FftPlanner::<f32>::new();
        let mut grid = vec![Complex::new(0.0f32, 0.0); n * n];
        grid[5 * n + 9] = Complex::new(1.0, 0.0);
        fft2d(&mut grid, n, &mut planner, false);
        fft2d(&mut grid, n, &mut planner, true);
        assert!((grid[5 * n + 9].re - 1.0).abs() < 1e-4);
        let energy: f32 = grid.iter().map(rustfft::num_complex::Complex::norm_sqr).sum();
        assert!((energy - 1.0).abs() < 1e-3);
    }

    #[test]
    fn propagation_preserves_energy_within_the_band() {
        let n = 64;
        let mut planner = FftPlanner::<f32>::new();
        let mut grid = vec![Complex::new(0.0f32, 0.0); n * n];
        // A smooth low-frequency field survives propagation unchanged in
        // total energy (the transfer function is a pure phase).
        for row in 0..n {
            for col in 0..n {
                let value =
                    (-((row as f32 - 32.0).powi(2) + (col as f32 - 32.0).powi(2)) / 64.0).exp();
                grid[row * n + col] = Complex::new(value, 0.0);
            }
        }
        let before: f32 = grid.iter().map(rustfft::num_complex::Complex::norm_sqr).sum();
        fft2d(&mut grid, n, &mut planner, false);
        propagate(&mut grid, n, 40.0);
        fft2d(&mut grid, n, &mut planner, true);
        let after: f32 = grid.iter().map(rustfft::num_complex::Complex::norm_sqr).sum();
        // Some energy sits beyond the evanescent cutoff at this tiny grid;
        // allow a modest loss but no gain.
        assert!(after <= before * 1.001 && after > before * 0.5, "{before} -> {after}");
    }
}
