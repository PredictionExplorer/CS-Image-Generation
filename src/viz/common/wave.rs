//! 2D scalar wave equation (master plan II.7): leapfrog integration of
//! `u_tt = c^2 lap(u) - gamma u_t` with automatic CFL substepping, a 48-cell
//! absorbing sponge boundary, and soft Gaussian oscillator sources. Fully
//! deterministic.

use rayon::prelude::*;

/// CFL number per substep (c * dt <= CFL cells).
const CFL: f32 = 0.45;
/// Sponge boundary width in cells.
const SPONGE_CELLS: f32 = 48.0;
/// Sponge absorption strength (per-substep exponential factor at the edge).
const SPONGE_STRENGTH: f32 = 0.12;
/// Source injection radius in cells.
const SOURCE_RADIUS: f32 = 2.2;

/// One oscillating point source.
#[derive(Clone, Copy)]
pub struct Oscillator {
    /// Source x in cells.
    pub x: f32,
    /// Source y in cells.
    pub y: f32,
    /// Drive amplitude.
    pub amplitude: f32,
    /// Frequency in cycles per second.
    pub frequency: f32,
}

/// A damped scalar wave field with absorbing boundaries.
pub struct WaveField {
    /// Grid width in cells.
    pub width: usize,
    /// Grid height in cells.
    pub height: usize,
    current: Vec<f32>,
    previous: Vec<f32>,
    next: Vec<f32>,
    /// Boundary-band absorption factors: (cell index, factor < 1).
    sponge_cells: Vec<(u32, f32)>,
    /// Wave speed in cells per second.
    pub wave_speed: f32,
    /// Velocity damping gamma (1/s).
    pub damping: f32,
}

impl WaveField {
    /// Create a still field.
    #[must_use]
    pub fn new(width: usize, height: usize, wave_speed: f32, damping: f32) -> Self {
        // Exponential absorption ramp toward the boundary (band cells only).
        let mut sponge_cells = Vec::new();
        for row in 0..height {
            for col in 0..width {
                let edge = (col.min(width - 1 - col).min(row).min(height - 1 - row)) as f32;
                if edge < SPONGE_CELLS {
                    let depth = (SPONGE_CELLS - edge) / SPONGE_CELLS;
                    let factor = (-SPONGE_STRENGTH * depth * depth).exp();
                    sponge_cells.push(((row * width + col) as u32, factor));
                }
            }
        }
        Self {
            width,
            height,
            current: vec![0.0; width * height],
            previous: vec![0.0; width * height],
            next: vec![0.0; width * height],
            sponge_cells,
            wave_speed,
            damping,
        }
    }

    /// The live field.
    #[must_use]
    pub fn field(&self) -> &[f32] {
        &self.current
    }

    /// Bilinear sample of the live field.
    #[must_use]
    pub fn value_at(&self, x: f32, y: f32) -> f32 {
        let x = x.clamp(0.0, self.width as f32 - 1.001);
        let y = y.clamp(0.0, self.height as f32 - 1.001);
        let x0 = x as usize;
        let y0 = y as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let top = self.current[y0 * self.width + x0] * (1.0 - fx)
            + self.current[y0 * self.width + x1] * fx;
        let bottom = self.current[y1 * self.width + x0] * (1.0 - fx)
            + self.current[y1 * self.width + x1] * fx;
        top * (1.0 - fy) + bottom * fy
    }

    /// One leapfrog substep of size `dt` seconds.
    fn substep(&mut self, dt: f32) {
        let width = self.width;
        let height = self.height;
        let courant_sq = (self.wave_speed * dt) * (self.wave_speed * dt);
        let half_damp = 0.5 * self.damping * dt;
        {
            let current = &self.current;
            let previous = &self.previous;
            self.next.par_chunks_mut(width).enumerate().for_each(|(row, line)| {
                let up = row.saturating_sub(1);
                let down = (row + 1).min(height - 1);
                for (col, slot) in line.iter_mut().enumerate() {
                    let left = col.saturating_sub(1);
                    let right = (col + 1).min(width - 1);
                    let center = current[row * width + col];
                    let laplacian = current[row * width + left]
                        + current[row * width + right]
                        + current[up * width + col]
                        + current[down * width + col]
                        - 4.0 * center;
                    *slot = (2.0 * center - previous[row * width + col] * (1.0 - half_damp)
                        + courant_sq * laplacian)
                        / (1.0 + half_damp);
                }
            });
        }
        // Absorb in the boundary band: damp the leapfrog pair consistently
        // (damping only one buffer acts as an impedance step and reflects).
        for &(index, factor) in &self.sponge_cells {
            self.next[index as usize] *= factor;
            self.current[index as usize] *= factor;
        }
        // Rotate buffers: previous <- current <- next.
        std::mem::swap(&mut self.previous, &mut self.current);
        std::mem::swap(&mut self.current, &mut self.next);
    }

    /// Inject oscillator drives at absolute time `time` for substep `dt`.
    fn inject(&mut self, oscillators: &[Oscillator], time: f32, dt: f32) {
        for source in oscillators {
            let value = source.amplitude
                * (std::f32::consts::TAU * source.frequency * time).sin()
                * dt
                * 60.0;
            let reach = (SOURCE_RADIUS * 2.0).ceil() as i64;
            let min_col = ((source.x - reach as f32).floor().max(0.0)) as usize;
            let max_col = ((source.x + reach as f32).ceil() as usize).min(self.width - 1);
            let min_row = ((source.y - reach as f32).floor().max(0.0)) as usize;
            let max_row = ((source.y + reach as f32).ceil() as usize).min(self.height - 1);
            let inv_two_r_sq = 1.0 / (2.0 * SOURCE_RADIUS * SOURCE_RADIUS);
            for row in min_row..=max_row {
                for col in min_col..=max_col {
                    let dx = col as f32 - source.x;
                    let dy = row as f32 - source.y;
                    let weight = (-(dx * dx + dy * dy) * inv_two_r_sq).exp();
                    self.current[row * self.width + col] += value * weight;
                }
            }
        }
    }

    /// Advance one frame of `dt_frame` seconds with CFL-safe substeps,
    /// driving the given oscillators (phases follow absolute `time`).
    /// Returns the number of substeps taken.
    pub fn advance_frame(&mut self, dt_frame: f32, oscillators: &[Oscillator], time: f32) -> usize {
        let substeps = ((self.wave_speed * dt_frame / CFL).ceil() as usize).max(1);
        let dt = dt_frame / substeps as f32;
        for sub in 0..substeps {
            self.inject(oscillators, time + sub as f32 * dt, dt);
            self.substep(dt);
        }
        substeps
    }

    /// Accumulate `|u|^2` into an envelope buffer of the same dimensions.
    pub fn accumulate_envelope(&self, envelope: &mut [f32]) {
        envelope
            .par_iter_mut()
            .zip(self.current.par_iter())
            .for_each(|(slot, &value)| *slot += value * value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Radius of the strongest ring along the +x axis from the center.
    fn ring_radius(field: &WaveField, center: (usize, usize)) -> f32 {
        let mut best = (0usize, 0.0f32);
        for offset in 4..field.width / 2 - 2 {
            let value = field.field()[center.1 * field.width + center.0 + offset].abs();
            if value > best.1 {
                best = (offset, value);
            }
        }
        best.0 as f32
    }

    #[test]
    fn pulse_propagates_at_the_wave_speed() {
        let size = 400usize;
        let mut field = WaveField::new(size, size, 120.0, 0.0);
        // One sharp central pulse.
        let center = size / 2;
        field.current[center * size + center] = 8.0;
        let dt_frame = 1.0 / 60.0;
        let frames = 60; // one second of travel
        for _ in 0..frames {
            field.advance_frame(dt_frame, &[], 0.0);
        }
        let radius = ring_radius(&field, (center, center));
        let expected = 120.0; // cells after one second
        assert!(
            (radius - expected).abs() / expected < 0.15,
            "ring should travel at c: radius {radius}, expected ~{expected}"
        );
    }

    #[test]
    fn sponge_absorbs_boundary_reflections() {
        let size = 256usize;
        let mut field = WaveField::new(size, size, 150.0, 0.0);
        // Smooth Gaussian pulse (a delta pulse leaves dispersive residue
        // that would mask the reflection measurement).
        let center = size / 2;
        let peak_initial = 8.0f32;
        for row in 0..size {
            for col in 0..size {
                let dx = col as f32 - center as f32;
                let dy = row as f32 - center as f32;
                let value = peak_initial * (-(dx * dx + dy * dy) / 18.0).exp();
                field.current[row * size + col] = value;
                field.previous[row * size + col] = value;
            }
        }
        // Long enough for the wave to reach the walls and bounce back twice.
        for _ in 0..240 {
            field.advance_frame(1.0 / 60.0, &[], 0.0);
        }
        let interior_peak = (size / 4..3 * size / 4)
            .flat_map(|row| (size / 4..3 * size / 4).map(move |col| (row, col)))
            .map(|(row, col)| field.field()[row * size + col].abs())
            .fold(0.0f32, f32::max);
        assert!(
            interior_peak < peak_initial * 0.03,
            "reflections should be absorbed: interior peak {interior_peak}"
        );
    }

    #[test]
    fn wave_field_is_deterministic() {
        let run = || {
            let mut field = WaveField::new(160, 120, 100.0, 0.25);
            let oscillators = [
                Oscillator { x: 60.0, y: 60.0, amplitude: 1.0, frequency: 3.0 },
                Oscillator { x: 100.0, y: 55.0, amplitude: 0.8, frequency: 4.2 },
            ];
            let mut envelope = vec![0.0f32; 160 * 120];
            for frame in 0..90 {
                field.advance_frame(1.0 / 60.0, &oscillators, frame as f32 / 60.0);
                field.accumulate_envelope(&mut envelope);
            }
            (field.current.clone(), envelope)
        };
        let (field_a, envelope_a) = run();
        let (field_b, envelope_b) = run();
        for (a, b) in field_a.iter().zip(field_b.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for (a, b) in envelope_a.iter().zip(envelope_b.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}
