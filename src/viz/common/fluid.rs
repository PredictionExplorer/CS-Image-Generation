//! Stam stable-fluids solver (master plan II.6): semi-Lagrangian velocity
//! advection on a collocated grid (the GPU-Gems formulation; recorded
//! deviation from the spec's staggered MAC grid), 48 Jacobi pressure
//! iterations, vorticity confinement, and MacCormack-corrected dye advection
//! for crisp marbling interfaces. Deterministic: no randomness, fixed
//! iteration counts, rayon used only over disjoint rows.

use rayon::prelude::*;

/// Number of Jacobi pressure iterations per projection.
const JACOBI_ITERATIONS: usize = 48;
/// Vorticity confinement strength.
const VORTICITY_EPS: f32 = 2.0;

/// A 2D incompressible fluid with three dye fields.
pub struct Fluid {
    /// Grid width in cells.
    pub width: usize,
    /// Grid height in cells.
    pub height: usize,
    u: Vec<f32>,
    v: Vec<f32>,
    u_scratch: Vec<f32>,
    v_scratch: Vec<f32>,
    pressure: Vec<f32>,
    pressure_scratch: Vec<f32>,
    divergence: Vec<f32>,
    /// Per-body dye concentration fields.
    pub dye: [Vec<f32>; 3],
    dye_forward: Vec<f32>,
    dye_backward: Vec<f32>,
    dye_corrected: Vec<f32>,
}

/// Clamped bilinear sample of a scalar grid at (x, y) in cell coordinates.
#[inline]
fn bilinear(field: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, width as f32 - 1.001);
    let y = y.clamp(0.0, height as f32 - 1.001);
    let x0 = x as usize;
    let y0 = y as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let top = field[y0 * width + x0] * (1.0 - fx) + field[y0 * width + x1] * fx;
    let bottom = field[y1 * width + x0] * (1.0 - fx) + field[y1 * width + x1] * fx;
    top * (1.0 - fy) + bottom * fy
}

impl Fluid {
    /// Create a still fluid.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        let cells = width * height;
        Self {
            width,
            height,
            u: vec![0.0; cells],
            v: vec![0.0; cells],
            u_scratch: vec![0.0; cells],
            v_scratch: vec![0.0; cells],
            pressure: vec![0.0; cells],
            pressure_scratch: vec![0.0; cells],
            divergence: vec![0.0; cells],
            dye: [vec![0.0; cells], vec![0.0; cells], vec![0.0; cells]],
            dye_forward: vec![0.0; cells],
            dye_backward: vec![0.0; cells],
            dye_corrected: vec![0.0; cells],
        }
    }

    /// Add a Gaussian-weighted force impulse centered at (x, y).
    pub fn add_force(&mut self, x: f32, y: f32, radius: f32, force_x: f32, force_y: f32) {
        let r_cells = radius.max(1.0);
        let reach = (r_cells * 3.0).ceil() as i64;
        let min_col = ((x - reach as f32).floor().max(0.0)) as usize;
        let max_col = ((x + reach as f32).ceil() as usize).min(self.width - 1);
        let min_row = ((y - reach as f32).floor().max(0.0)) as usize;
        let max_row = ((y + reach as f32).ceil() as usize).min(self.height - 1);
        let inv_two_r_sq = 1.0 / (2.0 * r_cells * r_cells);
        for row in min_row..=max_row {
            for col in min_col..=max_col {
                let dx = col as f32 - x;
                let dy = row as f32 - y;
                let weight = (-(dx * dx + dy * dy) * inv_two_r_sq).exp();
                let index = row * self.width + col;
                self.u[index] += force_x * weight;
                self.v[index] += force_y * weight;
            }
        }
    }

    /// Inject dye into one field in a soft disc.
    pub fn inject_dye(&mut self, field: usize, x: f32, y: f32, radius: f32, amount: f32) {
        let r_cells = radius.max(1.0);
        let reach = (r_cells * 1.5).ceil() as i64;
        let min_col = ((x - reach as f32).floor().max(0.0)) as usize;
        let max_col = ((x + reach as f32).ceil() as usize).min(self.width - 1);
        let min_row = ((y - reach as f32).floor().max(0.0)) as usize;
        let max_row = ((y + reach as f32).ceil() as usize).min(self.height - 1);
        for row in min_row..=max_row {
            for col in min_col..=max_col {
                let dx = col as f32 - x;
                let dy = row as f32 - y;
                let dist = (dx * dx + dy * dy).sqrt();
                let weight = (1.0 - dist / r_cells).clamp(0.0, 1.0);
                self.dye[field][row * self.width + col] += amount * weight;
            }
        }
    }

    /// Semi-Lagrangian advection of `source` into `dest` by (u, v) over dt.
    fn advect_into(
        dest: &mut [f32],
        source: &[f32],
        u: &[f32],
        v: &[f32],
        width: usize,
        height: usize,
        dt: f32,
    ) {
        dest.par_chunks_mut(width).enumerate().for_each(|(row, line)| {
            for (col, slot) in line.iter_mut().enumerate() {
                let index = row * width + col;
                let back_x = col as f32 - dt * u[index];
                let back_y = row as f32 - dt * v[index];
                *slot = bilinear(source, width, height, back_x, back_y);
            }
        });
    }

    /// Vorticity confinement: re-inject small-scale swirl lost to advection.
    fn confine_vorticity(&mut self, dt: f32) {
        let width = self.width;
        let height = self.height;
        // Curl into the divergence scratch (reused between stages).
        let curl = &mut self.divergence;
        curl.par_chunks_mut(width).enumerate().for_each(|(row, line)| {
            for (col, slot) in line.iter_mut().enumerate() {
                let left = col.saturating_sub(1);
                let right = (col + 1).min(width - 1);
                let up = row.saturating_sub(1);
                let down = (row + 1).min(height - 1);
                let dv_dx = (self.v[row * width + right] - self.v[row * width + left]) * 0.5;
                let du_dy = (self.u[down * width + col] - self.u[up * width + col]) * 0.5;
                *slot = dv_dx - du_dy;
            }
        });
        let curl = &self.divergence;
        let u = &mut self.u;
        let v = &mut self.v;
        u.par_chunks_mut(width).zip(v.par_chunks_mut(width)).enumerate().for_each(
            |(row, (u_line, v_line))| {
                for col in 0..width {
                    let left = col.saturating_sub(1);
                    let right = (col + 1).min(width - 1);
                    let up = row.saturating_sub(1);
                    let down = (row + 1).min(height - 1);
                    let grad_x =
                        (curl[row * width + right].abs() - curl[row * width + left].abs()) * 0.5;
                    let grad_y =
                        (curl[down * width + col].abs() - curl[up * width + col].abs()) * 0.5;
                    let norm = (grad_x * grad_x + grad_y * grad_y).sqrt().max(1e-6);
                    let omega = curl[row * width + col];
                    u_line[col] += VORTICITY_EPS * (grad_y / norm) * omega * dt;
                    v_line[col] -= VORTICITY_EPS * (grad_x / norm) * omega * dt;
                }
            },
        );
    }

    /// Pressure projection: 48 Jacobi iterations, then gradient subtraction.
    fn project(&mut self) {
        let width = self.width;
        let height = self.height;
        {
            let u = &self.u;
            let v = &self.v;
            self.divergence.par_chunks_mut(width).enumerate().for_each(|(row, line)| {
                for (col, slot) in line.iter_mut().enumerate() {
                    let left = col.saturating_sub(1);
                    let right = (col + 1).min(width - 1);
                    let up = row.saturating_sub(1);
                    let down = (row + 1).min(height - 1);
                    *slot = 0.5
                        * (u[row * width + right] - u[row * width + left] + v[down * width + col]
                            - v[up * width + col]);
                }
            });
        }
        self.pressure.par_iter_mut().for_each(|p| *p = 0.0);
        for _ in 0..JACOBI_ITERATIONS {
            {
                let pressure = &self.pressure;
                let divergence = &self.divergence;
                self.pressure_scratch.par_chunks_mut(width).enumerate().for_each(|(row, line)| {
                    for (col, slot) in line.iter_mut().enumerate() {
                        let left = col.saturating_sub(1);
                        let right = (col + 1).min(width - 1);
                        let up = row.saturating_sub(1);
                        let down = (row + 1).min(height - 1);
                        *slot = (pressure[row * width + left]
                            + pressure[row * width + right]
                            + pressure[up * width + col]
                            + pressure[down * width + col]
                            - divergence[row * width + col])
                            * 0.25;
                    }
                });
            }
            std::mem::swap(&mut self.pressure, &mut self.pressure_scratch);
        }
        {
            let pressure = &self.pressure;
            self.u.par_chunks_mut(width).zip(self.v.par_chunks_mut(width)).enumerate().for_each(
                |(row, (u_line, v_line))| {
                    for col in 0..width {
                        let left = col.saturating_sub(1);
                        let right = (col + 1).min(width - 1);
                        let up = row.saturating_sub(1);
                        let down = (row + 1).min(height - 1);
                        u_line[col] -=
                            0.5 * (pressure[row * width + right] - pressure[row * width + left]);
                        v_line[col] -=
                            0.5 * (pressure[down * width + col] - pressure[up * width + col]);
                    }
                },
            );
        }
        self.enforce_walls();
    }

    /// Zero the normal velocity on the box walls.
    fn enforce_walls(&mut self) {
        let width = self.width;
        let height = self.height;
        for col in 0..width {
            self.v[col] = 0.0;
            self.v[(height - 1) * width + col] = 0.0;
        }
        for row in 0..height {
            self.u[row * width] = 0.0;
            self.u[row * width + width - 1] = 0.0;
        }
    }

    /// One simulation step: advect velocity, confine vorticity, project,
    /// then MacCormack-advect each dye field with a monotonic limiter.
    pub fn step(&mut self, dt: f32) {
        let width = self.width;
        let height = self.height;

        Self::advect_into(&mut self.u_scratch, &self.u, &self.u, &self.v, width, height, dt);
        Self::advect_into(&mut self.v_scratch, &self.v, &self.u, &self.v, width, height, dt);
        std::mem::swap(&mut self.u, &mut self.u_scratch);
        std::mem::swap(&mut self.v, &mut self.v_scratch);
        self.confine_vorticity(dt);
        self.project();

        for field in 0..3 {
            // Forward pass.
            Self::advect_into(
                &mut self.dye_forward,
                &self.dye[field],
                &self.u,
                &self.v,
                width,
                height,
                dt,
            );
            // Backward pass over the forward result.
            Self::advect_into(
                &mut self.dye_backward,
                &self.dye_forward,
                &self.u,
                &self.v,
                width,
                height,
                -dt,
            );
            // MacCormack correction with a local min/max limiter.
            {
                let source = &self.dye[field];
                let forward = &self.dye_forward;
                let backward = &self.dye_backward;
                let u = &self.u;
                let v = &self.v;
                self.dye_corrected.par_chunks_mut(width).enumerate().for_each(|(row, line)| {
                    for (col, slot) in line.iter_mut().enumerate() {
                        let index = row * width + col;
                        let value = forward[index] + 0.5 * (source[index] - backward[index]);
                        // Limit to the bilinear footprint at the backtrace.
                        let back_x = (col as f32 - dt * u[index]).clamp(0.0, width as f32 - 1.001);
                        let back_y = (row as f32 - dt * v[index]).clamp(0.0, height as f32 - 1.001);
                        let x0 = back_x as usize;
                        let y0 = back_y as usize;
                        let x1 = (x0 + 1).min(width - 1);
                        let y1 = (y0 + 1).min(height - 1);
                        let corners = [
                            source[y0 * width + x0],
                            source[y0 * width + x1],
                            source[y1 * width + x0],
                            source[y1 * width + x1],
                        ];
                        let lo = corners.iter().copied().fold(f32::INFINITY, f32::min);
                        let hi = corners.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        *slot = value.clamp(lo, hi);
                    }
                });
            }
            std::mem::swap(&mut self.dye[field], &mut self.dye_corrected);
        }
    }

    /// Read access to the velocity field (diagnostics, checkpoints).
    #[must_use]
    pub fn velocity(&self) -> (&[f32], &[f32]) {
        (&self.u, &self.v)
    }

    /// Maximum absolute velocity divergence (diagnostics / tests).
    #[must_use]
    pub fn max_divergence(&self) -> f32 {
        let width = self.width;
        let height = self.height;
        let mut worst = 0.0f32;
        for row in 1..height - 1 {
            for col in 1..width - 1 {
                let div = 0.5
                    * (self.u[row * width + col + 1] - self.u[row * width + col - 1]
                        + self.v[(row + 1) * width + col]
                        - self.v[(row - 1) * width + col]);
                worst = worst.max(div.abs());
            }
        }
        worst
    }

    /// Bilinearly upsample the whole state to new dimensions (marbling's
    /// double-resolution tail re-run).
    #[must_use]
    pub fn upsampled(&self, width: usize, height: usize) -> Self {
        let mut out = Self::new(width, height);
        let scale_x = self.width as f32 / width as f32;
        let scale_y = self.height as f32 / height as f32;
        let velocity_scale = width as f32 / self.width as f32;
        let resample = |source: &[f32], dest: &mut [f32], value_scale: f32| {
            dest.par_chunks_mut(width).enumerate().for_each(|(row, line)| {
                let sy = (row as f32 + 0.5) * scale_y - 0.5;
                for (col, slot) in line.iter_mut().enumerate() {
                    let sx = (col as f32 + 0.5) * scale_x - 0.5;
                    *slot = bilinear(source, self.width, self.height, sx, sy) * value_scale;
                }
            });
        };
        // Velocities are in cells/second: scale with resolution.
        let (u, v) = (self.u.clone(), self.v.clone());
        resample(&u, &mut out.u, velocity_scale);
        resample(&v, &mut out.v, velocity_scale);
        for field in 0..3 {
            let dye = self.dye[field].clone();
            resample(&dye, &mut out.dye[field], 1.0);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_suppresses_divergence() {
        // 48 Jacobi iterations converge the smooth component; a concentrated
        // impulse retains some low-frequency residual, so expect a solid
        // reduction rather than elimination (visual-quality target).
        let mut fluid = Fluid::new(96, 64);
        fluid.add_force(30.0, 32.0, 6.0, 40.0, 8.0);
        fluid.add_force(60.0, 30.0, 5.0, -25.0, 18.0);
        let before = fluid.max_divergence();
        fluid.project();
        let after = fluid.max_divergence();
        assert!(
            after < before * 0.6,
            "projection should suppress divergence: {before:.4} -> {after:.4}"
        );
    }

    #[test]
    fn dye_mass_is_approximately_conserved() {
        let mut fluid = Fluid::new(96, 96);
        fluid.inject_dye(0, 48.0, 48.0, 8.0, 1.0);
        let initial: f32 = fluid.dye[0].iter().sum();
        // A gentle interior swirl that keeps the blob away from the walls.
        fluid.add_force(42.0, 48.0, 10.0, 0.0, 8.0);
        fluid.add_force(54.0, 48.0, 10.0, 0.0, -8.0);
        for _ in 0..90 {
            fluid.step(1.0 / 60.0);
        }
        let after: f32 = fluid.dye[0].iter().sum();
        let drift = (after - initial).abs() / initial;
        assert!(drift < 0.05, "dye mass should hold within 5%: drift {drift:.4}");
    }

    #[test]
    fn solver_is_deterministic() {
        let run = || {
            let mut fluid = Fluid::new(64, 48);
            fluid.inject_dye(1, 20.0, 24.0, 6.0, 1.0);
            for frame in 0..60 {
                let angle = frame as f32 * 0.1;
                fluid.add_force(
                    32.0 + 10.0 * angle.cos(),
                    24.0 + 8.0 * angle.sin(),
                    5.0,
                    angle.sin() * 20.0,
                    angle.cos() * 20.0,
                );
                fluid.step(1.0 / 60.0);
            }
            fluid.dye[1].clone()
        };
        let first = run();
        let second = run();
        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "fluid must be bit-deterministic");
        }
    }
}
