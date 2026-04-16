//! Nebula Dust Clouds post-effect
//!
//! Creates organic, flowing nebula clouds in the background using multi-octave
//! `OpenSimplex2S` noise. The clouds slowly drift and evolve over time, adding
//! atmospheric depth and cosmic beauty without overpowering the trajectories.

use super::{PixelBuffer, PostEffect, PostEffectError};
use nalgebra::Vector3;
use opensimplex2::smooth;
use rayon::prelude::*;

/// Configuration for nebula dust clouds effect
#[derive(Clone, Debug)]
pub struct NebulaCloudConfig {
    /// Overall strength/opacity of the effect (0.0-1.0)
    pub strength: f64,
    /// Number of octaves for multi-scale detail (3-5 recommended)
    pub octaves: usize,
    /// Base frequency for noise (lower = larger features)
    pub base_frequency: f64,
    /// Amplitude reduction per octave (0.5 = each octave half as strong)
    pub persistence: f64,
    /// Frequency increase per octave (2.0 = each octave twice the frequency)
    pub lacunarity: f64,
    /// Color palette for nebula (4 colors that blend smoothly)
    pub colors: [[f64; 3]; 4],
    /// Time scale for animation (controls drift speed in videos)
    pub time_scale: f64,
    /// Seed for noise generator (derived from simulation seed)
    pub noise_seed: i64,
    /// Edge fade distance (0.0 = no fade, 0.2 = fade 20% from edges)
    pub edge_fade: f64,
}

impl Default for NebulaCloudConfig {
    fn default() -> Self {
        Self::standard_mode(1920, 1080, 0)
    }
}

impl NebulaCloudConfig {
    /// Create configuration for special mode with enhanced atmosphere
    #[cfg(test)]
    #[must_use]
    pub fn special_mode(width: usize, height: usize, seed: i32) -> Self {
        let min_dim = width.min(height) as f64;
        Self {
            strength: 0.18, // 18% peak opacity - clearly visible yet elegant
            octaves: 4,     // Rich multi-scale detail
            base_frequency: 0.0014 * (1080.0 / min_dim), // Medium-scale features
            persistence: 0.54, // Natural octave contribution
            lacunarity: 2.15, // Good variation between scales
            colors: [
                [0.12, 0.06, 0.28], // Rich purple - visible and mysterious
                [0.05, 0.20, 0.24], // Vibrant teal - clear complement to gold
                [0.22, 0.05, 0.24], // Vibrant magenta - adds visual interest
                [0.03, 0.08, 0.20], // Deep blue - strong foundation
            ],
            time_scale: 0.0022, // Gentle drift - ~4 units over 30sec @60fps
            noise_seed: i64::from(seed),
            edge_fade: 0.25, // Gentle radial vignette
        }
    }

    /// Create configuration for standard mode (disabled)
    #[must_use]
    pub fn standard_mode(_width: usize, _height: usize, seed: i32) -> Self {
        Self {
            strength: 0.0, // Disabled in standard mode
            octaves: 4,
            base_frequency: 0.001,
            persistence: 0.5,
            lacunarity: 2.0,
            colors: [[0.0, 0.0, 0.0]; 4],
            time_scale: 0.002,
            noise_seed: i64::from(seed),
            edge_fade: 0.0,
        }
    }

    /// Build a cosmic-astrophoto nebula preset. Colors lean toward deep blue,
    /// magenta, and warm orange (Hubble-palette-ish). Suitable when the
    /// Cosmic mood has `enable_nebula_tint` active.
    #[must_use]
    pub fn cosmic_mode(width: usize, height: usize, seed: i32, strength: f64) -> Self {
        let min_dim = width.min(height) as f64;
        Self {
            strength: strength.clamp(0.0, 0.6),
            octaves: 5,
            base_frequency: 0.0012 * (1080.0 / min_dim.max(1.0)),
            persistence: 0.58,
            lacunarity: 2.1,
            colors: [
                [0.02, 0.05, 0.18], // deep blue
                [0.22, 0.04, 0.30], // magenta
                [0.05, 0.25, 0.30], // teal
                [0.38, 0.10, 0.06], // warm ember
            ],
            time_scale: 0.002,
            noise_seed: i64::from(seed),
            edge_fade: 0.20,
        }
    }

    /// Rotate all palette colors in RGB space by shifting hue via a simple
    /// primary-rotation trick: apply a 3x3 rotation matrix that cyclically
    /// mixes (R, G, B) components to shift the apparent hue. `shift` is a
    /// value in `[0, 1]` where 0 is no shift, 1 is a full 120° cycle.
    ///
    /// This is an approximation that works well for wash colors without
    /// needing to round-trip through `OKLab`.
    pub fn rotate_hue(&mut self, shift: f64) {
        let t = shift.clamp(0.0, 1.0) * std::f64::consts::TAU / 3.0;
        let (cs, sn) = (t.cos(), t.sin());
        let k = 1.0 - cs;
        // Axis (1,1,1) unit length is 1/sqrt(3); rotating RGB about it shifts hue
        // while preserving luminance.
        let axis = 1.0 / 3.0_f64.sqrt();
        let ax = axis;
        let ay = axis;
        let az = axis;
        let m00 = cs + ax * ax * k;
        let m01 = ax * ay * k - az * sn;
        let m02 = ax * az * k + ay * sn;
        let m10 = ay * ax * k + az * sn;
        let m11 = cs + ay * ay * k;
        let m12 = ay * az * k - ax * sn;
        let m20 = az * ax * k - ay * sn;
        let m21 = az * ay * k + ax * sn;
        let m22 = cs + az * az * k;
        for c in &mut self.colors {
            let r = c[0] * m00 + c[1] * m01 + c[2] * m02;
            let g = c[0] * m10 + c[1] * m11 + c[2] * m12;
            let b = c[0] * m20 + c[1] * m21 + c[2] * m22;
            c[0] = r.clamp(0.0, 1.0);
            c[1] = g.clamp(0.0, 1.0);
            c[2] = b.clamp(0.0, 1.0);
        }
    }
}

/// Nebula clouds post-effect
pub struct NebulaClouds {
    config: NebulaCloudConfig,
    enabled: bool,
}

impl NebulaClouds {
    /// Create new nebula clouds effect with given configuration
    #[must_use]
    pub fn new(config: NebulaCloudConfig) -> Self {
        let enabled = config.strength > 0.0;
        Self { config, enabled }
    }

    /// Evaluate multi-octave noise at given position and time
    /// Uses `noise3_ImproveXY` which is optimized for time-varied animations
    /// Returns value in [0, 1] range
    #[inline]
    fn evaluate_noise(&self, x: f64, y: f64, time: f64) -> f64 {
        let mut total = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = self.config.base_frequency;
        let mut max_amplitude = 0.0;

        for _ in 0..self.config.octaves {
            // Use noise3_ImproveXY for best visual isotropy in XY plane
            // This is recommended for time-varied animations
            let noise_val = smooth::noise3_ImproveXY(
                self.config.noise_seed,
                x * frequency,
                y * frequency,
                time,
            );
            total += f64::from(noise_val) * amplitude;
            max_amplitude += amplitude;

            amplitude *= self.config.persistence;
            frequency *= self.config.lacunarity;
        }

        // Normalize to [0, 1] accounting for actual amplitude sum
        let normalized = total / max_amplitude;
        (normalized + 1.0) * 0.5
    }

    /// Map noise value to nebula color using smooth interpolation
    #[inline]
    fn noise_to_color(&self, noise_value: f64) -> [f64; 3] {
        // Map noise [0,1] across all 4 colors with smooth cycling
        let scaled = noise_value * 4.0;
        let idx = scaled.floor() as usize;
        let t = scaled.fract();

        let color1 = self.config.colors[idx.min(3)];
        let color2 = self.config.colors[(idx + 1) % 4]; // Wrap to first color

        // Smooth interpolation (cosine for organic transitions)
        let smooth_t = (1.0 - (t * std::f64::consts::PI).cos()) * 0.5;

        [
            color1[0] + (color2[0] - color1[0]) * smooth_t,
            color1[1] + (color2[1] - color1[1]) * smooth_t,
            color1[2] + (color2[2] - color1[2]) * smooth_t,
        ]
    }

    /// Calculate edge fade factor (1.0 at center, fades to 0 near edges)
    /// Uses radial distance for smooth, circular vignette without corner artifacts
    #[inline]
    fn calculate_edge_fade(&self, x: f64, y: f64, width: f64, height: f64) -> f64 {
        if self.config.edge_fade <= 0.0 {
            return 1.0;
        }

        // Calculate radial distance from center (normalized)
        let center_x = width * 0.5;
        let center_y = height * 0.5;
        let dx = (x - center_x) / width;
        let dy = (y - center_y) / height;
        let dist_from_center = (dx * dx + dy * dy).sqrt();

        // Maximum distance (corner to center, normalized)
        let max_dist = 0.5_f64.sqrt(); // sqrt(0.5) ≈ 0.707

        // Fade starts at (1.0 - fade_dist) and reaches 0 at edges
        let fade_start = 1.0 - self.config.edge_fade;
        let normalized_dist = dist_from_center / max_dist;

        if normalized_dist < fade_start {
            1.0
        } else {
            // Smooth fade from fade_start to 1.0 (edge)
            let fade_range = 1.0 - fade_start;
            let fade_amount = (normalized_dist - fade_start) / fade_range;
            (1.0 - fade_amount).clamp(0.0, 1.0)
        }
    }

    /// Process buffer with frame number for time-based animation
    pub fn process_with_time(
        &self,
        buffer: &PixelBuffer,
        width: usize,
        height: usize,
        frame_number: usize,
    ) -> Result<PixelBuffer, PostEffectError> {
        if !self.enabled {
            return Ok(buffer.clone());
        }

        let mut result = buffer.clone();

        // Calculate time offset for smooth animation
        // At 60fps over 30 seconds: frame_number goes 0 to 1800
        // time_scale = 0.0022 means time goes from 0 to ~4.0 over the animation
        let time = frame_number as f64 * self.config.time_scale;

        let width_f = width as f64;
        let height_f = height as f64;

        // Process in parallel for maximum performance
        result.par_iter_mut().enumerate().for_each(|(idx, pixel)| {
            let x = (idx % width) as f64;
            let y = (idx / width) as f64;

            // Evaluate primary noise for color selection
            let noise_value = self.evaluate_noise(x, y, time);

            // Map to nebula color
            let nebula_color = self.noise_to_color(noise_value);

            // Calculate base opacity with edge fade
            let edge_fade = self.calculate_edge_fade(x, y, width_f, height_f);
            let base_opacity = self.config.strength * edge_fade;

            // Secondary noise layer for organic opacity variation
            // Use different seed offset and parameters for decorrelation
            let opacity_noise = self.evaluate_noise(
                x * 1.41 + 1234.56, // Offset and slightly different scale
                y * 1.41 + 6789.01,
                time * 0.77, // Different time rate for layered motion
            );

            // Opacity variation: [0.60, 1.40] - creates wispy, organic structure
            let opacity_variation = 0.60 + opacity_noise * 0.80;
            let final_opacity = base_opacity * opacity_variation;

            // Apply nebula as pure RGB color with coverage (straight alpha, not premultiplied)
            // This will be composited UNDER the trajectories, so no alpha tricks needed
            pixel.0 = nebula_color[0];
            pixel.1 = nebula_color[1];
            pixel.2 = nebula_color[2];
            pixel.3 = final_opacity; // Alpha represents nebula coverage
        });

        Ok(result)
    }
}

#[inline]
fn lerp3(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    let t = t.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn volumetric_palette(config: &NebulaCloudConfig, t: f64) -> [f64; 3] {
    let t = t.clamp(0.0, 1.0);
    if t < 0.33 {
        lerp3(config.colors[0], config.colors[1], t / 0.33)
    } else if t < 0.66 {
        lerp3(config.colors[1], config.colors[2], (t - 0.33) / 0.33)
    } else {
        lerp3(config.colors[2], config.colors[3], (t - 0.66) / 0.34)
    }
}

fn trilinear_sample(grid: &[f64], g: usize, nx: f64, ny: f64, nz: f64) -> f64 {
    let fx = nx * (g - 1) as f64;
    let fy = ny * (g - 1) as f64;
    let fz = nz * (g - 1) as f64;
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let z0 = fz.floor() as usize;
    let x1 = (x0 + 1).min(g - 1);
    let y1 = (y0 + 1).min(g - 1);
    let z1 = (z0 + 1).min(g - 1);
    let tx = fx - x0 as f64;
    let ty = fy - y0 as f64;
    let tz = fz - z0 as f64;
    let idx = |x: usize, y: usize, z: usize| grid[z * g * g + y * g + x];
    let c000 = idx(x0, y0, z0);
    let c100 = idx(x1, y0, z0);
    let c010 = idx(x0, y1, z0);
    let c110 = idx(x1, y1, z0);
    let c001 = idx(x0, y0, z1);
    let c101 = idx(x1, y0, z1);
    let c011 = idx(x0, y1, z1);
    let c111 = idx(x1, y1, z1);
    let c00 = c000 * (1.0 - tx) + c100 * tx;
    let c10 = c010 * (1.0 - tx) + c110 * tx;
    let c01 = c001 * (1.0 - tx) + c101 * tx;
    let c11 = c011 * (1.0 - tx) + c111 * tx;
    let c0 = c00 * (1.0 - ty) + c10 * ty;
    let c1 = c01 * (1.0 - ty) + c11 * ty;
    c0 * (1.0 - tz) + c1 * tz
}

/// Volumetric nebula: sparse `G³` density from trajectory samples, trilinear-sampled along Z rays.
///
/// `step_end` limits how much of the timeline seeds the volume (grows over a video).
#[must_use]
pub fn render_volumetric_trajectory_nebula(
    width: usize,
    height: usize,
    frame_number: usize,
    step_end: usize,
    positions: &[Vec<Vector3<f64>>],
    config: &NebulaCloudConfig,
) -> PixelBuffer {
    const G: usize = 40;
    let te = step_end.min(positions[0].len());
    if te == 0 || width == 0 || height == 0 {
        return vec![(0.0, 0.0, 0.0, 0.0); width * height];
    }

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut min_z = f64::INFINITY;
    let mut max_z = f64::NEG_INFINITY;
    for body_positions in positions.iter().take(3) {
        for p in body_positions.iter().take(te) {
            min_x = min_x.min(p.x);
            max_x = max_x.max(p.x);
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
            min_z = min_z.min(p.z);
            max_z = max_z.max(p.z);
        }
    }
    let bx = (max_x - min_x).max(1e-6);
    let by = (max_y - min_y).max(1e-6);
    let bz = (max_z - min_z).max(1e-6);
    let pad = (bx + by + bz) / 3.0 * 0.1 + 1e-3;

    let mut grid = vec![0.0f64; G * G * G];
    let stride = (te / 5_000).max(1);
    for (body, body_positions) in positions.iter().take(3).enumerate() {
        let wbody = 1.0 + 0.15 * body as f64;
        for t in (0..te).step_by(stride) {
            let p = body_positions[t];
            let nx = ((p.x - min_x + pad) / (bx + 2.0 * pad)).clamp(0.0, 1.0);
            let ny = ((p.y - min_y + pad) / (by + 2.0 * pad)).clamp(0.0, 1.0);
            let nz = ((p.z - min_z + pad) / (bz + 2.0 * pad)).clamp(0.0, 1.0);
            let ix = (nx * (G - 1) as f64).round() as usize;
            let iy = (ny * (G - 1) as f64).round() as usize;
            let iz = (nz * (G - 1) as f64).round() as usize;
            let ix = ix.min(G - 1);
            let iy = iy.min(G - 1);
            let iz = iz.min(G - 1);
            grid[iz * G * G + iy * G + ix] += wbody;
        }
    }

    let max_d = grid.iter().copied().fold(0.0_f64, f64::max).max(1e-9);
    let time = frame_number as f64 * 0.02;

    (0..height * width)
        .into_par_iter()
        .map(|idx| {
            let px = idx % width;
            let py = idx / width;
            let u = (px as f64 + 0.5) / width as f64;
            let v = (py as f64 + 0.5) / height as f64;
            let wx = min_x - pad + u * (bx + 2.0 * pad);
            let wy = min_y - pad + v * (by + 2.0 * pad);

            let nv = f64::from(smooth::noise3_ImproveXY(
                config.noise_seed,
                u * 3.0,
                v * 3.0,
                time,
            ));
            let nv = (nv + 1.0) * 0.5;

            const STEPS: usize = 12;
            let mut acc = 0.0f64;
            for s in 0..STEPS {
                let tz = (s as f64 + 0.5) / STEPS as f64;
                let wz = min_z - pad + tz * (bz + 2.0 * pad);
                let nx = ((wx - min_x + pad) / (bx + 2.0 * pad)).clamp(0.0, 1.0);
                let ny = ((wy - min_y + pad) / (by + 2.0 * pad)).clamp(0.0, 1.0);
                let nz = ((wz - min_z + pad) / (bz + 2.0 * pad)).clamp(0.0, 1.0);
                acc += trilinear_sample(&grid, G, nx, ny, nz);
            }
            acc /= STEPS as f64;
            let dens = (acc / max_d).clamp(0.0, 1.0);
            let mix = (0.55 * dens + 0.45 * nv).clamp(0.0, 1.0);
            let rgb = volumetric_palette(config, mix);

            let center_x = width as f64 * 0.5;
            let center_y = height as f64 * 0.5;
            let dx = (px as f64 - center_x) / width as f64;
            let dy = (py as f64 - center_y) / height as f64;
            let dist = (dx * dx + dy * dy).sqrt() / 0.707;
            let edge = if config.edge_fade > 0.0 {
                let fade_start = 1.0 - config.edge_fade;
                if dist < fade_start {
                    1.0
                } else {
                    (1.0 - (dist - fade_start) / config.edge_fade.max(1e-6)).clamp(0.0, 1.0)
                }
            } else {
                1.0
            };

            let opacity = config.strength * edge * (0.25 + 0.75 * dens);
            (rgb[0], rgb[1], rgb[2], opacity.clamp(0.0, 1.0))
        })
        .collect()
}

impl PostEffect for NebulaClouds {
    fn process(
        &self,
        buffer: &PixelBuffer,
        width: usize,
        height: usize,
    ) -> Result<PixelBuffer, PostEffectError> {
        // For static images, use frame 0
        self.process_with_time(buffer, width, height, 0)
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_evaluation() {
        let config = NebulaCloudConfig::special_mode(1920, 1080, 42);
        let nebula = NebulaClouds::new(config);

        // Noise should be in [0, 1] range
        let noise = nebula.evaluate_noise(100.0, 100.0, 0.0);
        assert!((0.0..=1.0).contains(&noise), "Noise value out of range: {noise}");

        // Different positions should give different values
        let noise2 = nebula.evaluate_noise(500.0, 500.0, 0.0);
        assert_ne!(noise, noise2);
    }

    #[test]
    fn test_color_mapping() {
        let config = NebulaCloudConfig::special_mode(1920, 1080, 42);
        let nebula = NebulaClouds::new(config);

        // Test color interpolation at different noise values
        let color_0 = nebula.noise_to_color(0.0);
        let color_quarter = nebula.noise_to_color(0.25);
        let color_mid = nebula.noise_to_color(0.5);
        let color_three_quarter = nebula.noise_to_color(0.75);
        let color_1 = nebula.noise_to_color(1.0);

        // All colors should be valid RGB
        for color in [color_0, color_quarter, color_mid, color_three_quarter, color_1] {
            assert!(color[0] >= 0.0 && color[0] <= 1.0, "R out of range");
            assert!(color[1] >= 0.0 && color[1] <= 1.0, "G out of range");
            assert!(color[2] >= 0.0 && color[2] <= 1.0, "B out of range");
        }

        // Colors should vary
        assert_ne!(color_0, color_mid);
        assert_ne!(color_mid, color_1);
    }

    #[test]
    fn test_edge_fade() {
        let config = NebulaCloudConfig::special_mode(1920, 1080, 42);
        let nebula = NebulaClouds::new(config);

        // Center should have full opacity
        let center_fade = nebula.calculate_edge_fade(960.0, 540.0, 1920.0, 1080.0);
        assert!(center_fade > 0.98, "Center fade too low: {center_fade}");

        // Edges should fade significantly
        let corner_fade = nebula.calculate_edge_fade(0.0, 0.0, 1920.0, 1080.0);
        assert!(corner_fade < 0.2, "Corner fade too high: {corner_fade}");
    }

    #[test]
    fn test_disabled_mode() {
        let config = NebulaCloudConfig::standard_mode(1920, 1080, 0);
        assert_eq!(config.strength, 0.0);

        let nebula = NebulaClouds::new(config);
        assert!(!nebula.is_enabled());
    }

    #[test]
    fn test_determinism() {
        let config = NebulaCloudConfig::special_mode(1920, 1080, 12345);
        let nebula1 = NebulaClouds::new(config.clone());
        let nebula2 = NebulaClouds::new(config);

        // Same seed and coordinates should produce identical noise
        let noise1 = nebula1.evaluate_noise(100.0, 200.0, 1.0);
        let noise2 = nebula2.evaluate_noise(100.0, 200.0, 1.0);
        assert_eq!(noise1, noise2, "Determinism failed!");
    }

    #[test]
    fn test_time_animation() {
        let config = NebulaCloudConfig::special_mode(1920, 1080, 42);
        let nebula = NebulaClouds::new(config);

        // Different times should produce different noise
        let noise_t0 = nebula.evaluate_noise(100.0, 100.0, 0.0);
        let noise_t1 = nebula.evaluate_noise(100.0, 100.0, 1.0);
        let noise_t10 = nebula.evaluate_noise(100.0, 100.0, 10.0);

        // All should be different (time evolution working)
        assert_ne!(noise_t0, noise_t1, "Time animation not working!");
        assert_ne!(noise_t0, noise_t10, "Time animation not working!");

        // Small time step should produce small change (continuity)
        let small_diff = (noise_t1 - noise_t0).abs();
        assert!(small_diff < 0.5, "Noise changing too fast over small time: {small_diff}");
    }

    #[test]
    fn test_volumetric_nebula_opacity_bounded() {
        let n = 64;
        let positions: Vec<Vec<Vector3<f64>>> = (0i32..3)
            .map(|body| {
                (0..n)
                    .map(|t| {
                        let ang = (t as f64) * std::f64::consts::TAU / n as f64
                            + f64::from(body) * 0.3;
                        Vector3::new(ang.cos(), ang.sin(), 0.1 * t as f64 / n as f64)
                    })
                    .collect()
            })
            .collect();
        let cfg = NebulaCloudConfig::special_mode(64, 36, 7);
        let buf = render_volumetric_trajectory_nebula(64, 36, 3, n, &positions, &cfg);
        assert_eq!(buf.len(), 64 * 36);
        for &(r, g, b, a) in &buf {
            assert!((0.0..=1.0).contains(&a), "alpha out of range {a}");
            assert!(r.is_finite() && g.is_finite() && b.is_finite());
        }
        let any_non_zero = buf.iter().any(|&(_, _, _, a)| a > 0.0);
        assert!(any_non_zero, "volumetric nebula should produce density somewhere");
    }

    #[test]
    fn test_volumetric_nebula_empty_timeline() {
        let positions = vec![vec![Vector3::zeros(); 2]; 3];
        let cfg = NebulaCloudConfig::special_mode(16, 16, 1);
        let buf = render_volumetric_trajectory_nebula(16, 16, 0, 0, &positions, &cfg);
        assert!(buf.iter().all(|&(r, g, b, a)| r == 0.0 && g == 0.0 && b == 0.0 && a == 0.0));
    }

    #[test]
    fn test_buffer_processing() {
        let config = NebulaCloudConfig::special_mode(100, 100, 42);
        let nebula = NebulaClouds::new(config);

        // Create test buffer (all black)
        let buffer = vec![(0.0, 0.0, 0.0, 0.0); 10000];
        let result =
            nebula.process(&buffer, 100, 100).expect("nebula clouds process should succeed");

        // Result should have nebula colors added
        let has_color = result.iter().any(|&(r, g, b, _a)| r > 0.0 || g > 0.0 || b > 0.0);
        assert!(has_color, "Nebula should add color to black background!");
    }
}
