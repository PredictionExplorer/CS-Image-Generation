//! Massless test-particle infrastructure shared by V31 `dust-nebula` and
//! V38 `galaxy-collision`: kick-drift-kick integration through the
//! time-varying three-body field, deterministic seeding helpers, and a
//! race-free banded splatter that renders particle strokes with the
//! production spectral rasterizer.

use crate::render::SpectralLineSegment;
use crate::render::context::PixelBuffer;
use crate::render::drawing::draw_line_segment_aa_spectral_rows;
use crate::render::effects::convert_spd_buffer_to_rgba;
use crate::sim::{G, Sha3RandomByteStream};
use crate::spectrum::NUM_BINS;
use nalgebra::Vector3;
use rayon::prelude::*;

/// Conservative vertical pad (pixels) covering the widest production stroke.
const BAND_PAD: f32 = 24.0;

/// A swarm of massless test particles.
pub struct Swarm {
    /// Particle positions (projected world coordinates).
    pub positions: Vec<Vector3<f64>>,
    /// Particle velocities.
    pub velocities: Vec<Vector3<f64>>,
    /// Frozen particles are no longer integrated (escaped the scene).
    pub frozen: Vec<bool>,
    /// Index of the body dominating each particle's acceleration.
    pub dominant_body: Vec<u8>,
}

/// Integration parameters for the three-body test-particle field.
#[derive(Clone, Copy)]
pub struct FieldParams {
    /// Plummer softening length squared (world units squared).
    pub softening_sq: f64,
    /// Particles farther than this from `freeze_center` freeze (squared).
    pub freeze_radius_sq: Option<f64>,
    /// Center for the freeze-radius test.
    pub freeze_center: Vector3<f64>,
}

/// Softened acceleration and dominant body at a point.
#[inline]
fn acceleration(
    point: Vector3<f64>,
    bodies: &[Vector3<f64>; 3],
    masses: [f64; 3],
    softening_sq: f64,
) -> (Vector3<f64>, u8) {
    let mut accel = Vector3::zeros();
    let mut dominant = 0u8;
    let mut dominant_strength = -1.0f64;
    for body in 0..3 {
        let delta = bodies[body] - point;
        let dist_sq = delta.norm_squared() + softening_sq;
        let strength = masses[body] / dist_sq;
        accel += delta * (G * strength / dist_sq.sqrt());
        if strength > dominant_strength {
            dominant_strength = strength;
            dominant = body as u8;
        }
    }
    (accel, dominant)
}

impl Swarm {
    /// Build a swarm from seeded positions and velocities.
    #[must_use]
    pub fn new(positions: Vec<Vector3<f64>>, velocities: Vec<Vector3<f64>>) -> Self {
        let count = positions.len();
        debug_assert_eq!(count, velocities.len());
        Self { positions, velocities, frozen: vec![false; count], dominant_body: vec![0; count] }
    }

    /// Number of particles.
    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the swarm is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// One kick-drift-kick step of size `dt` through the field defined by
    /// the body positions at the start and end of the step.
    pub fn step(
        &mut self,
        bodies_now: &[Vector3<f64>; 3],
        bodies_next: &[Vector3<f64>; 3],
        masses: [f64; 3],
        dt: f64,
        params: &FieldParams,
    ) {
        let half = 0.5 * dt;
        self.positions
            .par_iter_mut()
            .zip(self.velocities.par_iter_mut())
            .zip(self.frozen.par_iter_mut())
            .zip(self.dominant_body.par_iter_mut())
            .for_each(|(((position, velocity), frozen), dominant)| {
                if *frozen {
                    return;
                }
                let (accel_now, _) =
                    acceleration(*position, bodies_now, masses, params.softening_sq);
                *velocity += accel_now * half;
                *position += *velocity * dt;
                let (accel_next, dominant_next) =
                    acceleration(*position, bodies_next, masses, params.softening_sq);
                *velocity += accel_next * half;
                *dominant = dominant_next;
                if let Some(radius_sq) = params.freeze_radius_sq
                    && (*position - params.freeze_center).norm_squared() > radius_sq
                {
                    *frozen = true;
                }
            });
    }
}

/// An SPD canvas with race-free banded parallel splatting: segments are
/// bucketed into horizontal bands by padded y-range and each band renders
/// its bucket with the production rasterizer clamped to its own rows
/// (bit-deterministic regardless of thread count).
pub struct BandedSpd {
    spd: Vec<[f64; NUM_BINS]>,
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    band_rows: usize,
    buckets: Vec<Vec<u32>>,
}

impl BandedSpd {
    /// Create a zeroed banded canvas.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let threads = rayon::current_num_threads().max(1);
        let band_rows = (height as usize).div_ceil(threads * 3).max(32);
        let band_count = (height as usize).div_ceil(band_rows);
        Self {
            spd: vec![[0.0; NUM_BINS]; width as usize * height as usize],
            width,
            height,
            band_rows,
            buckets: vec![Vec::new(); band_count],
        }
    }

    /// Splat a batch of segments in parallel bands.
    pub fn splat(&mut self, segments: &[SpectralLineSegment]) {
        for bucket in &mut self.buckets {
            bucket.clear();
        }
        let band_count = self.buckets.len();
        for (index, segment) in segments.iter().enumerate() {
            let y_min = segment.start.y.min(segment.end.y) - BAND_PAD;
            let y_max = segment.start.y.max(segment.end.y) + BAND_PAD;
            let first = ((y_min.max(0.0) as usize) / self.band_rows).min(band_count - 1);
            let last = ((y_max.max(0.0) as usize) / self.band_rows).min(band_count - 1);
            for band in first..=last {
                self.buckets[band].push(index as u32);
            }
        }

        let width = self.width;
        let height = self.height;
        let band_rows = self.band_rows;
        let pixels_per_band = band_rows * width as usize;
        self.spd.par_chunks_mut(pixels_per_band).zip(self.buckets.par_iter()).enumerate().for_each(
            |(band, (rows, bucket))| {
                let row_start = band * band_rows;
                let row_end = row_start + rows.len() / width as usize;
                for &segment_index in bucket {
                    draw_line_segment_aa_spectral_rows(
                        rows,
                        width,
                        height,
                        row_start,
                        row_end,
                        segments[segment_index as usize],
                    );
                }
            },
        );
    }

    /// Convert the accumulated SPD into a linear RGBA buffer.
    pub fn convert_into(&self, rgba: &mut PixelBuffer) {
        rgba.resize(self.spd.len(), (0.0, 0.0, 0.0, 0.0));
        convert_spd_buffer_to_rgba(&self.spd, rgba, self.width as usize, self.height as usize);
    }

    /// Immutable access to the SPD (composites, audits).
    #[must_use]
    pub fn spd(&self) -> &[[f64; NUM_BINS]] {
        &self.spd
    }
}

/// Standard normal sample (Box-Muller).
fn gaussian(rng: &mut Sha3RandomByteStream) -> f64 {
    let u1 = rng.next_f64().max(1e-12);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Seed a Gaussian annulus of near-virial dust around a center point.
///
/// Radii are normally distributed across `[r_min, r_max]`; speeds are
/// `virial_fraction` of the local circular speed for `total_mass`, in random
/// in-plane directions with a thin vertical scatter.
#[must_use]
pub fn seed_annulus(
    count: usize,
    center: Vector3<f64>,
    r_min: f64,
    r_max: f64,
    total_mass: f64,
    virial_fraction: f64,
    rng: &mut Sha3RandomByteStream,
) -> (Vec<Vector3<f64>>, Vec<Vector3<f64>>) {
    let mut positions = Vec::with_capacity(count);
    let mut velocities = Vec::with_capacity(count);
    let mid = 0.5 * (r_min + r_max);
    let sigma = 0.25 * (r_max - r_min).max(1e-9);
    for _ in 0..count {
        let radius = (mid + sigma * gaussian(rng)).clamp(r_min * 0.75, r_max * 1.25);
        let angle = rng.next_f64() * std::f64::consts::TAU;
        let z = 0.03 * radius * gaussian(rng);
        positions.push(center + Vector3::new(radius * angle.cos(), radius * angle.sin(), z));

        let speed = virial_fraction * (G * total_mass / radius.max(1e-9)).sqrt();
        let direction = rng.next_f64() * std::f64::consts::TAU;
        velocities.push(Vector3::new(
            speed * direction.cos(),
            speed * direction.sin(),
            speed * 0.1 * gaussian(rng),
        ));
    }
    (positions, velocities)
}

/// Seed an exponential disk of stars on circular orbits around a host body.
///
/// The disk lies in the xy plane tilted by `tilt` radians about the x axis;
/// `spin` (+1/-1) sets the rotation sense. Star velocities are the softened
/// circular speed for the host's point mass plus the host's own velocity.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn seed_disk(
    count: usize,
    host_position: Vector3<f64>,
    host_velocity: Vector3<f64>,
    host_mass: f64,
    scale_radius: f64,
    tilt: f64,
    spin: f64,
    softening: f64,
    rng: &mut Sha3RandomByteStream,
) -> (Vec<Vector3<f64>>, Vec<Vector3<f64>>) {
    let mut positions = Vec::with_capacity(count);
    let mut velocities = Vec::with_capacity(count);
    let (sin_tilt, cos_tilt) = tilt.sin_cos();
    let softening_sq = softening * softening;
    for _ in 0..count {
        // Exponential surface density: radius ~ Gamma(2, scale_radius).
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64().max(1e-12);
        let radius = (-scale_radius * (u1.ln() + u2.ln())).min(scale_radius * 8.0);
        let angle = rng.next_f64() * std::f64::consts::TAU;
        let in_plane = (radius * angle.cos(), radius * angle.sin());
        // Tilt about the x axis.
        let local = Vector3::new(in_plane.0, in_plane.1 * cos_tilt, in_plane.1 * sin_tilt);
        positions.push(host_position + local);

        let dist_sq = radius * radius + softening_sq;
        let circular = (G * host_mass * radius * radius / dist_sq.powf(1.5)).max(0.0).sqrt();
        let tangent_plane = (-angle.sin() * spin, angle.cos() * spin);
        let tangent =
            Vector3::new(tangent_plane.0, tangent_plane.1 * cos_tilt, tangent_plane.1 * sin_tilt);
        velocities.push(host_velocity + tangent * circular);
    }
    (positions, velocities)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{LineVertex, draw_line_segment_aa_spectral};

    fn single_mass_field() -> ([Vector3<f64>; 3], [f64; 3]) {
        // One real mass; the others are massless and parked far away.
        let bodies = [Vector3::zeros(), Vector3::new(1e9, 0.0, 0.0), Vector3::new(0.0, 1e9, 0.0)];
        (bodies, [200.0, 0.0, 0.0])
    }

    #[test]
    fn circular_orbit_stays_bounded() {
        let (bodies, masses) = single_mass_field();
        let radius = 50.0;
        let speed = (G * masses[0] / radius).sqrt();
        let mut swarm =
            Swarm::new(vec![Vector3::new(radius, 0.0, 0.0)], vec![Vector3::new(0.0, speed, 0.0)]);
        let params = FieldParams {
            softening_sq: 1e-6,
            freeze_radius_sq: None,
            freeze_center: Vector3::zeros(),
        };
        let dt = 0.001 * (radius / speed);
        for _ in 0..20_000 {
            swarm.step(&bodies, &bodies, masses, dt, &params);
        }
        let final_radius = swarm.positions[0].norm();
        assert!(
            (final_radius - radius).abs() / radius < 0.01,
            "circular orbit should stay near r = {radius}, got {final_radius}"
        );
        assert_eq!(swarm.dominant_body[0], 0);
    }

    #[test]
    fn integration_is_deterministic() {
        let (bodies, masses) = single_mass_field();
        let seed = [7u8; 32];
        let run = || {
            let mut rng = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
            let (positions, velocities) =
                seed_annulus(256, Vector3::zeros(), 40.0, 80.0, masses[0], 0.3, &mut rng);
            let mut swarm = Swarm::new(positions, velocities);
            let params = FieldParams {
                softening_sq: 0.01,
                freeze_radius_sq: Some(1e6),
                freeze_center: Vector3::zeros(),
            };
            for _ in 0..500 {
                swarm.step(&bodies, &bodies, masses, 0.05, &params);
            }
            swarm.positions
        };
        let first = run();
        let second = run();
        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!(a.x.to_bits(), b.x.to_bits());
            assert_eq!(a.y.to_bits(), b.y.to_bits());
            assert_eq!(a.z.to_bits(), b.z.to_bits());
        }
    }

    #[test]
    fn escaped_particles_freeze() {
        let (bodies, masses) = single_mass_field();
        let mut swarm =
            Swarm::new(vec![Vector3::new(10.0, 0.0, 0.0)], vec![Vector3::new(1e4, 0.0, 0.0)]);
        let params = FieldParams {
            softening_sq: 0.01,
            freeze_radius_sq: Some(100.0 * 100.0),
            freeze_center: Vector3::zeros(),
        };
        for _ in 0..10 {
            swarm.step(&bodies, &bodies, masses, 0.01, &params);
        }
        assert!(swarm.frozen[0], "particle far beyond the freeze radius must freeze");
    }

    #[test]
    fn banded_splat_matches_serial_rasterizer() {
        let (width, height) = (192u32, 128u32);
        let segments: Vec<SpectralLineSegment> = (0..300)
            .map(|index| {
                let t = index as f32;
                let x = 8.0 + (t * 7.3) % 176.0;
                let y = 4.0 + (t * 11.7) % 120.0;
                SpectralLineSegment {
                    start: LineVertex { x, y, z: 0.0, color: (0.7, 0.12, 0.03), alpha: 0.5 },
                    end: LineVertex {
                        x: x + 2.0,
                        y: y + 1.5,
                        z: 0.0,
                        color: (0.7, 0.12, 0.03),
                        alpha: 0.5,
                    },
                    hdr_scale: 0.02,
                    thickness_factor: 0.8,
                }
            })
            .collect();

        let mut banded = BandedSpd::new(width, height);
        banded.splat(&segments);

        let mut reference = vec![[0.0f64; NUM_BINS]; width as usize * height as usize];
        for &segment in &segments {
            draw_line_segment_aa_spectral(&mut reference, width, height, segment);
        }

        for (pixel, (lhs, rhs)) in banded.spd().iter().zip(reference.iter()).enumerate() {
            for (bin, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
                assert!(
                    a.to_bits() == b.to_bits(),
                    "banded splat diverged at pixel {pixel} bin {bin}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn disk_seeding_orbits_the_host() {
        let mut rng = Sha3RandomByteStream::new(&[3u8; 32], 100.0, 300.0, 300.0, 1.0);
        let host = Vector3::new(5.0, -2.0, 0.0);
        let (positions, velocities) =
            seed_disk(512, host, Vector3::zeros(), 250.0, 10.0, 0.2, 1.0, 0.5, &mut rng);
        // Angular momentum about the host must be consistently signed.
        let mut positive = 0usize;
        for (position, velocity) in positions.iter().zip(velocities.iter()) {
            let r = position - host;
            let lz = r.x * velocity.y - r.y * velocity.x;
            if lz > 0.0 {
                positive += 1;
            }
        }
        assert!(positive > 500, "disk should rotate coherently: {positive}/512 prograde");
    }
}
