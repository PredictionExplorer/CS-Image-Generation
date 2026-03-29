//! Gauss-map camera — the camera's position on the viewing sphere follows the
//! plane normal's trajectory on S², with close-encounter zoom bursts, look-at
//! wander toward interacting pairs, and physics-driven roll from the normal's
//! azimuthal precession.
//!
//! During stable orbital phases the normal barely moves and the camera drifts
//! gently; during chaotic tumbles the normal sweeps across the sphere and the
//! camera follows dramatically.  Close encounters trigger quadratic zoom-in
//! bursts while the focus point shifts toward the interacting pair.

use crate::utils::{angular_momentum_vec, plane_normal_at};
use nalgebra::Vector3;
use std::f64::consts::PI;
use tracing::info;

use super::context::BoundingBox;

// ─── Configuration ───────────────────────────────────────────

/// Tuning knobs for the Gauss-map camera.
#[derive(Clone, Debug)]
pub struct Camera3DConfig {
    /// Angle between the plane normal and the camera direction (radians).
    /// ~1.13 rad (65°) — enough depth parallax while still seeing the triangle.
    pub view_offset_angle: f64,
    /// Slow background revolutions of the offset direction around the normal.
    /// Provides angular variety even when the normal is nearly constant.
    pub drift_revolutions: f64,
    /// Vertical field-of-view (degrees).
    pub fov_degrees: f64,
    /// How aggressively close encounters pull the camera in (0–1).
    pub zoom_burst_strength: f64,
    /// How much the look-at point shifts toward the closest pair (0–1).
    pub focus_wander_strength: f64,
    /// EMA time-constant for the smoothed normal (fraction of total steps).
    pub normal_tau_frac: f64,
    /// EMA time-constant for the look-at / focus point.
    pub target_tau_frac: f64,
    /// EMA time-constant for the camera distance.
    pub distance_tau_frac: f64,
    /// Base distance as a multiple of the system's max pairwise extent.
    pub distance_scale: f64,
}

impl Default for Camera3DConfig {
    fn default() -> Self {
        Self {
            view_offset_angle: 1.13,
            drift_revolutions: 0.5,
            fov_degrees: 50.0,
            zoom_burst_strength: 0.50,
            focus_wander_strength: 0.30,
            normal_tau_frac: 0.015,
            target_tau_frac: 0.005,
            distance_tau_frac: 0.008,
            distance_scale: 3.0,
        }
    }
}

// ─── Internal pose ───────────────────────────────────────────

struct CameraPose {
    eye: Vector3<f64>,
    target: Vector3<f64>,
    up: Vector3<f64>,
}

// ─── Public API ──────────────────────────────────────────────

/// Pre-computed Gauss-map camera trajectory.
pub struct Camera3D {
    poses: Vec<CameraPose>,
    half_fov_tan: f64,
    depth_scale: f64,
}

impl Camera3D {
    /// Build the full camera trajectory from simulation positions.
    pub fn new(config: &Camera3DConfig, positions: &[Vec<Vector3<f64>>]) -> Self {
        let total_steps = positions[0].len();
        let half_fov_tan = (config.fov_degrees.to_radians() * 0.5).tan();

        // ── Angular-momentum axis (conserved → compute once) ──
        let l_step = 1.min(total_steps.saturating_sub(1));
        let l_hat = {
            let l = angular_momentum_vec(positions, l_step);
            let n = l.norm();
            if n > 1e-14 { l / n } else { Vector3::new(0.0, 0.0, 1.0) }
        };

        // Stable reference vectors for fallbacks
        let aux = if l_hat[0].abs() < 0.9 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        let e_x = aux.cross(&l_hat).normalize();

        // ── Global extents ──
        let max_extent = max_pairwise_distance(positions).max(1e-6);
        let base_distance = config.distance_scale * max_extent;
        let depth_scale = 50.0 / max_extent;

        // ── Per-step raw normals ──
        let mut normals = Vec::with_capacity(total_steps);
        for s in 0..total_steps {
            normals.push(oriented_normal(positions, s, &l_hat));
        }

        // Smooth the normal vectors
        let n_alpha = ema_alpha(config.normal_tau_frac, total_steps);
        ema_smooth_vec(&mut normals, n_alpha);
        // Re-normalize after smoothing (EMA of unit vectors isn't unit)
        for n in &mut normals {
            let len = n.norm();
            if len > 1e-14 {
                *n /= len;
            }
        }

        // ── Per-step raw distance + focus ──
        let mut dist = vec![0.0_f64; total_steps];
        let mut focus = vec![Vector3::zeros(); total_steps];

        for step in 0..total_steps {
            let centroid =
                (positions[0][step] + positions[1][step] + positions[2][step]) / 3.0;

            // Close-encounter zoom burst (quadratic response)
            let min_pair = min_pairwise_distance_at(positions, step);
            let closeness = 1.0 - (min_pair / max_extent).clamp(0.0, 1.0);
            let zoom_burst = closeness * closeness;
            dist[step] = base_distance * (1.0 - config.zoom_burst_strength * zoom_burst);

            // Look-at wander toward closest pair
            let closest_mid = midpoint_of_closest_pair(positions, step);
            let wander = closeness * closeness * config.focus_wander_strength;
            focus[step] = centroid * (1.0 - wander) + closest_mid * wander;
        }

        // Smooth distance and focus
        let dist_a = ema_alpha(config.distance_tau_frac, total_steps);
        let tgt_a = ema_alpha(config.target_tau_frac, total_steps);
        ema_smooth(&mut dist, dist_a);
        ema_smooth_vec(&mut focus, tgt_a);

        // ── Build camera poses ──
        let drift_rate = 2.0 * PI * config.drift_revolutions / total_steps.max(1) as f64;

        let poses: Vec<CameraPose> = (0..total_steps)
            .map(|s| {
                let n = normals[s];

                // Perpendicular to the normal for the offset rotation axis
                let perp = if n.cross(&e_x).norm() > 0.3 {
                    n.cross(&e_x).normalize()
                } else {
                    n.cross(&l_hat).normalize()
                };

                // Drift: slowly rotate the offset axis around the normal
                let drift_phase = drift_rate * s as f64;
                let rot_axis = rodrigues(&perp, &n, drift_phase);

                // Camera direction: normal rotated away by the offset angle
                let cam_dir = rodrigues(&n, &rot_axis, config.view_offset_angle);

                let eye = focus[s] + dist[s] * cam_dir;

                // Physics-driven roll: project the smoothed normal onto the
                // image plane to determine "up"
                let cam_fwd = (focus[s] - eye).normalize();
                let n_proj = n - n.dot(&cam_fwd) * cam_fwd;
                let up = if n_proj.norm() > 0.1 {
                    n_proj.normalize()
                } else {
                    let l_proj = l_hat - l_hat.dot(&cam_fwd) * cam_fwd;
                    if l_proj.norm() > 0.1 { l_proj.normalize() } else { e_x }
                };

                CameraPose { eye, target: focus[s], up }
            })
            .collect();

        info!(
            steps = total_steps,
            max_extent,
            base_distance,
            "Gauss-map camera trajectory built"
        );

        Self { poses, half_fov_tan, depth_scale }
    }

    /// Project every body position through the per-step camera.
    ///
    /// Returns new position arrays where X,Y are perspective-projected screen
    /// coordinates and Z is the focal-plane-relative depth (scaled so existing
    /// DOF / fog constants remain reasonable).
    pub fn project_all_positions(
        &self,
        positions: &[Vec<Vector3<f64>>],
    ) -> Vec<Vec<Vector3<f64>>> {
        let num_bodies = positions.len();
        let total_steps = positions[0].len();
        let mut out = vec![vec![Vector3::zeros(); total_steps]; num_bodies];

        for step in 0..total_steps {
            let p = &self.poses[step];
            let fwd = (p.target - p.eye).normalize();
            let right = fwd.cross(&p.up).normalize();
            let true_up = right.cross(&fwd);
            let focus_dist = (p.target - p.eye).norm();
            let z_floor = focus_dist * 0.05;

            for body in 0..num_bodies {
                let d = positions[body][step] - p.eye;
                let z_cam = d.dot(&fwd).max(z_floor);
                let x_cam = d.dot(&right);
                let y_cam = d.dot(&true_up);

                let proj_x = x_cam / (z_cam * self.half_fov_tan);
                let proj_y = y_cam / (z_cam * self.half_fov_tan);
                let depth = (z_cam - focus_dist) * self.depth_scale;

                out[body][step] = Vector3::new(proj_x, proj_y, depth);
            }
        }

        info!(bodies = num_bodies, steps = total_steps, "Positions projected through Gauss-map camera");
        out
    }

    /// Number of poses (one per simulation step).
    pub fn step_count(&self) -> usize {
        self.poses.len()
    }

    /// Project a single world-space point through the camera at a given step.
    pub fn project_point(&self, world_pos: &Vector3<f64>, step: usize) -> Vector3<f64> {
        let p = &self.poses[step];
        let fwd = (p.target - p.eye).normalize();
        let right = fwd.cross(&p.up).normalize();
        let true_up = right.cross(&fwd);
        let focus_dist = (p.target - p.eye).norm();
        let z_floor = focus_dist * 0.05;

        let d = world_pos - p.eye;
        let z_cam = d.dot(&fwd).max(z_floor);
        let x_cam = d.dot(&right);
        let y_cam = d.dot(&true_up);

        Vector3::new(
            x_cam / (z_cam * self.half_fov_tan),
            y_cam / (z_cam * self.half_fov_tan),
            (z_cam - focus_dist) * self.depth_scale,
        )
    }

    /// Extract raw camera pose data at a given step for GPU upload.
    #[cfg(feature = "gpu")]
    pub fn camera_uniforms_at_step(
        &self,
        step: usize,
    ) -> super::gpu::CameraRawUniforms {
        let p = &self.poses[step];
        let fwd = (p.target - p.eye).normalize();
        let right = fwd.cross(&p.up).normalize();
        let true_up = right.cross(&fwd);
        let focus_dist = (p.target - p.eye).norm();

        super::gpu::CameraRawUniforms {
            eye: [p.eye[0] as f32, p.eye[1] as f32, p.eye[2] as f32],
            fwd: [fwd[0] as f32, fwd[1] as f32, fwd[2] as f32],
            right: [right[0] as f32, right[1] as f32, right[2] as f32],
            true_up: [true_up[0] as f32, true_up[1] as f32, true_up[2] as f32],
            half_fov_tan: self.half_fov_tan as f32,
            focus_dist: focus_dist as f32,
            z_floor: (focus_dist * 0.05) as f32,
            depth_scale: self.depth_scale as f32,
        }
    }

    /// Project ALL body positions through a single camera pose (Option A).
    ///
    /// Unlike `project_all_positions` which uses per-step cameras, this projects
    /// every step's positions through the camera at `camera_step`, giving a
    /// consistent viewpoint for the entire trajectory.
    pub fn project_all_positions_at_step(
        &self,
        positions: &[Vec<Vector3<f64>>],
        camera_step: usize,
    ) -> Vec<Vec<Vector3<f64>>> {
        let num_bodies = positions.len();
        let total_steps = positions[0].len();
        let mut out = vec![vec![Vector3::zeros(); total_steps]; num_bodies];

        let p = &self.poses[camera_step];
        let fwd = (p.target - p.eye).normalize();
        let right = fwd.cross(&p.up).normalize();
        let true_up = right.cross(&fwd);
        let focus_dist = (p.target - p.eye).norm();
        let z_floor = focus_dist * 0.05;

        for step in 0..total_steps {
            for body in 0..num_bodies {
                let d = positions[body][step] - p.eye;
                let z_cam = d.dot(&fwd).max(z_floor);
                let x_cam = d.dot(&right);
                let y_cam = d.dot(&true_up);

                out[body][step] = Vector3::new(
                    x_cam / (z_cam * self.half_fov_tan),
                    y_cam / (z_cam * self.half_fov_tan),
                    (z_cam - focus_dist) * self.depth_scale,
                );
            }
        }
        out
    }

    /// Compute a global bounding box that covers projections from all sampled
    /// camera poses, ensuring stable image scale across video frames.
    pub fn compute_global_bounds(
        &self,
        positions: &[Vec<Vector3<f64>>],
        checkpoints: &[usize],
    ) -> BoundingBox {
        let num_bodies = positions.len();
        let total_steps = positions[0].len();
        let mut min_x = f64::MAX;
        let mut max_x = f64::MIN;
        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;

        let sample_interval = (checkpoints.len() / 50).max(1);
        let sampled: Vec<usize> = checkpoints
            .iter()
            .step_by(sample_interval)
            .copied()
            .chain(std::iter::once(*checkpoints.last().unwrap_or(&0)))
            .collect();

        for &cam_step in &sampled {
            let p = &self.poses[cam_step];
            let fwd = (p.target - p.eye).normalize();
            let right = fwd.cross(&p.up).normalize();
            let true_up = right.cross(&fwd);
            let focus_dist = (p.target - p.eye).norm();
            let z_floor = focus_dist * 0.05;

            for step in (0..total_steps).step_by((total_steps / 500).max(1)) {
                for body in 0..num_bodies {
                    let d = positions[body][step] - p.eye;
                    let z_cam = d.dot(&fwd).max(z_floor);
                    let x_cam = d.dot(&right);
                    let y_cam = d.dot(&true_up);

                    let proj_x = x_cam / (z_cam * self.half_fov_tan);
                    let proj_y = y_cam / (z_cam * self.half_fov_tan);

                    min_x = min_x.min(proj_x);
                    max_x = max_x.max(proj_x);
                    min_y = min_y.min(proj_y);
                    max_y = max_y.max(proj_y);
                }
            }
        }

        let pad_x = (max_x - min_x) * 0.05;
        let pad_y = (max_y - min_y) * 0.05;
        min_x -= pad_x;
        max_x += pad_x;
        min_y -= pad_y;
        max_y += pad_y;

        BoundingBox {
            min_x,
            max_x,
            min_y,
            max_y,
            width: (max_x - min_x).max(1e-12),
            height: (max_y - min_y).max(1e-12),
        }
    }
}

// ─── Geometry helpers ────────────────────────────────────────

/// Rodrigues rotation: rotate `v` around unit axis `k` by angle `theta`.
fn rodrigues(v: &Vector3<f64>, k: &Vector3<f64>, theta: f64) -> Vector3<f64> {
    let (s, c) = theta.sin_cos();
    v * c + k.cross(v) * s + k * k.dot(v) * (1.0 - c)
}

/// Minimum pairwise body distance at a single timestep.
fn min_pairwise_distance_at(positions: &[Vec<Vector3<f64>>], step: usize) -> f64 {
    let mut min_d = f64::MAX;
    let nb = positions.len();
    for i in 0..nb {
        for j in (i + 1)..nb {
            let d = (positions[i][step] - positions[j][step]).norm();
            min_d = min_d.min(d);
        }
    }
    min_d
}

/// Midpoint of the closest body pair at a single timestep.
fn midpoint_of_closest_pair(positions: &[Vec<Vector3<f64>>], step: usize) -> Vector3<f64> {
    let nb = positions.len();
    let mut min_d = f64::MAX;
    let mut best_i = 0;
    let mut best_j = 1;
    for i in 0..nb {
        for j in (i + 1)..nb {
            let d = (positions[i][step] - positions[j][step]).norm();
            if d < min_d {
                min_d = d;
                best_i = i;
                best_j = j;
            }
        }
    }
    (positions[best_i][step] + positions[best_j][step]) * 0.5
}

fn oriented_normal(
    positions: &[Vec<Vector3<f64>>],
    step: usize,
    reference: &Vector3<f64>,
) -> Vector3<f64> {
    let n = plane_normal_at(positions, step);
    if n.dot(reference) < 0.0 { -n } else { n }
}

#[allow(clippy::needless_range_loop)]
fn max_pairwise_distance(positions: &[Vec<Vector3<f64>>]) -> f64 {
    let nb = positions.len();
    let steps = positions[0].len();
    let mut max_d = 0.0_f64;
    for step in 0..steps {
        for i in 0..nb {
            for j in (i + 1)..nb {
                let d = (positions[i][step] - positions[j][step]).norm();
                max_d = max_d.max(d);
            }
        }
    }
    max_d
}

// ─── EMA helpers ─────────────────────────────────────────────

fn ema_alpha(tau_frac: f64, total_steps: usize) -> f64 {
    let tau = (tau_frac * total_steps as f64).max(1.0);
    1.0 - (-1.0 / tau).exp()
}

fn ema_smooth(v: &mut [f64], alpha: f64) {
    for i in 1..v.len() {
        v[i] = alpha * v[i] + (1.0 - alpha) * v[i - 1];
    }
}

fn ema_smooth_vec(v: &mut [Vector3<f64>], alpha: f64) {
    for i in 1..v.len() {
        v[i] = alpha * v[i] + (1.0 - alpha) * v[i - 1];
    }
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::triangle_area_at;

    fn equilateral_orbit(steps: usize) -> Vec<Vec<Vector3<f64>>> {
        let mut bodies: Vec<Vec<Vector3<f64>>> = vec![Vec::with_capacity(steps); 3];
        for s in 0..steps {
            let t = s as f64 * 0.01;
            bodies[0].push(Vector3::new(t.cos(), t.sin(), 0.0));
            bodies[1].push(Vector3::new(
                (t + 2.0 * PI / 3.0).cos(),
                (t + 2.0 * PI / 3.0).sin(),
                0.0,
            ));
            bodies[2].push(Vector3::new(
                (t + 4.0 * PI / 3.0).cos(),
                (t + 4.0 * PI / 3.0).sin(),
                0.0,
            ));
        }
        bodies
    }

    fn tumbling_orbit(steps: usize) -> Vec<Vec<Vector3<f64>>> {
        let mut bodies: Vec<Vec<Vector3<f64>>> = vec![Vec::with_capacity(steps); 3];
        for s in 0..steps {
            let t = s as f64 * 0.01;
            let tilt = (t * 0.1).sin() * 0.3;
            bodies[0].push(Vector3::new(
                t.cos(),
                t.sin() * tilt.cos(),
                t.sin() * tilt.sin(),
            ));
            bodies[1].push(Vector3::new(
                (t + 2.0 * PI / 3.0).cos(),
                (t + 2.0 * PI / 3.0).sin() * tilt.cos(),
                (t + 2.0 * PI / 3.0).sin() * tilt.sin(),
            ));
            bodies[2].push(Vector3::new(
                (t + 4.0 * PI / 3.0).cos(),
                (t + 4.0 * PI / 3.0).sin() * tilt.cos(),
                (t + 4.0 * PI / 3.0).sin() * tilt.sin(),
            ));
        }
        bodies
    }

    fn close_encounter_orbit(steps: usize) -> Vec<Vec<Vector3<f64>>> {
        let mut bodies: Vec<Vec<Vector3<f64>>> = vec![Vec::with_capacity(steps); 3];
        for s in 0..steps {
            let t = s as f64 / (steps - 1).max(1) as f64;
            // V-shaped spread: 1.0 at edges, 0.1 at center (close encounter at midpoint)
            let spread = 0.1 + 0.9 * (2.0 * (t - 0.5)).abs();
            bodies[0].push(Vector3::new(spread, 0.0, 0.0));
            bodies[1].push(Vector3::new(-0.5 * spread, 0.866 * spread, 0.0));
            bodies[2].push(Vector3::new(-0.5 * spread, -0.866 * spread, 0.1));
        }
        bodies
    }

    #[test]
    fn plane_normal_equilateral_in_xy() {
        let pos = equilateral_orbit(10);
        let n = plane_normal_at(&pos, 0);
        assert!(
            n[2].abs() > 0.99,
            "normal of an XY-plane triangle should point along Z, got {:?}",
            n
        );
    }

    #[test]
    fn triangle_area_known() {
        let pos = vec![
            vec![Vector3::new(0.0, 0.0, 0.0)],
            vec![Vector3::new(1.0, 0.0, 0.0)],
            vec![Vector3::new(0.0, 1.0, 0.0)],
        ];
        let area = triangle_area_at(&pos, 0);
        assert!((area - 0.5).abs() < 1e-10, "expected 0.5, got {}", area);
    }

    #[test]
    fn rodrigues_identity() {
        let v = Vector3::new(1.0, 0.0, 0.0);
        let k = Vector3::new(0.0, 0.0, 1.0);
        let rotated = rodrigues(&v, &k, 0.0);
        assert!((rotated - v).norm() < 1e-12);
    }

    #[test]
    fn rodrigues_quarter_turn() {
        let v = Vector3::new(1.0, 0.0, 0.0);
        let k = Vector3::new(0.0, 0.0, 1.0);
        let rotated = rodrigues(&v, &k, PI / 2.0);
        let expected = Vector3::new(0.0, 1.0, 0.0);
        assert!(
            (rotated - expected).norm() < 1e-12,
            "quarter turn around Z should send X to Y, got {:?}",
            rotated
        );
    }

    #[test]
    fn min_pairwise_distance_known() {
        let pos = vec![
            vec![Vector3::new(0.0, 0.0, 0.0)],
            vec![Vector3::new(3.0, 0.0, 0.0)],
            vec![Vector3::new(10.0, 0.0, 0.0)],
        ];
        let d = min_pairwise_distance_at(&pos, 0);
        assert!((d - 3.0).abs() < 1e-10);
    }

    #[test]
    fn midpoint_of_closest_pair_known() {
        let pos = vec![
            vec![Vector3::new(0.0, 0.0, 0.0)],
            vec![Vector3::new(2.0, 0.0, 0.0)],
            vec![Vector3::new(100.0, 0.0, 0.0)],
        ];
        let mid = midpoint_of_closest_pair(&pos, 0);
        let expected = Vector3::new(1.0, 0.0, 0.0);
        assert!((mid - expected).norm() < 1e-10);
    }

    #[test]
    fn ema_smooth_converges() {
        let mut vals = vec![0.0, 10.0, 10.0, 10.0, 10.0, 10.0];
        ema_smooth(&mut vals, 0.3);
        assert!(vals[1] < 10.0, "first smoothed value should lag behind");
        for w in vals.windows(2) {
            assert!(w[1] >= w[0], "smoothed series should be monotonically non-decreasing");
        }
    }

    #[test]
    fn camera_construction_does_not_panic() {
        let pos = equilateral_orbit(500);
        let _cam = Camera3D::new(&Camera3DConfig::default(), &pos);
    }

    #[test]
    fn projected_positions_are_finite() {
        let pos = equilateral_orbit(200);
        let cam = Camera3D::new(&Camera3DConfig::default(), &pos);
        let proj = cam.project_all_positions(&pos);

        assert_eq!(proj.len(), 3);
        assert_eq!(proj[0].len(), 200);
        for body in &proj {
            for v in body {
                assert!(v[0].is_finite(), "proj_x not finite: {:?}", v);
                assert!(v[1].is_finite(), "proj_y not finite: {:?}", v);
                assert!(v[2].is_finite(), "depth not finite: {:?}", v);
            }
        }
    }

    #[test]
    fn tumbling_orbit_gives_different_projection_than_planar() {
        let planar = equilateral_orbit(200);
        let tumbled = tumbling_orbit(200);

        let cam_p = Camera3D::new(&Camera3DConfig::default(), &planar);
        let cam_t = Camera3D::new(&Camera3DConfig::default(), &tumbled);

        let proj_p = cam_p.project_all_positions(&planar);
        let proj_t = cam_t.project_all_positions(&tumbled);

        let diff: f64 = proj_p[0]
            .iter()
            .zip(proj_t[0].iter())
            .map(|(a, b)| (a - b).norm())
            .sum();

        assert!(diff > 0.1, "tumbling orbit should produce distinct projections");
    }

    #[test]
    fn close_encounter_zooms_in() {
        let pos = close_encounter_orbit(200);
        let cam = Camera3D::new(&Camera3DConfig::default(), &pos);

        let mid_step = 100;
        let edge_step = 0;
        let mid_dist = (cam.poses[mid_step].eye - cam.poses[mid_step].target).norm();
        let edge_dist = (cam.poses[edge_step].eye - cam.poses[edge_step].target).norm();

        assert!(
            mid_dist < edge_dist,
            "camera should be closer during the close encounter (mid={:.3}, edge={:.3})",
            mid_dist,
            edge_dist
        );
    }

    #[test]
    fn single_step_does_not_panic() {
        let pos = vec![
            vec![Vector3::new(1.0, 0.0, 0.0)],
            vec![Vector3::new(-0.5, 0.866, 0.0)],
            vec![Vector3::new(-0.5, -0.866, 0.0)],
        ];
        let cam = Camera3D::new(&Camera3DConfig::default(), &pos);
        let proj = cam.project_all_positions(&pos);
        assert_eq!(proj[0].len(), 1);
    }

    #[test]
    fn depth_near_focal_plane_is_small() {
        let pos = equilateral_orbit(100);
        let cam = Camera3D::new(&Camera3DConfig::default(), &pos);
        let proj = cam.project_all_positions(&pos);

        let avg_depth: f64 =
            proj.iter().flat_map(|b| b.iter().map(|v| v[2].abs())).sum::<f64>()
                / (proj.len() * proj[0].len()) as f64;

        assert!(
            avg_depth < 100.0,
            "average depth should be moderate (got {})",
            avg_depth
        );
    }
}
