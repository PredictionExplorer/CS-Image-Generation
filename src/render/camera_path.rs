//! Simple perspective camera for orbit framing (thin-lens style `CoC` in pixels).

use crate::render::context::BoundingBox;
use nalgebra::Vector3;

/// Perspective camera with a stable look-at frame.
#[derive(Clone, Debug)]
pub struct PerspectiveCamera {
    /// Camera position in world space.
    pub eye: Vector3<f64>,
    /// Look-at target.
    pub target: Vector3<f64>,
    /// Up vector (not required orthonormal until `build_frame`).
    pub up: Vector3<f64>,
    /// Vertical field of view in radians.
    pub fov_y: f64,
    /// Image aspect `width / height`.
    pub aspect: f64,
    /// Focus distance along forward axis from `eye` (world units).
    pub focus_distance: f64,
    /// Max circle-of-confusion radius in pixels at far defocus.
    pub coc_max_px: f64,
    /// Cached orthonormal camera X axis in world space (points right on the image).
    pub right: Vector3<f64>,
    /// Cached orthonormal camera Y axis in world space (image up).
    pub cam_up: Vector3<f64>,
    /// Cached unit vector from `eye` toward `target` (into the scene).
    pub forward: Vector3<f64>,
}

impl PerspectiveCamera {
    /// Build a camera that frames `bbox` from an elevated viewpoint.
    #[must_use]
    pub fn orbit_default(bbox: &BoundingBox, width: u32, height: u32) -> Self {
        let cx = f64::midpoint(bbox.min_x, bbox.max_x);
        let cy = f64::midpoint(bbox.min_y, bbox.max_y);
        let cz = f64::midpoint(
            // z extent from positions not in bbox — use 0 centre in z for stability
            -bbox.height * 0.1,
            bbox.height * 0.1,
        );
        let center = Vector3::new(cx, cy, cz);
        let span = bbox.width.max(bbox.height).max(1e-6);
        let eye = center + Vector3::new(span * 0.15, -span * 1.8, span * 1.2);
        let target = center;
        let up = Vector3::new(0.0, 0.0, 1.0);
        let aspect = f64::from(width) / f64::from(height).max(1.0);
        let mut cam = Self {
            eye,
            target,
            up,
            fov_y: 55_f64.to_radians(),
            aspect,
            focus_distance: (eye - target).norm(),
            coc_max_px: 6.0,
            right: Vector3::zeros(),
            cam_up: Vector3::zeros(),
            forward: Vector3::zeros(),
        };
        cam.build_frame();
        cam
    }

    fn build_frame(&mut self) {
        self.forward = (self.target - self.eye).normalize();
        self.right = self.forward.cross(&self.up).normalize();
        self.cam_up = self.right.cross(&self.forward).normalize();
    }

    /// Project world point to pixel coordinates and an approximate `CoC` radius in pixels.
    #[must_use]
    pub fn project(&self, width: u32, height: u32, p: Vector3<f64>) -> (f32, f32, f32) {
        let w = f64::from(width);
        let h = f64::from(height);
        let v = p - self.eye;
        let x_c = v.dot(&self.right);
        let y_c = v.dot(&self.cam_up);
        let z_c = v.dot(&self.forward).max(1e-9);
        let tan_half = (self.fov_y * 0.5).tan();
        let ndc_x = x_c / (z_c * tan_half * self.aspect);
        let ndc_y = y_c / (z_c * tan_half);
        let px = (0.5 * (ndc_x + 1.0)) * w;
        let py = (0.5 * (1.0 - ndc_y)) * h;
        let z_focus = self.focus_distance.max(1e-9);
        let coc = ((z_c - z_focus).abs() / z_focus).min(1.0) * self.coc_max_px;
        (px as f32, py as f32, coc as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::context::BoundingBox;

    fn test_bbox() -> BoundingBox {
        BoundingBox { min_x: -4.0, max_x: 4.0, min_y: -3.0, max_y: 3.0, width: 8.0, height: 6.0 }
    }

    #[test]
    fn test_orbit_default_builds_orthonormal_basis() {
        let cam = PerspectiveCamera::orbit_default(&test_bbox(), 1920, 1080);
        assert!((cam.forward.norm() - 1.0).abs() < 1e-12);
        assert!((cam.right.norm() - 1.0).abs() < 1e-12);
        assert!((cam.cam_up.norm() - 1.0).abs() < 1e-12);
        assert!(cam.right.dot(&cam.forward).abs() < 1e-12);
        assert!(cam.cam_up.dot(&cam.forward).abs() < 1e-12);
        assert!(cam.right.dot(&cam.cam_up).abs() < 1e-12);
    }

    #[test]
    fn test_projection_at_target_falls_near_image_center() {
        let cam = PerspectiveCamera::orbit_default(&test_bbox(), 1920, 1080);
        let (px, py, coc) = cam.project(1920, 1080, cam.target);
        assert!((px - 960.0).abs() < 1.0, "target should land near x=960, got {px}");
        assert!((py - 540.0).abs() < 1.0, "target should land near y=540, got {py}");
        assert!(coc.abs() < 1e-4, "focus pixel should have near-zero CoC");
    }

    #[test]
    fn test_projection_behind_camera_is_clamped() {
        let cam = PerspectiveCamera::orbit_default(&test_bbox(), 400, 300);
        let back = cam.eye - cam.forward * 5.0;
        let (_, _, coc) = cam.project(400, 300, back);
        assert!(coc.is_finite(), "CoC for back-of-camera must remain finite");
    }

    #[test]
    fn test_coc_increases_with_defocus() {
        let cam = PerspectiveCamera::orbit_default(&test_bbox(), 1280, 720);
        let focus = cam.eye + cam.forward * cam.focus_distance;
        let far = cam.eye + cam.forward * (cam.focus_distance * 3.0);
        let (_, _, c_focus) = cam.project(1280, 720, focus);
        let (_, _, c_far) = cam.project(1280, 720, far);
        assert!(c_far >= c_focus);
    }
}
