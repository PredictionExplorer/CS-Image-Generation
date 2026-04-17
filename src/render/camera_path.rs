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
    ///
    /// Legacy behaviour: fixed 55 FOV with a heuristic eye distance. Retained
    /// under the `--framing classic` escape hatch; prefer [`orbit_fit`] for
    /// content that actually fills the frame.
    #[must_use]
    pub fn orbit_classic(bbox: &BoundingBox, width: u32, height: u32) -> Self {
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

    /// Backwards-compatible alias for [`orbit_classic`].
    #[must_use]
    #[inline]
    pub fn orbit_default(bbox: &BoundingBox, width: u32, height: u32) -> Self {
        Self::orbit_classic(bbox, width, height)
    }

    /// Build a camera that **fits** the bbox to the frame.
    ///
    /// Uses a 45 FOV (less wide-angle distortion than 55) and walks the eye
    /// along a slightly off-axis elevated viewing direction until the furthest
    /// bbox corner projects within `fill` of the frame (e.g. `fill = 0.90`
    /// gives 90% coverage with 5% safety margin per edge).
    ///
    /// Both horizontal and vertical screen-space extents are constrained and
    /// the tighter distance is kept, so the orbit fills the output regardless
    /// of aspect ratio.
    #[must_use]
    pub fn orbit_fit(bbox: &BoundingBox, width: u32, height: u32, fill: f64) -> Self {
        Self::orbit_fit_with_offset(bbox, width, height, fill, Vector3::zeros())
    }

    /// Same as [`orbit_fit`] but applies a world-space shift to the camera
    /// target before solving for the distance. The shift is used to place
    /// the subject on a rule-of-thirds intersection instead of dead center.
    #[must_use]
    pub fn orbit_fit_with_offset(
        bbox: &BoundingBox,
        width: u32,
        height: u32,
        fill: f64,
        focal_offset: Vector3<f64>,
    ) -> Self {
        let fill_clamped = fill.clamp(0.2, 0.98);
        let aspect = f64::from(width) / f64::from(height).max(1.0);

        let cx = f64::midpoint(bbox.min_x, bbox.max_x);
        let cy = f64::midpoint(bbox.min_y, bbox.max_y);
        let cz = f64::midpoint(bbox.min_z, bbox.max_z);
        let target = Vector3::new(cx, cy, cz) + focal_offset;
        let up = Vector3::new(0.0, 0.0, 1.0);

        let fov_y = 45_f64.to_radians();
        let tan_half_y = (fov_y * 0.5).tan();

        // Viewing direction: mildly off-axis for subtle 3D parallax without
        // wasting slivers of screen space on a highly oblique composition.
        // The earlier direction `(0.15, -1.8, 1.2)` left large empty corners
        // because the apparent bbox was stretched diagonally on screen.
        let dir = Vector3::new(0.08, -1.6, 0.55).normalize();

        // Seed distance from the span so corner projections are well-defined.
        let span = bbox.width.max(bbox.height).max(bbox.depth).max(1e-6);
        let mut dist = span * 1.5;

        // Build a provisional camera to obtain right/cam_up; these do not
        // depend on `eye`'s distance, only on the viewing direction.
        let eye = target - dir * dist;
        let forward = (target - eye).normalize();
        let right = forward.cross(&up).normalize();
        let cam_up = right.cross(&forward).normalize();

        // For each bbox corner, compute the minimum eye distance along `-dir`
        // that puts its projected NDC within +/- fill (horizontal and vertical
        // separately). We solve: |x_c| <= fill * aspect * tan_half * z_c.
        // With eye = target - dir * d, let o = corner - target. Then
        //   x_c = o.right,  y_c = o.up,  z_c = o.forward + d
        // so required d is (|x_c| / (fill*aspect*tan_half)) - o.forward for x,
        // and (|y_c| / (fill*tan_half)) - o.forward for y. We take the max.
        let mut needed_d: f64 = span * 0.25;
        for corner in bbox.corners() {
            let o = corner - target;
            let ox = o.dot(&right);
            let oy = o.dot(&cam_up);
            let oz = o.dot(&forward);
            let limit_x = fill_clamped * aspect * tan_half_y;
            let limit_y = fill_clamped * tan_half_y;
            let dx = ox.abs() / limit_x.max(1e-9) - oz;
            let dy = oy.abs() / limit_y.max(1e-9) - oz;
            needed_d = needed_d.max(dx).max(dy);
        }
        // Keep a small floor so we never end up with the eye inside the bbox.
        dist = needed_d.max(span * 0.25);

        let eye = target - dir * dist;
        let mut cam = Self {
            eye,
            target,
            up,
            fov_y,
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

    /// Recompute the cached orthonormal camera basis (`forward`, `right`,
    /// `cam_up`) from the current `eye`, `target`, and `up`.
    ///
    /// Called automatically by the `orbit_*` constructors; only callers
    /// who mutate `eye`/`target`/`up` directly (fit-to-ink framing)
    /// need to invoke it manually.
    pub fn build_frame(&mut self) {
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
        BoundingBox {
            min_x: -4.0,
            max_x: 4.0,
            min_y: -3.0,
            max_y: 3.0,
            min_z: -1.0,
            max_z: 1.0,
            width: 8.0,
            height: 6.0,
            depth: 2.0,
        }
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

    #[test]
    fn test_orbit_fit_all_corners_inside_frame() {
        let bbox = test_bbox();
        let (w, h) = (1920u32, 1080u32);
        let cam = PerspectiveCamera::orbit_fit(&bbox, w, h, 0.90);
        for corner in bbox.corners() {
            let (px, py, _) = cam.project(w, h, corner);
            assert!(
                px >= 0.0 && px <= w as f32,
                "corner {corner:?} projected x={px} outside [0,{w}]"
            );
            assert!(
                py >= 0.0 && py <= h as f32,
                "corner {corner:?} projected y={py} outside [0,{h}]"
            );
        }
    }

    #[test]
    fn test_orbit_fit_covers_at_least_target_fraction() {
        // The widest corner in either axis should project near (1 - fill)/2
        // from the edge in the tighter dimension.
        let bbox = test_bbox();
        let (w, h) = (1920u32, 1080u32);
        let fill = 0.90;
        let cam = PerspectiveCamera::orbit_fit(&bbox, w, h, fill);
        let mut max_ndc_x = 0.0f32;
        let mut max_ndc_y = 0.0f32;
        for corner in bbox.corners() {
            let (px, py, _) = cam.project(w, h, corner);
            let nx = (px / w as f32 - 0.5).abs() * 2.0;
            let ny = (py / h as f32 - 0.5).abs() * 2.0;
            max_ndc_x = max_ndc_x.max(nx);
            max_ndc_y = max_ndc_y.max(ny);
        }
        // At least one axis should reach close to `fill` (the constraining axis).
        let max_axis = max_ndc_x.max(max_ndc_y);
        assert!(
            max_axis >= fill as f32 - 0.02,
            "orbit_fit underfills frame: max_ndc={max_axis}, expected >= {}",
            fill - 0.02
        );
        // And no axis exceeds fill + small numeric slack.
        assert!(
            max_axis <= fill as f32 + 0.02,
            "orbit_fit overfills frame: max_ndc={max_axis}, expected <= {}",
            fill + 0.02
        );
    }

    #[test]
    fn test_orbit_fit_target_at_center() {
        let bbox = test_bbox();
        let cam = PerspectiveCamera::orbit_fit(&bbox, 1600, 900, 0.90);
        let (px, py, _) = cam.project(1600, 900, cam.target);
        assert!((px - 800.0).abs() < 1.0);
        assert!((py - 450.0).abs() < 1.0);
    }
}
