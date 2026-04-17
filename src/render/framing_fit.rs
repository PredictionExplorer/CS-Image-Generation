//! Fit-to-ink camera framing.
//!
//! The legacy [`super::camera_path::PerspectiveCamera::orbit_fit`] places the
//! bounding box **corners** at `fill` of the frame. For a thin three-body
//! curve threading through a mostly empty 3D bbox, that leaves the visible
//! ink occupying only a small fraction of the frame.
//!
//! This module projects every trajectory sample with a scratch camera, finds
//! the percentile-trimmed **pixel** extent of the actual ink, then rebuilds
//! the camera with a shifted target and tighter distance so the visible ink
//! fills `fill` of the frame.
//!
//! Before the iterated zoom refit, the seed camera is **roll-aligned** about
//! its forward axis so the principal axis of the projected ink matches the
//! frame's long axis (horizontal for landscape, vertical for portrait). This
//! converts diagonally elongated trajectories — which would otherwise waste
//! most of the frame on empty corners — into compositions that actually fill
//! the constraining axis. The roll is skipped for near-isotropic scenes to
//! avoid introducing a spurious tilt on symmetric orbits.
//!
//! The refit itself is **iterated to convergence** (up to five passes) rather
//! than a fixed two-pass overshoot correction, because perspective is
//! non-linear in `1/distance` and high-drift orbits embedded in an inflated
//! bounding box can require several corrective passes before the projected
//! ink fills the configured fraction. A final recentre-only pass locks the
//! subject to dead-centre regardless of any parallax introduced by the last
//! zoom step.

use super::camera_path::PerspectiveCamera;
use super::context::BoundingBox;
use nalgebra::Vector3;

/// Build a perspective camera that fits the **projected ink** of `positions`
/// into `fill` of the frame.
///
/// `bbox` seeds the initial camera (target, direction, distance) and should
/// be the percentile-trimmed 3D bounding box of the same trajectories.
///
/// `fill` is the target fraction of the frame covered by the ink
/// (clamped to `[0.2, 0.98]`).
///
/// `ink_pct` is the fraction of projected samples kept before computing the
/// pixel bbox (clamped to `[0.5, 1.0]`). `0.98` trims 1% from each tail,
/// which robustly excludes the rare outlier blown-out by drift without
/// sacrificing the body of the curve.
#[must_use]
pub fn fit_to_ink_camera(
    positions: &[Vec<Vector3<f64>>],
    bbox: &BoundingBox,
    width: u32,
    height: u32,
    fill: f64,
    ink_pct: f64,
    focal_offset_world: Vector3<f64>,
) -> PerspectiveCamera {
    let fill = fill.clamp(0.2, 0.98);
    let ink_pct = ink_pct.clamp(0.5, 1.0);

    // Seed camera: fill=1.0 so corners just touch the frame; the projected
    // ink is guaranteed to lie inside the viewport (or close to it).
    let mut seed = PerspectiveCamera::orbit_fit(bbox, width, height, 1.0);

    let total: usize = positions.iter().map(std::vec::Vec::len).sum();
    if total == 0 {
        return offset_target(seed, focal_offset_world);
    }

    // Roll-align the seed: rotate `up` about `forward` so the projected
    // ink's principal axis matches the frame's long axis. This eliminates
    // the diagonal-curve-in-landscape-frame failure mode where a huge chunk
    // of the frame is wasted on empty corners.
    apply_principal_axis_roll(&mut seed, positions, width, height);

    // Pass 1 is zoom-in-only to protect already-tight seeds from regressing,
    // and it carries the rule-of-thirds focal offset into the target.
    let mut cam = refit_once(
        &seed,
        positions,
        width,
        height,
        fill,
        ink_pct,
        focal_offset_world,
        /* allow_zoom_out = */ false,
    );

    // Iterate to convergence. Perspective is non-linear in `1/distance`, so
    // a single overshoot correction isn't always enough — a drift-inflated
    // bbox can leave the projected ink a small fraction of the seed viewport
    // after pass 1, and the subsequent passes need to be allowed to zoom
    // out (via `fill * 0.99` headroom) to settle without clipping the frame
    // edge. Five total passes converge reliably on the seeds we've seen.
    let w = f64::from(width);
    let h = f64::from(height);
    for _ in 0..MAX_REFIT_ITERS {
        let Some(extent) = measure_ink_pixel_extent(&cam, positions, width, height, ink_pct)
        else {
            break;
        };
        let fill_actual = (extent.width() / w).max(extent.height() / h);
        if (fill_actual - fill).abs() <= FILL_CONVERGENCE_TOL {
            break;
        }
        cam = refit_once(
            &cam,
            positions,
            width,
            height,
            fill * 0.99,
            ink_pct,
            Vector3::zeros(),
            /* allow_zoom_out = */ true,
        );
    }

    // Final recentre-only pass: lock the subject to dead-centre without
    // touching zoom. Perspective parallax from the last zoom step can leave
    // the projected centre a few pixels off-axis; this pass absorbs that
    // residual without destabilising the just-converged fill.
    if let Some(extent) = measure_ink_pixel_extent(&cam, positions, width, height, ink_pct) {
        cam = recenter_only(&cam, &extent, width, height);
    }

    cam
}

/// Maximum additional refit passes after the zoom-in-only first pass.
/// Empirically, three or four are enough; five is a safety margin.
const MAX_REFIT_ITERS: usize = 4;

/// Stop iterating once the constraining-axis fill is within this tolerance
/// of the target. 1.5% = a ~15-pixel envelope on a 1080-pixel axis, which is
/// below visual perception for composition and well under the motion-blur
/// footprint.
const FILL_CONVERGENCE_TOL: f64 = 0.015;

/// Percentile-trimmed pixel-space extent of the projected ink.
#[derive(Clone, Copy, Debug)]
struct InkExtent {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

impl InkExtent {
    #[inline]
    fn width(&self) -> f64 {
        (self.max_x - self.min_x).max(1.0)
    }
    #[inline]
    fn height(&self) -> f64 {
        (self.max_y - self.min_y).max(1.0)
    }
    #[inline]
    fn center_x(&self) -> f64 {
        0.5 * (self.min_x + self.max_x)
    }
    #[inline]
    fn center_y(&self) -> f64 {
        0.5 * (self.min_y + self.max_y)
    }
}

/// Project every trajectory sample through `camera` and return the
/// percentile-trimmed pixel bounding box of the result. Returns `None` when
/// no finite projections exist (empty positions or degenerate camera).
fn measure_ink_pixel_extent(
    camera: &PerspectiveCamera,
    positions: &[Vec<Vector3<f64>>],
    width: u32,
    height: u32,
    ink_pct: f64,
) -> Option<InkExtent> {
    let total: usize = positions.iter().map(std::vec::Vec::len).sum();
    let mut xs: Vec<f64> = Vec::with_capacity(total);
    let mut ys: Vec<f64> = Vec::with_capacity(total);
    for body in positions {
        for &p in body {
            let (px, py, _) = camera.project(width, height, p);
            if px.is_finite() && py.is_finite() {
                xs.push(f64::from(px));
                ys.push(f64::from(py));
            }
        }
    }
    if xs.is_empty() {
        return None;
    }

    let pct_low = 0.5 * (1.0 - ink_pct);
    let (min_x, max_x) = axis_percentile_f64(&mut xs, pct_low);
    let (min_y, max_y) = axis_percentile_f64(&mut ys, pct_low);
    Some(InkExtent { min_x, max_x, min_y, max_y })
}

fn refit_once(
    camera: &PerspectiveCamera,
    positions: &[Vec<Vector3<f64>>],
    width: u32,
    height: u32,
    fill: f64,
    ink_pct: f64,
    focal_offset_world: Vector3<f64>,
    allow_zoom_out: bool,
) -> PerspectiveCamera {
    let Some(extent) = measure_ink_pixel_extent(camera, positions, width, height, ink_pct) else {
        return offset_target(camera.clone(), focal_offset_world);
    };

    let w = f64::from(width);
    let h = f64::from(height);

    let ink_w_px = extent.width();
    let ink_h_px = extent.height();
    let ink_cx = extent.center_x();
    let ink_cy = extent.center_y();

    let zoom_x = (fill * w) / ink_w_px;
    let zoom_y = (fill * h) / ink_h_px;
    let mut zoom = zoom_x.min(zoom_y);
    if !zoom.is_finite() {
        zoom = 1.0;
    }
    if !allow_zoom_out && zoom < 1.0 {
        zoom = 1.0;
    }
    // Upper bound of 64x lets the iteration close the gap when the seed
    // viewport is drift-inflated and the projected ink is a small fraction
    // of it. The lower bound stays at 0.2 for stability on degenerate inputs
    // (single-point orbits, NaNs, etc.).
    zoom = zoom.clamp(0.2, 64.0);

    let world_per_pixel_y = camera.focus_distance * (camera.fov_y * 0.5).tan() * 2.0 / h.max(1.0);
    let world_per_pixel_x = world_per_pixel_y;

    let dx_px = ink_cx - 0.5 * w;
    let dy_px = ink_cy - 0.5 * h;

    let target_offset =
        camera.right * (dx_px * world_per_pixel_x) + camera.cam_up * (-dy_px * world_per_pixel_y);

    let new_target = camera.target + target_offset + focal_offset_world;
    let dir = (camera.target - camera.eye).normalize();
    let new_distance = (camera.focus_distance / zoom).max(1e-6);
    let new_eye = new_target - dir * new_distance;

    let mut cam = PerspectiveCamera {
        eye: new_eye,
        target: new_target,
        up: camera.up,
        fov_y: camera.fov_y,
        aspect: camera.aspect,
        focus_distance: new_distance,
        coc_max_px: camera.coc_max_px,
        right: Vector3::zeros(),
        cam_up: Vector3::zeros(),
        forward: Vector3::zeros(),
    };
    cam.build_frame();
    cam
}

/// Translate the camera target so the projected ink lands dead-centre,
/// preserving the current zoom / focus distance. Used as a final pass after
/// the iterated refit so any perspective-parallax drift introduced by the
/// last zoom step doesn't leave the subject visibly off-axis.
fn recenter_only(
    camera: &PerspectiveCamera,
    extent: &InkExtent,
    width: u32,
    height: u32,
) -> PerspectiveCamera {
    let w = f64::from(width);
    let h = f64::from(height);

    let dx_px = extent.center_x() - 0.5 * w;
    let dy_px = extent.center_y() - 0.5 * h;

    // Already centred within sub-pixel tolerance — skip the rebuild.
    if dx_px.abs() < 0.5 && dy_px.abs() < 0.5 {
        return camera.clone();
    }

    let world_per_pixel_y = camera.focus_distance * (camera.fov_y * 0.5).tan() * 2.0 / h.max(1.0);
    let world_per_pixel_x = world_per_pixel_y;

    let target_offset =
        camera.right * (dx_px * world_per_pixel_x) + camera.cam_up * (-dy_px * world_per_pixel_y);

    let new_target = camera.target + target_offset;
    let dir = (camera.target - camera.eye).normalize();
    let new_eye = new_target - dir * camera.focus_distance;

    let mut cam = PerspectiveCamera {
        eye: new_eye,
        target: new_target,
        up: camera.up,
        fov_y: camera.fov_y,
        aspect: camera.aspect,
        focus_distance: camera.focus_distance,
        coc_max_px: camera.coc_max_px,
        right: Vector3::zeros(),
        cam_up: Vector3::zeros(),
        forward: Vector3::zeros(),
    };
    cam.build_frame();
    cam
}

fn offset_target(mut cam: PerspectiveCamera, offset: Vector3<f64>) -> PerspectiveCamera {
    if offset.norm() < 1e-12 {
        return cam;
    }
    cam.target += offset;
    let dir = (cam.target - cam.eye).normalize();
    cam.eye = cam.target - dir * cam.focus_distance;
    cam.build_frame();
    cam
}

/// Project all ink through `cam` and, if the projected point cloud is
/// meaningfully anisotropic, roll the camera about its forward axis so the
/// principal axis of the projection aligns with the frame's long axis.
///
/// No-op when the ink is near-isotropic (major-variance fraction below
/// `ANISOTROPY_THRESHOLD`); this avoids introducing a spurious tilt on
/// symmetric orbits.
fn apply_principal_axis_roll(
    cam: &mut PerspectiveCamera,
    positions: &[Vec<Vector3<f64>>],
    width: u32,
    height: u32,
) {
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for body in positions {
        for &p in body {
            let (px, py, _) = cam.project(width, height, p);
            if px.is_finite() && py.is_finite() {
                xs.push(f64::from(px));
                ys.push(f64::from(py));
            }
        }
    }
    if xs.len() < 16 {
        return;
    }

    let Some((theta, major_frac)) = principal_angle_rad(&xs, &ys) else {
        return;
    };

    /// Minimum fraction of total projected variance captured by the major
    /// axis to bother rolling. `0.55` means the major axis must hold at
    /// least 55% of the variance (i.e. clearly elongated, not isotropic).
    const ANISOTROPY_THRESHOLD: f64 = 0.55;
    if major_frac < ANISOTROPY_THRESHOLD {
        return;
    }

    let target_angle = if width >= height { 0.0 } else { std::f64::consts::FRAC_PI_2 };
    // Normalize to the shortest equivalent rotation in [-PI/2, PI/2] since
    // the principal axis is a line, not a direction.
    let mut roll = target_angle - theta;
    while roll > std::f64::consts::FRAC_PI_2 {
        roll -= std::f64::consts::PI;
    }
    while roll < -std::f64::consts::FRAC_PI_2 {
        roll += std::f64::consts::PI;
    }
    if roll.abs() < 1e-4 {
        return;
    }

    roll_camera(cam, roll);
}

/// Closed-form 2x2-covariance PCA on centered (`xs`, `ys`).
///
/// Returns `Some((theta, major_frac))` where:
/// * `theta` is the angle (in radians) of the major axis measured from the
///   image-x axis. Image coordinates have y growing downward, but since we
///   only use this to align a line — not a direction — the sign flip is
///   irrelevant.
/// * `major_frac` is `lambda_major / (lambda_major + lambda_minor)`,
///   i.e. the fraction of total variance captured by the major eigenvalue
///   (range `[0.5, 1.0]`; `0.5` = perfectly isotropic, `1.0` = degenerate
///   line).
///
/// Returns `None` if the inputs degenerate (zero variance or NaN).
fn principal_angle_rad(xs: &[f64], ys: &[f64]) -> Option<(f64, f64)> {
    debug_assert_eq!(xs.len(), ys.len());
    let n = xs.len();
    if n < 2 {
        return None;
    }
    let nf = n as f64;
    let mean_x: f64 = xs.iter().sum::<f64>() / nf;
    let mean_y: f64 = ys.iter().sum::<f64>() / nf;

    let mut cxx = 0.0;
    let mut cyy = 0.0;
    let mut cxy = 0.0;
    for i in 0..n {
        let dx = xs[i] - mean_x;
        let dy = ys[i] - mean_y;
        cxx += dx * dx;
        cyy += dy * dy;
        cxy += dx * dy;
    }
    cxx /= nf;
    cyy /= nf;
    cxy /= nf;

    let trace = cxx + cyy;
    if trace <= 1e-12 || !trace.is_finite() {
        return None;
    }

    // 2x2 symmetric eigenvalues:
    //   lambda_{1,2} = (cxx+cyy)/2  +/-  sqrt(((cxx-cyy)/2)^2 + cxy^2)
    let half_sum = 0.5 * trace;
    let half_diff = 0.5 * (cxx - cyy);
    let radius = (half_diff * half_diff + cxy * cxy).sqrt();
    let lambda_major = half_sum + radius;
    let lambda_minor = (half_sum - radius).max(0.0);
    let denom = lambda_major + lambda_minor;
    if denom <= 1e-12 {
        return None;
    }
    let major_frac = lambda_major / denom;

    // Principal-axis angle: major eigenvector direction.
    // For a 2x2 symmetric covariance, the angle is 0.5 * atan2(2*cxy, cxx - cyy).
    let theta = 0.5 * (2.0 * cxy).atan2(cxx - cyy);
    Some((theta, major_frac))
}

/// Rotate the camera's `up` vector by `roll_rad` radians about its forward
/// axis, then rebuild the cached orthonormal basis.
///
/// Positive `roll_rad` rotates the image such that the horizontal axis
/// picks up content from the original `+roll_rad` direction in image space.
fn roll_camera(cam: &mut PerspectiveCamera, roll_rad: f64) {
    let (s, c) = roll_rad.sin_cos();
    // Rotate about `forward`: new_up = cos * cam_up + sin * (forward x cam_up).
    // With `right = forward x up` (same sign convention as `build_frame`),
    // `forward x cam_up = -right`, so new_up = cos*cam_up - sin*right.
    let new_up = cam.cam_up * c - cam.right * s;
    cam.up = new_up;
    cam.build_frame();
}

fn axis_percentile_f64(values: &mut [f64], pct_low: f64) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len();
    let lo_idx = ((pct_low * n as f64).floor() as usize).min(n - 1);
    let hi_idx = (((1.0 - pct_low) * n as f64).ceil() as usize).saturating_sub(1).min(n - 1);
    values.select_nth_unstable_by(lo_idx, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let lo = values[lo_idx];
    values.select_nth_unstable_by(hi_idx, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let hi = values[hi_idx];
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_bbox() -> BoundingBox {
        BoundingBox {
            min_x: -10.0,
            max_x: 10.0,
            min_y: -10.0,
            max_y: 10.0,
            min_z: -10.0,
            max_z: 10.0,
            width: 20.0,
            height: 20.0,
            depth: 20.0,
        }
    }

    /// A thin 2D orbit that occupies only the central region of the bbox.
    fn thin_orbit() -> Vec<Vec<Vector3<f64>>> {
        let mut body = Vec::new();
        for i in 0..2000 {
            let t = std::f64::consts::TAU * f64::from(i) / 2000.0;
            body.push(Vector3::new(t.cos() * 1.5, t.sin() * 1.5, 0.0));
        }
        vec![body]
    }

    #[test]
    fn fit_to_ink_zooms_tighter_than_orbit_fit() {
        let bbox = test_bbox();
        let positions = thin_orbit();
        let (w, h) = (1920u32, 1080u32);

        let cam_fit = PerspectiveCamera::orbit_fit(&bbox, w, h, 0.90);
        let cam_ink = fit_to_ink_camera(&positions, &bbox, w, h, 0.90, 0.98, Vector3::zeros());

        // The ink-fit camera must be closer than the bbox-fit camera because
        // the actual ink is much smaller than the bbox.
        assert!(cam_ink.focus_distance < cam_fit.focus_distance);
    }

    #[test]
    fn fit_to_ink_fills_configured_fraction() {
        let bbox = test_bbox();
        let positions = thin_orbit();
        let (w, h) = (1920u32, 1080u32);
        let fill = 0.95;

        let cam = fit_to_ink_camera(&positions, &bbox, w, h, fill, 0.98, Vector3::zeros());

        let mut max_ndc_x = 0.0f32;
        let mut max_ndc_y = 0.0f32;
        for body in &positions {
            for &p in body {
                let (px, py, _) = cam.project(w, h, p);
                let nx = (px / w as f32 - 0.5).abs() * 2.0;
                let ny = (py / h as f32 - 0.5).abs() * 2.0;
                max_ndc_x = max_ndc_x.max(nx);
                max_ndc_y = max_ndc_y.max(ny);
            }
        }
        let max_axis = max_ndc_x.max(max_ndc_y);
        // `fit_to_ink` targets the constraining axis; aspect mismatches
        // mean the other axis can be noticeably less filled. The lower
        // bound just checks the ink expands to a respectable fraction of
        // the frame (much better than `orbit_fit`'s bbox-corners
        // envelope) rather than matching `fill` exactly on every axis.
        // The upper bound allows ~5% extra because refit targets the
        // 98% envelope, so the remaining 1% tail can extend past `fill`.
        assert!(max_axis >= 0.65, "fit_to_ink underfills: max_ndc={max_axis}, expected >= 0.65");
        assert!(
            max_axis <= fill as f32 + 0.10,
            "fit_to_ink overfills: max_ndc={max_axis}, expected <= {}",
            fill + 0.10
        );
    }

    #[test]
    fn fit_to_ink_empty_positions_is_stable() {
        let bbox = test_bbox();
        let empty: Vec<Vec<Vector3<f64>>> = Vec::new();
        let cam = fit_to_ink_camera(&empty, &bbox, 1920, 1080, 0.95, 0.98, Vector3::zeros());
        assert!(cam.focus_distance.is_finite());
    }

    #[test]
    fn fit_to_ink_applies_focal_offset() {
        let bbox = test_bbox();
        let positions = thin_orbit();
        let offset = Vector3::new(0.6, 0.2, 0.0);
        let cam_off = fit_to_ink_camera(&positions, &bbox, 1920, 1080, 0.95, 0.98, offset);
        let cam_no_off =
            fit_to_ink_camera(&positions, &bbox, 1920, 1080, 0.95, 0.98, Vector3::zeros());
        assert!((cam_off.target - cam_no_off.target).norm() > 1e-6);
    }

    #[test]
    fn principal_angle_detects_45_deg_line() {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for i in 0..200 {
            let t = (f64::from(i) - 100.0) / 100.0 * 5.0;
            xs.push(t);
            ys.push(t);
        }
        let (theta, frac) = super::principal_angle_rad(&xs, &ys).expect("nondegenerate");
        assert!(
            (theta - std::f64::consts::FRAC_PI_4).abs() < 1e-6,
            "expected PI/4, got {theta}"
        );
        assert!(frac > 0.95, "expected near-degenerate major_frac, got {frac}");
    }

    #[test]
    fn principal_angle_isotropic_returns_near_half() {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for i in 0..2000 {
            let a = std::f64::consts::TAU * f64::from(i) / 2000.0;
            xs.push(a.cos());
            ys.push(a.sin());
        }
        let (_, frac) = super::principal_angle_rad(&xs, &ys).expect("nondegenerate");
        assert!((frac - 0.5).abs() < 0.05, "expected near-isotropic major_frac, got {frac}");
    }

    #[test]
    fn roll_camera_rotates_basis_by_90_degrees() {
        let bbox = test_bbox();
        let mut cam = PerspectiveCamera::orbit_fit(&bbox, 1920, 1080, 0.95);
        let orig_right = cam.right;
        let orig_cam_up = cam.cam_up;
        super::roll_camera(&mut cam, std::f64::consts::FRAC_PI_2);
        // After +PI/2 roll about forward: new right = orig cam_up,
        //                                 new cam_up = -orig right.
        assert!(
            (cam.right - orig_cam_up).norm() < 1e-9,
            "right mismatch: {:?} vs {:?}",
            cam.right,
            orig_cam_up
        );
        assert!(
            (cam.cam_up - (-orig_right)).norm() < 1e-9,
            "cam_up mismatch: {:?} vs {:?}",
            cam.cam_up,
            -orig_right
        );
    }

    #[test]
    fn fit_to_ink_rolls_elongated_ink_to_horizontal() {
        // A thin line in world space aligned with world-Z. The seed camera's
        // `up` is world-Z, so the seed projection is a near-vertical line.
        // After fit-to-ink (which includes roll-align), the projected line
        // must land along the horizontal axis for a 16:9 frame.
        let bbox = BoundingBox {
            min_x: -0.1,
            max_x: 0.1,
            min_y: -0.1,
            max_y: 0.1,
            min_z: -3.0,
            max_z: 3.0,
            width: 0.2,
            height: 0.2,
            depth: 6.0,
        };
        let mut body = Vec::new();
        for i in 0..500 {
            let t = (f64::from(i) / 499.0 - 0.5) * 6.0;
            // Tiny jitter so the minor variance is nonzero (avoids degenerate
            // edge cases in the projection).
            let e = 0.01 * f64::from(i).sin();
            body.push(Vector3::new(e, e, t));
        }
        let positions = vec![body];
        let (w, h) = (1920u32, 1080u32);

        let cam = fit_to_ink_camera(&positions, &bbox, w, h, 0.95, 0.98, Vector3::zeros());

        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for b in &positions {
            for &p in b {
                let (px, py, _) = cam.project(w, h, p);
                xs.push(f64::from(px));
                ys.push(f64::from(py));
            }
        }
        let nf = xs.len() as f64;
        let mx: f64 = xs.iter().sum::<f64>() / nf;
        let my: f64 = ys.iter().sum::<f64>() / nf;
        let mut cxx = 0.0;
        let mut cyy = 0.0;
        for i in 0..xs.len() {
            let dx = xs[i] - mx;
            let dy = ys[i] - my;
            cxx += dx * dx;
            cyy += dy * dy;
        }
        cxx /= nf;
        cyy /= nf;

        assert!(cxx > 3.0 * cyy, "ink not rolled to horizontal: cxx={cxx}, cyy={cyy}");
    }

    /// Project all positions through `cam` and return the maximum absolute
    /// NDC value on each axis (`|2*px/w - 1|` and `|2*py/h - 1|`).
    fn max_ndc(
        cam: &PerspectiveCamera,
        positions: &[Vec<Vector3<f64>>],
        w: u32,
        h: u32,
    ) -> (f32, f32) {
        let mut max_ndc_x = 0.0f32;
        let mut max_ndc_y = 0.0f32;
        for body in positions {
            for &p in body {
                let (px, py, _) = cam.project(w, h, p);
                let nx = (px / w as f32 - 0.5).abs() * 2.0;
                let ny = (py / h as f32 - 0.5).abs() * 2.0;
                max_ndc_x = max_ndc_x.max(nx);
                max_ndc_y = max_ndc_y.max(ny);
            }
        }
        (max_ndc_x, max_ndc_y)
    }

    /// Regression: when drift inflates the seeding bbox so the actual ink
    /// occupies only ~1/6th of the seed viewport, the iterated refit must
    /// still fill the constraining axis to within a small tolerance of the
    /// configured `fill`. The legacy two-pass implementation left ~35% fill
    /// for seeds like `0xb6dbcf780360` (drift.scale = 1.87).
    #[test]
    fn fit_to_ink_fills_when_drift_inflates_bbox() {
        // Thin orbit of radius 1.5 (diameter 3) embedded in a bbox 3x larger
        // per axis, simulating a drift-inflated percentile bbox.
        let bbox = BoundingBox {
            min_x: -9.0,
            max_x: 9.0,
            min_y: -9.0,
            max_y: 9.0,
            min_z: -9.0,
            max_z: 9.0,
            width: 18.0,
            height: 18.0,
            depth: 18.0,
        };
        let positions = thin_orbit();
        let (w, h) = (1920u32, 1080u32);
        let fill = 0.95;

        let cam = fit_to_ink_camera(&positions, &bbox, w, h, fill, 0.98, Vector3::zeros());
        let (mx, my) = max_ndc(&cam, &positions, w, h);
        let constraining = mx.max(my);

        assert!(
            constraining >= 0.90,
            "drift-inflated bbox underfilled: max_ndc=({mx},{my}), expected constraining >= 0.90"
        );
        assert!(
            constraining <= fill as f32 + 0.06,
            "drift-inflated bbox overshot: max_ndc=({mx},{my}), expected <= {}",
            fill + 0.06
        );
    }

    /// Regression: the legacy `zoom.clamp(0.2, 12.0)` upper bound could bind
    /// when the seeding bbox was an extreme multiple of the projected ink,
    /// leaving a single pass far short of the target fill. The new 64x clamp
    /// combined with convergence iteration must close this gap even at a
    /// 30x ratio.
    #[test]
    fn fit_to_ink_iterates_past_zoom_clamp_regression() {
        let bbox = BoundingBox {
            min_x: -45.0,
            max_x: 45.0,
            min_y: -45.0,
            max_y: 45.0,
            min_z: -45.0,
            max_z: 45.0,
            width: 90.0,
            height: 90.0,
            depth: 90.0,
        };
        let positions = thin_orbit();
        let (w, h) = (1920u32, 1080u32);
        let fill = 0.95;

        let cam = fit_to_ink_camera(&positions, &bbox, w, h, fill, 0.98, Vector3::zeros());
        let (mx, my) = max_ndc(&cam, &positions, w, h);
        let constraining = mx.max(my);

        assert!(
            constraining >= 0.85,
            "30x bbox underfilled: max_ndc=({mx},{my}), expected constraining >= 0.85"
        );
    }

    /// Regression: after the iterated refit + final recenter, the projected
    /// ink centroid must land within a small fraction of the frame centre,
    /// even when the seeding bbox is drift-inflated.
    #[test]
    fn fit_to_ink_centers_subject_after_refit() {
        let bbox = BoundingBox {
            min_x: -9.0,
            max_x: 9.0,
            min_y: -9.0,
            max_y: 9.0,
            min_z: -9.0,
            max_z: 9.0,
            width: 18.0,
            height: 18.0,
            depth: 18.0,
        };
        let positions = thin_orbit();
        let (w, h) = (1920u32, 1080u32);

        let cam = fit_to_ink_camera(&positions, &bbox, w, h, 0.95, 0.98, Vector3::zeros());
        let extent =
            measure_ink_pixel_extent(&cam, &positions, w, h, 0.98).expect("positions non-empty");
        let cx_err = (extent.center_x() / f64::from(w) - 0.5).abs();
        let cy_err = (extent.center_y() / f64::from(h) - 0.5).abs();

        // Within 2% of frame centre on both axes after the final recentre
        // pass. 2% is below visual perception (~20 px on a 1080 axis) and
        // accounts for the linear-approximation error in the single-pass
        // recentre on perspective-projected ink.
        assert!(cx_err < 0.02, "x centring error too large: {cx_err}");
        assert!(cy_err < 0.02, "y centring error too large: {cy_err}");
    }
}
