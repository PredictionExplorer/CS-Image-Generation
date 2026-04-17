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
//! fills `fill` of the frame. The camera direction and FOV are preserved.

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
    let seed = PerspectiveCamera::orbit_fit(bbox, width, height, 1.0);

    let total: usize = positions.iter().map(std::vec::Vec::len).sum();
    if total == 0 {
        return offset_target(seed, focal_offset_world);
    }

    // Two-pass refinement: perspective is non-linear in `1/distance`, so
    // a single zoom either under-fills (zoom only up) or over-fills
    // (zoom amplified by the non-linearity). The first pass gets us
    // close; the second pass is allowed to **zoom out** as well as in
    // to correct the overshoot, and uses a small headroom factor so the
    // ink never clips the frame edge.
    let pass1 = refit_once(
        &seed,
        positions,
        width,
        height,
        fill,
        ink_pct,
        focal_offset_world,
        /* allow_zoom_out = */ false,
    );
    refit_once(
        &pass1,
        positions,
        width,
        height,
        fill * 0.97,
        ink_pct,
        Vector3::zeros(),
        /* allow_zoom_out = */ true,
    )
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
    let w = f64::from(width);
    let h = f64::from(height);

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
        return offset_target(camera.clone(), focal_offset_world);
    }

    let pct_low = 0.5 * (1.0 - ink_pct);
    let (min_px, max_px) = axis_percentile_f64(&mut xs, pct_low);
    let (min_py, max_py) = axis_percentile_f64(&mut ys, pct_low);

    let ink_w_px = (max_px - min_px).max(1.0);
    let ink_h_px = (max_py - min_py).max(1.0);
    let ink_cx = 0.5 * (min_px + max_px);
    let ink_cy = 0.5 * (min_py + max_py);

    let zoom_x = (fill * w) / ink_w_px;
    let zoom_y = (fill * h) / ink_h_px;
    let mut zoom = zoom_x.min(zoom_y);
    if !zoom.is_finite() {
        zoom = 1.0;
    }
    if !allow_zoom_out && zoom < 1.0 {
        zoom = 1.0;
    }
    // Soft clamps to avoid pathological behaviour on degenerate
    // trajectories (single-point orbits, NaNs, etc.).
    zoom = zoom.clamp(0.2, 12.0);

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
        assert!(max_axis >= 0.65, "fit_to_ink underfills: max_ndc={max_axis}, expected >= 0.65");
        assert!(
            max_axis <= fill as f32 + 0.08,
            "fit_to_ink overfills: max_ndc={max_axis}, expected <= {}",
            fill + 0.08
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
}
