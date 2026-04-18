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
//!
//! Museum-quality framing invariants (enforced by `fit_to_ink_camera`):
//! * **Viewport occupancy** — at least [`MIN_VIEWPORT_OCCUPANCY`] (95%) of
//!   trajectory samples project inside the framebuffer. Catches the
//!   "tiny splat in one corner" failure mode where a drift-inflated
//!   orbit would otherwise leave most of the ink off-screen.
//! * **Centroid centring** — the percentile-trimmed ink centroid sits
//!   within [`MAX_CENTROID_OFFSET_FRAC`] (2%) of frame centre on both
//!   axes.
//!
//! If the primary `fit_to_ink` attempt misses either invariant, the
//! target `fill` is progressively shrunk (×[`SHRINK_FACTOR`] per attempt,
//! up to [`MAX_SHRINK_ATTEMPTS`] times) and the fit is retried. Each
//! shrink step uniformly zooms the camera **out**, so more of the ink
//! falls inside the viewport and any off-axis parallax residual also
//! shrinks proportionally. As a terminal safety net, after the
//! retries are exhausted the camera falls back to a corner-safe
//! `orbit_fit` at a conservative fill — because `orbit_fit` puts the
//! entire 3D bbox corners inside the viewport, occupancy is
//! guaranteed by construction.

use super::camera_path::PerspectiveCamera;
use super::context::BoundingBox;
use nalgebra::Vector3;

/// Minimum fraction of trajectory samples that must land inside the
/// framebuffer for a camera to be accepted. Tighter than the legacy 85%
/// so the "tiny splat in one corner" failure mode (seed
/// `0x8f1d87d78ba2`) is rejected before the PNG is ever written.
pub const MIN_VIEWPORT_OCCUPANCY: f64 = 0.95;

/// Maximum offset of the percentile-trimmed ink centroid from the frame
/// centre, measured as a fraction of each axis's length. 2% is ~20 px on
/// a 1080 axis — below visual perception for composition.
pub const MAX_CENTROID_OFFSET_FRAC: f64 = 0.02;

/// Number of progressively-shrinking refit attempts before the corner-safe
/// fallback. At [`SHRINK_FACTOR`] = 0.92, the last attempt sits at ~0.51
/// of the original fill — comfortably loose enough for any non-degenerate
/// orbit.
const MAX_SHRINK_ATTEMPTS: usize = 8;

/// Multiplicative shrink applied to the target fill between retry
/// attempts when an invariant is violated. 0.92 is gentle enough that
/// most seeds pass on the first or second try while still guaranteeing
/// convergence within [`MAX_SHRINK_ATTEMPTS`] attempts.
const SHRINK_FACTOR: f64 = 0.92;

/// Minimum allowed fill during retries. Below this, we stop retrying and
/// fall through to the terminal corner-safe `orbit_fit` fallback.
const MIN_RETRY_FILL: f64 = 0.30;

/// Corner-safe fill used by the terminal `orbit_fit` fallback. Since
/// `orbit_fit` at fill ≤ 1.0 places the 3D bbox corners inside the
/// viewport, every trajectory sample (which by definition lies inside
/// that bbox) is also inside the viewport — occupancy is 1.0 by
/// construction.
const TERMINAL_FALLBACK_FILL: f64 = 0.75;

/// Build a perspective camera that fits the **projected ink** of `positions`
/// into `fill` of the frame, subject to the museum-quality framing
/// invariants ([`MIN_VIEWPORT_OCCUPANCY`] and [`MAX_CENTROID_OFFSET_FRAC`]).
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

    // Try the primary fit with the requested fill. If either invariant
    // fails, progressively shrink `fill` and retry the whole refit —
    // every shrink step pulls the camera farther back and so strictly
    // increases both occupancy and centring headroom. This replaces
    // the legacy one-shot `orbit_fit` fallback with a monotonic
    // convergence loop.
    let mut attempt_fill = fill;
    for _ in 0..MAX_SHRINK_ATTEMPTS {
        let cam = fit_to_ink_primary_attempt(
            positions,
            bbox,
            width,
            height,
            attempt_fill,
            ink_pct,
            focal_offset_world,
        );
        if invariants_met(&cam, positions, width, height, ink_pct) {
            record_framing_telemetry(&cam, positions, width, height, ink_pct);
            return cam;
        }
        attempt_fill *= SHRINK_FACTOR;
        if attempt_fill < MIN_RETRY_FILL {
            break;
        }
    }

    // Terminal safety net: `orbit_fit` at [`TERMINAL_FALLBACK_FILL`] is
    // guaranteed by construction to keep all trajectory samples inside
    // the viewport (bbox corners sit at 75% of the frame; samples are
    // strictly inside the bbox). Iteratively recentre on the ink so the
    // subject lands dead-centre even for off-centre orbits.
    let mut cam = PerspectiveCamera::orbit_fit(bbox, width, height, TERMINAL_FALLBACK_FILL);
    for _ in 0..MAX_RECENTER_ITERS {
        let Some(extent) = measure_ink_pixel_extent(&cam, positions, width, height, ink_pct) else {
            break;
        };
        let dx = extent.center_x() - 0.5 * f64::from(width);
        let dy = extent.center_y() - 0.5 * f64::from(height);
        if dx.abs() < RECENTER_TOL_PX && dy.abs() < RECENTER_TOL_PX {
            break;
        }
        cam = recenter_only(&cam, &extent, width, height);
    }
    record_framing_telemetry(&cam, positions, width, height, ink_pct);
    cam
}

/// Push the converged camera's framing metrics into the
/// process-global [`generation_log::record_telemetry`] slot so the
/// final `GenerationRecord` can include them.
///
/// Skips recording if the trajectory is empty (no meaningful metrics
/// to emit). Silently no-ops when called from test harnesses that
/// don't initialise the telemetry store — the lock is always
/// available, so this is really just a guard against degenerate
/// inputs.
fn record_framing_telemetry(
    camera: &PerspectiveCamera,
    positions: &[Vec<Vector3<f64>>],
    width: u32,
    height: u32,
    ink_pct: f64,
) {
    let total: usize = positions.iter().map(std::vec::Vec::len).sum();
    if total == 0 {
        return;
    }
    let occupancy = measure_viewport_occupancy(camera, positions, width, height);
    let (cx_frac, cy_frac) = measure_ink_pixel_extent(camera, positions, width, height, ink_pct)
        .map_or((0.0, 0.0), |e| {
            (
                (e.center_x() / f64::from(width) - 0.5).abs(),
                (e.center_y() / f64::from(height) - 0.5).abs(),
            )
        });
    crate::generation_log::record_telemetry(|t| {
        t.viewport_occupancy = Some(occupancy);
        t.centroid_offset_x_frac = Some(cx_frac);
        t.centroid_offset_y_frac = Some(cy_frac);
    });
}

/// Run the full roll-align + iterated-refit + recentre pipeline without
/// any fallback. Factored out of [`fit_to_ink_camera`] so the outer
/// loop can retry with a shrunk fill target when an invariant fails.
fn fit_to_ink_primary_attempt(
    positions: &[Vec<Vector3<f64>>],
    bbox: &BoundingBox,
    width: u32,
    height: u32,
    fill: f64,
    ink_pct: f64,
    focal_offset_world: Vector3<f64>,
) -> PerspectiveCamera {
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

    // Final recentre-only phase: lock the subject to dead-centre without
    // touching zoom. Perspective parallax from the last zoom step can leave
    // the projected centre a few pixels off-axis; a single pass is only a
    // *linear* correction (world-per-pixel is exact on the focus plane, but
    // biased when the ink sits off-target by a non-trivial fraction of the
    // focus distance), so we iterate. Each pass is a geometric contraction
    // in the centroid error, so convergence is reached within a handful of
    // iterations even on drift-inflated bboxes.
    for _ in 0..MAX_RECENTER_ITERS {
        let Some(extent) = measure_ink_pixel_extent(&cam, positions, width, height, ink_pct) else {
            break;
        };
        let dx = extent.center_x() - 0.5 * f64::from(width);
        let dy = extent.center_y() - 0.5 * f64::from(height);
        if dx.abs() < RECENTER_TOL_PX && dy.abs() < RECENTER_TOL_PX {
            break;
        }
        cam = recenter_only(&cam, &extent, width, height);
    }

    cam
}

/// Maximum recentre passes after the zoom-convergence loop. Five passes
/// is comfortably above the empirical worst case (3 on a 30x-drift bbox)
/// while keeping the tail bounded on pathological inputs.
const MAX_RECENTER_ITERS: usize = 5;

/// Sub-pixel centroid tolerance: stop iterating `recenter_only` once the
/// projected ink centroid is within half a pixel of the frame centre on
/// both axes. A tighter tolerance would mostly chase float noise in the
/// projection routine.
const RECENTER_TOL_PX: f64 = 0.5;

/// Check both framing invariants: viewport occupancy and centroid centring.
fn invariants_met(
    camera: &PerspectiveCamera,
    positions: &[Vec<Vector3<f64>>],
    width: u32,
    height: u32,
    ink_pct: f64,
) -> bool {
    let occupancy = measure_viewport_occupancy(camera, positions, width, height);
    if occupancy < MIN_VIEWPORT_OCCUPANCY {
        return false;
    }
    // Centroid check: the percentile-trimmed ink centre must sit within
    // `MAX_CENTROID_OFFSET_FRAC` of the frame centre. `None` (empty or
    // degenerate) is treated as passing — occupancy already caught the
    // degeneracy above.
    let Some(extent) = measure_ink_pixel_extent(camera, positions, width, height, ink_pct) else {
        return true;
    };
    let w = f64::from(width);
    let h = f64::from(height);
    let cx_off = (extent.center_x() / w - 0.5).abs();
    let cy_off = (extent.center_y() / h - 0.5).abs();
    cx_off < MAX_CENTROID_OFFSET_FRAC && cy_off < MAX_CENTROID_OFFSET_FRAC
}

/// Fraction of projected trajectory samples that land inside the
/// framebuffer `[0, width] x [0, height]`.
///
/// Returns `1.0` for empty input so that trivial scenes don't trigger
/// the fallback. Infinite or NaN projections count as "outside".
pub fn measure_viewport_occupancy(
    camera: &PerspectiveCamera,
    positions: &[Vec<Vector3<f64>>],
    width: u32,
    height: u32,
) -> f64 {
    let mut total = 0usize;
    let mut inside = 0usize;
    let wf = f64::from(width);
    let hf = f64::from(height);
    for body in positions {
        for &p in body {
            let (px, py, _) = camera.project(width, height, p);
            let pxf = f64::from(px);
            let pyf = f64::from(py);
            total += 1;
            if pxf.is_finite()
                && pyf.is_finite()
                && (0.0..=wf).contains(&pxf)
                && (0.0..=hf).contains(&pyf)
            {
                inside += 1;
            }
        }
    }
    if total == 0 {
        return 1.0;
    }
    inside as f64 / total as f64
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

    /// The core invariant `fit_to_ink_camera` must uphold: for any
    /// non-empty trajectory and any reasonable target fill, the returned
    /// camera has viewport occupancy >= `MIN_VIEWPORT_OCCUPANCY` and the
    /// percentile-trimmed ink centroid is within `MAX_CENTROID_OFFSET_FRAC`
    /// of the frame centre.
    ///
    /// We exercise a mix of shapes (tight orbit, drift-inflated bbox,
    /// elongated line, off-centre cluster) at multiple fill targets to
    /// guarantee the retry + fallback path covers every realistic seed.
    #[test]
    fn fit_to_ink_enforces_framing_invariants() {
        let shapes: Vec<(BoundingBox, Vec<Vec<Vector3<f64>>>)> = vec![
            (test_bbox(), thin_orbit()),
            (
                BoundingBox {
                    min_x: -9.0,
                    max_x: 9.0,
                    min_y: -9.0,
                    max_y: 9.0,
                    min_z: -9.0,
                    max_z: 9.0,
                    width: 18.0,
                    height: 18.0,
                    depth: 18.0,
                },
                thin_orbit(),
            ),
            (
                BoundingBox {
                    min_x: -45.0,
                    max_x: 45.0,
                    min_y: -45.0,
                    max_y: 45.0,
                    min_z: -45.0,
                    max_z: 45.0,
                    width: 90.0,
                    height: 90.0,
                    depth: 90.0,
                },
                thin_orbit(),
            ),
        ];

        let frames: &[(u32, u32)] = &[(1920, 1080), (1080, 1920), (2048, 2048), (800, 600)];
        let fills = [0.80, 0.90, 0.95];

        for (bbox, positions) in &shapes {
            for &(w, h) in frames {
                for &fill in &fills {
                    let cam = fit_to_ink_camera(positions, bbox, w, h, fill, 0.98, Vector3::zeros());
                    let occupancy = measure_viewport_occupancy(&cam, positions, w, h);
                    assert!(
                        occupancy >= MIN_VIEWPORT_OCCUPANCY,
                        "occupancy invariant: {occupancy} < {MIN_VIEWPORT_OCCUPANCY} \
                         (frame={w}x{h}, fill={fill}, bbox_w={:.1})",
                        bbox.width,
                    );
                    let extent = measure_ink_pixel_extent(&cam, positions, w, h, 0.98)
                        .expect("non-empty ink");
                    let cx = (extent.center_x() / f64::from(w) - 0.5).abs();
                    let cy = (extent.center_y() / f64::from(h) - 0.5).abs();
                    assert!(
                        cx < MAX_CENTROID_OFFSET_FRAC && cy < MAX_CENTROID_OFFSET_FRAC,
                        "centroid invariant: ({cx:.4}, {cy:.4}) > {MAX_CENTROID_OFFSET_FRAC} \
                         (frame={w}x{h}, fill={fill})",
                    );
                }
            }
        }
    }

    /// Regression fixture for the `0x8f1d87d78ba2` geometry: a tight main
    /// orbit accompanied by a small outlier splat in a distant corner of
    /// the bbox. The legacy pipeline would zoom aggressively onto the main
    /// orbit and leave the splat as a bright pixel near the frame edge
    /// (the "tiny white blob in the corner" failure the user reported).
    /// The new invariant-driven pipeline must either fit the whole thing
    /// inside the viewport with >= 95% occupancy or zoom out enough to
    /// absorb the outlier.
    #[test]
    fn fit_to_ink_handles_outlier_splat_regression() {
        // Main orbit: small circle centred near the origin.
        let mut body_a = Vec::new();
        for i in 0..2000 {
            let t = std::f64::consts::TAU * f64::from(i) / 2000.0;
            body_a.push(Vector3::new(t.cos() * 1.5, t.sin() * 1.5, 0.0));
        }
        // Outlier body: small cluster 8 units away, in one 3D bbox corner.
        let mut body_b = Vec::new();
        for i in 0..50 {
            let t = std::f64::consts::TAU * f64::from(i) / 50.0;
            body_b.push(Vector3::new(8.0 + 0.2 * t.cos(), 7.5 + 0.2 * t.sin(), 4.0));
        }

        let positions = vec![body_a, body_b];
        let bbox = BoundingBox {
            min_x: -1.6,
            max_x: 8.4,
            min_y: -1.6,
            max_y: 7.9,
            min_z: 0.0,
            max_z: 4.0,
            width: 10.0,
            height: 9.5,
            depth: 4.0,
        };
        let (w, h) = (1920u32, 1080u32);
        let cam = fit_to_ink_camera(&positions, &bbox, w, h, 0.95, 0.98, Vector3::zeros());

        // Invariant: at least 95% of trajectory samples are inside the
        // viewport. The legacy pipeline failed this test by aggressively
        // zooming onto the main orbit and leaving the outlier splat at
        // ~1.0 NDC.
        let occupancy = measure_viewport_occupancy(&cam, &positions, w, h);
        assert!(
            occupancy >= MIN_VIEWPORT_OCCUPANCY,
            "outlier-splat regression: occupancy {occupancy:.3} < {MIN_VIEWPORT_OCCUPANCY}",
        );

        // Centroid of the percentile-trimmed ink (which excludes the 2%
        // tail — the outlier lies in that tail) must still land near the
        // frame centre.
        let extent =
            measure_ink_pixel_extent(&cam, &positions, w, h, 0.98).expect("non-empty ink");
        let cx = (extent.center_x() / f64::from(w) - 0.5).abs();
        let cy = (extent.center_y() / f64::from(h) - 0.5).abs();
        assert!(
            cx < MAX_CENTROID_OFFSET_FRAC && cy < MAX_CENTROID_OFFSET_FRAC,
            "outlier-splat regression: centroid ({cx:.4}, {cy:.4}) \
             > {MAX_CENTROID_OFFSET_FRAC}",
        );
    }

    // Property test: on random combinations of orbit radii, bbox padding
    // factors (drift inflation), fill targets, and frame sizes, the
    // framing invariants must hold for every output camera.
    //
    // The `bbox` is **derived from the positions** (with a random
    // padding factor) rather than sampled independently, so every input
    // faithfully mirrors how the real pipeline constructs its bbox:
    // always a superset of the trajectory, possibly loose (drift) but
    // never tighter than the ink.
    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig {
            cases: 256,
            .. proptest::prelude::ProptestConfig::default()
        })]

        #[test]
        fn proptest_framing_invariants_always_hold(
            orbit_radius in 0.2f64..8.0,
            bbox_padding in 1.1f64..4.0,
            offset_frac_x in -0.4f64..0.4,
            offset_frac_y in -0.4f64..0.4,
            fill in 0.70f64..0.95,
            aspect_is_landscape in proptest::prelude::any::<bool>(),
            frame_size in 600u32..=2048,
        ) {
            let short_side = (f64::from(frame_size) * 0.5625).round() as u32;
            let (w, h) = if aspect_is_landscape {
                (frame_size, short_side)
            } else {
                (short_side, frame_size)
            };
            // Ensure non-degenerate frame.
            let (w, h) = (w.max(256), h.max(256));

            // Place the orbit at a world offset that's a fraction of its
            // own radius — the full [min, max] then feeds the bbox so
            // the bbox always contains the orbit.
            let cx = offset_frac_x * orbit_radius;
            let cy = offset_frac_y * orbit_radius;

            let mut body = Vec::new();
            for i in 0..500 {
                let t = std::f64::consts::TAU * f64::from(i) / 500.0;
                body.push(Vector3::new(
                    cx + t.cos() * orbit_radius,
                    cy + t.sin() * orbit_radius,
                    0.0,
                ));
            }
            let positions = vec![body];

            // Derive the bbox from the positions, then pad by
            // `bbox_padding` to simulate drift-inflated bounding boxes
            // produced by the real pipeline.
            let half = orbit_radius * bbox_padding;
            let bbox = BoundingBox {
                min_x: cx - half,
                max_x: cx + half,
                min_y: cy - half,
                max_y: cy + half,
                min_z: -half,
                max_z: half,
                width: 2.0 * half,
                height: 2.0 * half,
                depth: 2.0 * half,
            };

            let cam = fit_to_ink_camera(&positions, &bbox, w, h, fill, 0.98, Vector3::zeros());
            let occupancy = measure_viewport_occupancy(&cam, &positions, w, h);
            proptest::prop_assert!(
                occupancy >= MIN_VIEWPORT_OCCUPANCY,
                "occupancy invariant: {} < {} (frame={}x{}, fill={}, orbit_r={}, bbox_pad={})",
                occupancy, MIN_VIEWPORT_OCCUPANCY, w, h, fill, orbit_radius, bbox_padding,
            );
            let extent = measure_ink_pixel_extent(&cam, &positions, w, h, 0.98)
                .expect("non-empty ink");
            let cx_off = (extent.center_x() / f64::from(w) - 0.5).abs();
            let cy_off = (extent.center_y() / f64::from(h) - 0.5).abs();
            proptest::prop_assert!(
                cx_off < MAX_CENTROID_OFFSET_FRAC && cy_off < MAX_CENTROID_OFFSET_FRAC,
                "centroid invariant: ({}, {}) > {} (frame={}x{}, fill={})",
                cx_off, cy_off, MAX_CENTROID_OFFSET_FRAC, w, h, fill,
            );
        }
    }
}
