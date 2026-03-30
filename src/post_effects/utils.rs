//! Shared utilities for post-processing effects.

use super::{CameraOrientation, EffectContext, PixelBuffer};
use rayon::prelude::*;

/// Map a screen pixel to world-stable coordinates via inverse camera projection.
///
/// When the `EffectContext` carries both a current and reference camera, the
/// pixel is unprojected through the current camera into a world-space direction,
/// then re-projected through the reference camera to yield coordinates that are
/// stable under camera rotation. When camera data is absent, returns the raw
/// pixel position unchanged.
#[inline]
pub fn stable_pattern_coords(
    px: usize,
    py: usize,
    width: usize,
    height: usize,
    ctx: &EffectContext,
) -> (f64, f64) {
    match (&ctx.current_camera, &ctx.reference_camera) {
        (Some(cur), Some(ref_cam)) => {
            reproject_pixel(px, py, width, height, cur, ref_cam)
        }
        _ => (px as f64, py as f64),
    }
}

fn reproject_pixel(
    px: usize,
    py: usize,
    width: usize,
    height: usize,
    cur: &CameraOrientation,
    ref_cam: &CameraOrientation,
) -> (f64, f64) {
    let aspect = width as f64 / height as f64;
    let ndc_x = ((px as f64 + 0.5) / width as f64 * 2.0 - 1.0) * aspect * cur.half_fov_tan;
    let ndc_y = ((py as f64 + 0.5) / height as f64 * 2.0 - 1.0) * cur.half_fov_tan;

    let wx = cur.fwd[0] + cur.right[0] * ndc_x + cur.up[0] * ndc_y;
    let wy = cur.fwd[1] + cur.right[1] * ndc_x + cur.up[1] * ndc_y;
    let wz = cur.fwd[2] + cur.right[2] * ndc_x + cur.up[2] * ndc_y;

    let ref_z = wx * ref_cam.fwd[0] + wy * ref_cam.fwd[1] + wz * ref_cam.fwd[2];

    if ref_z.abs() < 1e-8 {
        return (px as f64, py as f64);
    }

    let ref_x = wx * ref_cam.right[0] + wy * ref_cam.right[1] + wz * ref_cam.right[2];
    let ref_y = wx * ref_cam.up[0] + wy * ref_cam.up[1] + wz * ref_cam.up[2];

    let ref_ndc_x = ref_x / ref_z;
    let ref_ndc_y = ref_y / ref_z;

    let stable_x = (ref_ndc_x / (aspect * ref_cam.half_fov_tan) + 1.0) * 0.5 * width as f64;
    let stable_y = (ref_ndc_y / ref_cam.half_fov_tan + 1.0) * 0.5 * height as f64;

    (stable_x, stable_y)
}

/// A simple 2D hash to generate pseudo-random points for distance fields.
#[inline]
pub fn hash2(p: (f64, f64)) -> (f64, f64) {
    let h = (p.0 * 12.9898 + p.1 * 78.233).sin() * 43758.5453;
    (h.fract(), (h * 1.57).fract())
}

/// Hermite smoothstep interpolation between two edges.
#[inline]
pub fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    if (edge1 - edge0).abs() < f64::EPSILON {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Soft-knee highlight extraction factor based on luminance.
#[inline]
pub fn highlight_extract_factor(luminance: f64) -> f64 {
    let knee = crate::render::constants::DEFAULT_HIGHLIGHT_EXTRACT_KNEE;
    let threshold = crate::render::constants::DEFAULT_HIGHLIGHT_EXTRACT_THRESHOLD;
    smoothstep(threshold - knee * 0.5, threshold + knee * 0.5, luminance)
}

/// Computes luminance gradients for a given pixel buffer, used for flow-aware effects.
pub fn calculate_gradients(buffer: &PixelBuffer, width: usize, height: usize) -> Vec<(f64, f64)> {
    let mut luminance = vec![0.0f64; buffer.len()];
    luminance.par_iter_mut().enumerate().for_each(|(idx, lum)| {
        let (r, g, b, a) = buffer[idx];
        *lum = if a > 0.0 { (0.2126 * r + 0.7152 * g + 0.0722 * b) / a } else { 0.0 };
    });

    let mut gradients = vec![(0.0, 0.0); buffer.len()];
    gradients.par_iter_mut().enumerate().for_each(|(idx, grad)| {
        let x = (idx % width) as isize;
        let y = (idx / width) as isize;
        let sample = |sx: isize, sy: isize| -> f64 {
            let sx = sx.clamp(0, width as isize - 1);
            let sy = sy.clamp(0, height as isize - 1);
            luminance[sy as usize * width + sx as usize]
        };
        let gx = sample(x + 1, y) - sample(x - 1, y);
        let gy = sample(x, y + 1) - sample(x, y - 1);

        // Normalize gradients to avoid overpowering chroma modulation
        let magnitude = (gx * gx + gy * gy).sqrt().max(1e-5);
        let scale = (0.85 / magnitude).min(1.0);

        *grad = (gx * scale, gy * scale);
    });
    gradients
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_camera() -> CameraOrientation {
        CameraOrientation {
            right: [1.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fwd: [0.0, 0.0, 1.0],
            half_fov_tan: 0.4663,
        }
    }

    fn yaw_30_camera() -> CameraOrientation {
        let c = 30.0_f64.to_radians().cos();
        let s = 30.0_f64.to_radians().sin();
        CameraOrientation {
            right: [-c, 0.0, s],
            up: [0.0, 1.0, 0.0],
            fwd: [s, 0.0, c],
            half_fov_tan: 0.4663,
        }
    }

    fn identity_ctx() -> EffectContext {
        EffectContext {
            current_camera: Some(identity_camera()),
            reference_camera: Some(identity_camera()),
        }
    }

    fn rotated_ctx() -> EffectContext {
        EffectContext {
            current_camera: Some(yaw_30_camera()),
            reference_camera: Some(identity_camera()),
        }
    }

    // ── stable_pattern_coords ────────────────────────────────────

    #[test]
    fn test_stable_coords_no_camera_returns_raw_pixel() {
        let ctx = EffectContext::default();
        let (x, y) = stable_pattern_coords(42, 17, 200, 100, &ctx);
        assert_eq!(x, 42.0);
        assert_eq!(y, 17.0);
    }

    #[test]
    fn test_stable_coords_only_current_camera_returns_raw_pixel() {
        let ctx = EffectContext {
            current_camera: Some(identity_camera()),
            reference_camera: None,
        };
        let (x, y) = stable_pattern_coords(10, 20, 100, 100, &ctx);
        assert_eq!(x, 10.0);
        assert_eq!(y, 20.0);
    }

    #[test]
    fn test_stable_coords_only_reference_camera_returns_raw_pixel() {
        let ctx = EffectContext {
            current_camera: None,
            reference_camera: Some(identity_camera()),
        };
        let (x, y) = stable_pattern_coords(10, 20, 100, 100, &ctx);
        assert_eq!(x, 10.0);
        assert_eq!(y, 20.0);
    }

    #[test]
    fn test_stable_coords_identity_is_near_pixel_center() {
        let ctx = identity_ctx();
        for py in [0, 25, 49, 50, 99] {
            for px in [0, 25, 49, 50, 99] {
                let (sx, sy) = stable_pattern_coords(px, py, 100, 100, &ctx);
                assert!(
                    (sx - (px as f64 + 0.5)).abs() < 1e-8,
                    "px={px}: expected ~{}, got {sx}",
                    px as f64 + 0.5
                );
                assert!(
                    (sy - (py as f64 + 0.5)).abs() < 1e-8,
                    "py={py}: expected ~{}, got {sy}",
                    py as f64 + 0.5
                );
            }
        }
    }

    #[test]
    fn test_stable_coords_identity_non_square() {
        let ctx = EffectContext {
            current_camera: Some(identity_camera()),
            reference_camera: Some(identity_camera()),
        };
        for py in [0, 54, 107] {
            for px in [0, 96, 191] {
                let (sx, sy) = stable_pattern_coords(px, py, 192, 108, &ctx);
                assert!(
                    (sx - (px as f64 + 0.5)).abs() < 1e-7,
                    "non-square px={px}: expected ~{}, got {sx}",
                    px as f64 + 0.5
                );
                assert!(
                    (sy - (py as f64 + 0.5)).abs() < 1e-7,
                    "non-square py={py}: expected ~{}, got {sy}",
                    py as f64 + 0.5
                );
            }
        }
    }

    #[test]
    fn test_stable_coords_rotated_differs_from_identity() {
        let id_ctx = identity_ctx();
        let rot_ctx = rotated_ctx();

        let (ix, _iy) = stable_pattern_coords(30, 50, 100, 100, &id_ctx);
        let (rx, ry) = stable_pattern_coords(30, 50, 100, 100, &rot_ctx);

        assert!(
            (ix - rx).abs() > 1.0,
            "30° yaw should shift x significantly: id={ix}, rot={rx}"
        );
        assert!(
            ry.is_finite(),
            "y should remain finite under yaw rotation"
        );
    }

    #[test]
    fn test_stable_coords_center_pixel_stays_centered_for_identity() {
        let ctx = identity_ctx();
        let (cx, cy) = stable_pattern_coords(50, 50, 101, 101, &ctx);
        assert!((cx - 50.5).abs() < 1e-7, "center x: {cx}");
        assert!((cy - 50.5).abs() < 1e-7, "center y: {cy}");
    }

    #[test]
    fn test_stable_coords_all_pixels_finite() {
        let ctx = rotated_ctx();
        let w = 64;
        let h = 48;
        for py in 0..h {
            for px in 0..w {
                let (sx, sy) = stable_pattern_coords(px, py, w, h, &ctx);
                assert!(sx.is_finite(), "NaN/Inf at ({px},{py}): sx={sx}");
                assert!(sy.is_finite(), "NaN/Inf at ({px},{py}): sy={sy}");
            }
        }
    }

    #[test]
    fn test_stable_coords_deterministic() {
        let ctx = rotated_ctx();
        let (a1, b1) = stable_pattern_coords(33, 44, 100, 100, &ctx);
        let (a2, b2) = stable_pattern_coords(33, 44, 100, 100, &ctx);
        assert_eq!(a1.to_bits(), a2.to_bits());
        assert_eq!(b1.to_bits(), b2.to_bits());
    }

    #[test]
    fn test_stable_coords_symmetric_around_center() {
        let ctx = identity_ctx();
        let w = 100;
        let h = 100;
        let (lx, _) = stable_pattern_coords(20, 50, w, h, &ctx);
        let (rx, _) = stable_pattern_coords(79, 50, w, h, &ctx);
        assert!(
            ((lx + rx) / 2.0 - 50.0).abs() < 0.6,
            "mirror pixels should average to center: lx={lx}, rx={rx}"
        );
    }

    #[test]
    fn test_reproject_pixel_near_singularity_falls_back() {
        let cur = CameraOrientation {
            right: [0.0, 0.0, -1.0],
            up: [0.0, 1.0, 0.0],
            fwd: [1.0, 0.0, 0.0],
            half_fov_tan: 0.4663,
        };
        let ref_cam = identity_camera();
        let (sx, sy) = reproject_pixel(50, 50, 100, 100, &cur, &ref_cam);
        assert!(sx.is_finite() && sy.is_finite(), "should not produce NaN at singularity");
    }

    // ── existing tests below ─────────────────────────────────────

    #[test]
    fn test_hash2_returns_finite_values() {
        for i in 0..100 {
            let (x, y) = hash2((i as f64 * 1.7, i as f64 * 3.1));
            assert!(x.is_finite(), "hash2 x={x} not finite");
            assert!(y.is_finite(), "hash2 y={y} not finite");
        }
    }

    #[test]
    fn test_hash2_deterministic() {
        let (a1, b1) = hash2((42.0, 99.0));
        let (a2, b2) = hash2((42.0, 99.0));
        assert_eq!(a1.to_bits(), a2.to_bits());
        assert_eq!(b1.to_bits(), b2.to_bits());
    }

    #[test]
    fn test_smoothstep_edges() {
        assert!((smoothstep(0.0, 1.0, 0.0) - 0.0).abs() < 1e-12);
        assert!((smoothstep(0.0, 1.0, 1.0) - 1.0).abs() < 1e-12);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_smoothstep_clamps() {
        assert!((smoothstep(0.0, 1.0, -1.0) - 0.0).abs() < 1e-12);
        assert!((smoothstep(0.0, 1.0, 2.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_smoothstep_degenerate_edges() {
        let result = smoothstep(0.5, 0.5, 0.5);
        assert!(!result.is_nan(), "degenerate edges should not produce NaN");
    }

    #[test]
    fn test_highlight_extract_factor_range() {
        for i in 0..=20 {
            let lum = i as f64 * 0.1;
            let f = highlight_extract_factor(lum);
            assert!((0.0..=1.0).contains(&f), "extract_factor({lum}) = {f} out of [0,1]");
        }
    }

    #[test]
    fn test_highlight_extract_monotonic() {
        let mut prev = highlight_extract_factor(0.0);
        for i in 1..=20 {
            let lum = i as f64 * 0.1;
            let curr = highlight_extract_factor(lum);
            assert!(curr >= prev, "extract_factor should be monotonic: f({lum}) = {curr} < f(prev) = {prev}");
            prev = curr;
        }
    }

    #[test]
    fn test_calculate_gradients_dimensions() {
        let buf: PixelBuffer = vec![(0.5, 0.5, 0.5, 1.0); 16 * 12];
        let grads = calculate_gradients(&buf, 16, 12);
        assert_eq!(grads.len(), 16 * 12);
    }

    #[test]
    fn test_calculate_gradients_uniform_is_zero() {
        let buf: PixelBuffer = vec![(0.4, 0.4, 0.4, 1.0); 8 * 8];
        let grads = calculate_gradients(&buf, 8, 8);
        for &(gx, gy) in &grads {
            assert!(gx.abs() < 1e-10 && gy.abs() < 1e-10, "uniform buffer should have zero gradient");
        }
    }
}
