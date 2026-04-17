//! Catmull–Rom interpolation for fractional simulation times (motion blur / smooth draws).

use nalgebra::Vector3;

/// Catmull–Rom sample on uniform knots `p0,p1,p2,p3` at `t ∈ [0,1]` between `p1` and `p2`.
#[must_use]
pub fn catmull_rom(
    p0: Vector3<f64>,
    p1: Vector3<f64>,
    p2: Vector3<f64>,
    p3: Vector3<f64>,
    t: f64,
) -> Vector3<f64> {
    let t = t.clamp(0.0, 1.0);
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * (2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

/// Interpolate one body’s position at global time `u` in `[0, n-1]` (continuous index).
#[must_use]
pub fn interpolate_body_at(pos: &[Vector3<f64>], u: f64) -> Vector3<f64> {
    let n = pos.len();
    if n == 0 {
        return Vector3::zeros();
    }
    if n == 1 {
        return pos[0];
    }
    let max_u = (n - 1) as f64;
    let u = u.clamp(0.0, max_u);
    let i = u.floor() as usize;
    let t = u - i as f64;
    // Special case: n == 2 — fall back to linear interpolation.
    if n == 2 {
        return pos[0] + (pos[1] - pos[0]) * t;
    }
    if i >= n - 1 {
        return pos[n - 1];
    }
    if i == 0 {
        // Mirror p0 to extend the sample to 4 knots.
        return catmull_rom(pos[0], pos[0], pos[1], pos[2], t);
    }
    if i >= n - 2 {
        return catmull_rom(pos[n - 3], pos[n - 2], pos[n - 1], pos[n - 1], t);
    }
    catmull_rom(pos[i - 1], pos[i], pos[i + 1], pos[i + 2], t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64, y: f64, z: f64) -> Vector3<f64> {
        Vector3::new(x, y, z)
    }

    #[test]
    fn test_catmull_rom_endpoints_pass_through_p1_and_p2() {
        let p0 = v(-1.0, 0.0, 0.0);
        let p1 = v(0.0, 0.0, 0.0);
        let p2 = v(1.0, 1.0, 0.0);
        let p3 = v(2.0, 1.0, 0.0);
        assert!((catmull_rom(p0, p1, p2, p3, 0.0) - p1).norm() < 1e-12);
        assert!((catmull_rom(p0, p1, p2, p3, 1.0) - p2).norm() < 1e-12);
    }

    #[test]
    fn test_catmull_rom_matches_line_for_collinear_knots() {
        let p0 = v(0.0, 0.0, 0.0);
        let p1 = v(1.0, 1.0, 1.0);
        let p2 = v(2.0, 2.0, 2.0);
        let p3 = v(3.0, 3.0, 3.0);
        let mid = catmull_rom(p0, p1, p2, p3, 0.5);
        assert!((mid - v(1.5, 1.5, 1.5)).norm() < 1e-12);
    }

    #[test]
    fn test_interpolate_body_at_clamps_and_hits_samples() {
        let pos = vec![v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(2.0, 0.0, 0.0), v(3.0, 0.0, 0.0)];
        assert!((interpolate_body_at(&pos, -5.0) - pos[0]).norm() < 1e-12);
        assert!((interpolate_body_at(&pos, 99.0) - pos[3]).norm() < 1e-12);
        let mid = interpolate_body_at(&pos, 1.5);
        assert!((mid.x - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_interpolate_body_at_handles_tiny_arrays() {
        let empty: Vec<Vector3<f64>> = vec![];
        assert_eq!(interpolate_body_at(&empty, 0.0), Vector3::zeros());
        let single = vec![v(7.0, 8.0, 9.0)];
        assert_eq!(interpolate_body_at(&single, 0.0), v(7.0, 8.0, 9.0));
        let two = vec![v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0)];
        let mid = interpolate_body_at(&two, 0.5);
        assert!((mid - v(0.5, 0.0, 0.0)).norm() < 1e-6);
    }
}
