//! Rule-of-thirds composition helpers.
//!
//! The camera normally targets the **centroid** of the scene, producing
//! symmetric, static compositions. Gallery photography instead places the
//! subject on a rule-of-thirds intersection point. This module derives a
//! small deterministic world-space offset from the seed RNG that nudges the
//! perspective camera target toward one of the four 3:3 grid intersections.
//!
//! The offset is intentionally small (≈ 1/3 of the shorter ink dimension)
//! so the fit-to-ink camera still keeps the trajectory inside the safe area.

use super::context::BoundingBox;
use crate::sim::Sha3RandomByteStream;
use nalgebra::Vector3;

/// Fraction of the shorter bbox axis by which the target is shifted.
///
/// `0.18` shifts the subject to roughly the 1/3 intersection when combined
/// with the downstream fit-to-ink zoom that occupies ~95% of the frame.
const COMPOSITION_OFFSET_FRACTION: f64 = 0.18;

/// Return a deterministic rule-of-thirds shift for the camera target, in
/// world units, expressed as a 3D vector in the XY plane (Z is zero).
///
/// The RNG choice is pulled from `rng` once; callers already perform a
/// single pull for this at a stable position in the seed stream so the
/// choice is reproducible across runs.
#[must_use]
pub fn rule_of_thirds_offset(bbox: &BoundingBox, rng: &mut Sha3RandomByteStream) -> Vector3<f64> {
    let short_axis = bbox.width.min(bbox.height).max(1e-6);
    let amp = short_axis * COMPOSITION_OFFSET_FRACTION;

    // 4 intersection points on a 3x3 grid centered at origin:
    //   (-1/3, +1/3), (+1/3, +1/3), (-1/3, -1/3), (+1/3, -1/3)
    // Encoded as 4 sign pairs; two bits from a single RNG draw pick one.
    let r = rng.next_f64();
    let idx = (r * 4.0).floor().clamp(0.0, 3.0) as usize;
    let (sx, sy) = match idx {
        0 => (-1.0, 1.0),
        1 => (1.0, 1.0),
        2 => (-1.0, -1.0),
        _ => (1.0, -1.0),
    };

    Vector3::new(sx * amp, sy * amp, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_bbox() -> BoundingBox {
        BoundingBox {
            min_x: -1.0,
            max_x: 1.0,
            min_y: -1.0,
            max_y: 1.0,
            min_z: 0.0,
            max_z: 0.0,
            width: 2.0,
            height: 2.0,
            depth: 1e-12,
        }
    }

    #[test]
    fn offset_is_bounded_by_fraction_of_short_axis() {
        let bbox = unit_bbox();
        let mut rng = Sha3RandomByteStream::new(&[0xAB, 0xCD], 100.0, 300.0, 300.0, 1.0);
        let v = rule_of_thirds_offset(&bbox, &mut rng);
        let bound = 2.0 * COMPOSITION_OFFSET_FRACTION + 1e-9;
        assert!(v.x.abs() <= bound);
        assert!(v.y.abs() <= bound);
        assert_eq!(v.z, 0.0);
    }

    #[test]
    fn offset_is_deterministic_per_seed() {
        let bbox = unit_bbox();
        let mut rng_a = Sha3RandomByteStream::new(&[1, 2, 3, 4], 100.0, 300.0, 300.0, 1.0);
        let mut rng_b = Sha3RandomByteStream::new(&[1, 2, 3, 4], 100.0, 300.0, 300.0, 1.0);
        let a = rule_of_thirds_offset(&bbox, &mut rng_a);
        let b = rule_of_thirds_offset(&bbox, &mut rng_b);
        assert_eq!(a, b);
    }

    #[test]
    fn offset_covers_all_four_quadrants() {
        let bbox = unit_bbox();
        let mut seen = [false; 4];
        for byte in 0u8..=255 {
            let mut rng = Sha3RandomByteStream::new(&[byte], 100.0, 300.0, 300.0, 1.0);
            let v = rule_of_thirds_offset(&bbox, &mut rng);
            let i = match (v.x > 0.0, v.y > 0.0) {
                (false, true) => 0,
                (true, true) => 1,
                (false, false) => 2,
                (true, false) => 3,
            };
            seen[i] = true;
        }
        assert!(seen.iter().all(|&s| s), "all four quadrants should be sampled");
    }
}
