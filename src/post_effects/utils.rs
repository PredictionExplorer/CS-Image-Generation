//! Shared utilities for active post-processing effects.

/// Hermite smoothstep interpolation between two edges.
#[inline]
pub(super) fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    if (edge1 - edge0).abs() < f64::EPSILON {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Soft-knee highlight extraction factor based on luminance.
#[inline]
pub(super) fn highlight_extract_factor(luminance: f64) -> f64 {
    let knee = crate::render::constants::DEFAULT_HIGHLIGHT_EXTRACT_KNEE;
    let threshold = crate::render::constants::DEFAULT_HIGHLIGHT_EXTRACT_THRESHOLD;
    smoothstep(threshold - knee * 0.5, threshold + knee * 0.5, luminance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoothstep_below_edge0() {
        assert_eq!(smoothstep(0.0, 1.0, -0.5), 0.0);
    }

    #[test]
    fn smoothstep_above_edge1() {
        assert_eq!(smoothstep(0.0, 1.0, 1.5), 1.0);
    }

    #[test]
    fn smoothstep_at_midpoint() {
        let mid = smoothstep(0.0, 1.0, 0.5);
        assert!((mid - 0.5).abs() < 1e-10);
    }

    #[test]
    fn smoothstep_at_edges() {
        assert_eq!(smoothstep(0.0, 1.0, 0.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 1.0), 1.0);
    }

    #[test]
    fn smoothstep_equal_edges() {
        assert_eq!(smoothstep(0.5, 0.5, 0.3), 0.0);
        assert_eq!(smoothstep(0.5, 0.5, 0.5), 1.0);
        assert_eq!(smoothstep(0.5, 0.5, 0.7), 1.0);
    }

    #[test]
    fn highlight_extract_factor_zero_luminance() {
        let f = highlight_extract_factor(0.0);
        assert_eq!(f, 0.0);
    }

    #[test]
    fn highlight_extract_factor_high_luminance() {
        let f = highlight_extract_factor(1.0);
        assert!((f - 1.0).abs() < 1e-6);
    }

    #[test]
    fn highlight_extract_factor_monotonic() {
        let mut prev = highlight_extract_factor(0.0);
        for i in 1..=100 {
            let lum = f64::from(i) / 100.0;
            let cur = highlight_extract_factor(lum);
            assert!(cur >= prev - 1e-10, "not monotonic at lum={lum}");
            prev = cur;
        }
    }
}
