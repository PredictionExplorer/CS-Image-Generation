//! Common rendering context and utilities
//!
//! This module provides the core rendering context that manages coordinate transformations
//! and pixel operations, as well as utilities for iterating through animation frames.

use nalgebra::Vector3;

/// Type alias for pixel buffers used throughout the pipeline.
///
/// Format: `(R, G, B, A)` with premultiplied alpha.
/// Color channels may be linear or display-space depending on the render stage.
pub type PixelBuffer = Vec<(f64, f64, f64, f64)>;

/// Encapsulates common rendering operations and coordinate transformations
#[derive(Debug)]
pub struct RenderContext {
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// Output image width as `usize` (avoids repeated casts).
    pub width_usize: usize,
    /// Output image height as `usize` (avoids repeated casts).
    pub height_usize: usize,
    bounds: BoundingBox,
}

impl RenderContext {
    /// Creates a new render context from position data
    #[must_use]
    pub fn new(
        width: u32,
        height: u32,
        positions: &[Vec<Vector3<f64>>],
        aspect_correction: bool,
    ) -> Self {
        Self::new_with_framing(width, height, positions, aspect_correction, 1.0)
    }

    /// Creates a new render context from position data, optionally padding the
    /// bounding box to leave more negative space around the orbit.
    #[must_use]
    pub fn new_with_framing(
        width: u32,
        height: u32,
        positions: &[Vec<Vector3<f64>>],
        aspect_correction: bool,
        framing_zoom: f64,
    ) -> Self {
        Self::new_with_composition(width, height, positions, aspect_correction, framing_zoom, 0.0, 0.0)
    }

    /// Creates a new render context with framing zoom and bounded center shifts.
    #[must_use]
    pub fn new_with_composition(
        width: u32,
        height: u32,
        positions: &[Vec<Vector3<f64>>],
        aspect_correction: bool,
        framing_zoom: f64,
        framing_shift_x: f64,
        framing_shift_y: f64,
    ) -> Self {
        let mut bounds = BoundingBox::from_positions(positions);
        if aspect_correction {
            bounds.apply_aspect_correction(width, height);
        }
        bounds.apply_framing_zoom(framing_zoom);
        bounds.apply_framing_shift(framing_shift_x, framing_shift_y);

        Self { width, height, width_usize: width as usize, height_usize: height as usize, bounds }
    }

    /// Convert world coordinates to pixel coordinates
    #[must_use]
    #[inline]
    pub fn to_pixel(&self, x: f64, y: f64) -> (f32, f32) {
        self.bounds.world_to_pixel(x, y, self.width, self.height)
    }

    /// Get total pixel count
    #[must_use]
    #[inline]
    pub fn pixel_count(&self) -> usize {
        self.width_usize * self.height_usize
    }

    /// Get the bounding box used for coordinate transformations
    #[must_use]
    #[inline]
    pub fn bounds(&self) -> &BoundingBox {
        &self.bounds
    }
}

/// Bounding box for coordinate transformations
#[derive(Clone, Copy, Debug)]
pub struct BoundingBox {
    /// Minimum x-coordinate of the bounding region.
    pub min_x: f64,
    /// Maximum x-coordinate of the bounding region.
    pub max_x: f64,
    /// Minimum y-coordinate of the bounding region.
    pub min_y: f64,
    /// Maximum y-coordinate of the bounding region.
    pub max_y: f64,
    /// Horizontal span (`max_x - min_x`), clamped to avoid division by zero.
    pub width: f64,
    /// Vertical span (`max_y - min_y`), clamped to avoid division by zero.
    pub height: f64,
}

impl BoundingBox {
    /// Create a new bounding box from position data
    #[must_use]
    pub fn from_positions(positions: &[Vec<Vector3<f64>>]) -> Self {
        let (min_x, max_x, min_y, max_y) = crate::utils::bounding_box(positions);
        Self {
            min_x,
            max_x,
            min_y,
            max_y,
            width: (max_x - min_x).max(1e-12),
            height: (max_y - min_y).max(1e-12),
        }
    }

    /// Convert world coordinates to normalized coordinates (0..1)
    #[must_use]
    #[inline]
    pub fn normalize(&self, x: f64, y: f64) -> (f64, f64) {
        let nx = (x - self.min_x) / self.width;
        let ny = (y - self.min_y) / self.height;
        (nx, ny)
    }

    /// Convert world coordinates to pixel coordinates
    #[must_use]
    #[inline]
    pub fn world_to_pixel(&self, x: f64, y: f64, width: u32, height: u32) -> (f32, f32) {
        let (nx, ny) = self.normalize(x, y);
        let px = nx * f64::from(width);
        let py = ny * f64::from(height);
        (px as f32, py as f32)
    }

    /// Inflate the bounding box symmetrically around its center.
    ///
    /// Values below `1.0` are treated as `1.0` so the orbit is never cropped
    /// more tightly than the legacy framing. Values above `2.0` are clamped to
    /// keep the camera pull-back conservative.
    pub fn apply_framing_zoom(&mut self, factor: f64) {
        if !factor.is_finite() {
            return;
        }
        let factor = factor.clamp(1.0, 2.0);
        if (factor - 1.0).abs() < 1e-9 {
            return;
        }

        let center_x = f64::midpoint(self.min_x, self.max_x);
        let center_y = f64::midpoint(self.min_y, self.max_y);
        let new_width = self.width * factor;
        let new_height = self.height * factor;

        self.min_x = center_x - new_width * 0.5;
        self.max_x = center_x + new_width * 0.5;
        self.min_y = center_y - new_height * 0.5;
        self.max_y = center_y + new_height * 0.5;
        self.width = new_width;
        self.height = new_height;
    }

    /// Shift the framing center within the existing bounds, normalized to the
    /// current width/height so composition can bias toward negative space.
    pub fn apply_framing_shift(&mut self, shift_x: f64, shift_y: f64) {
        if !shift_x.is_finite() || !shift_y.is_finite() {
            return;
        }
        let shift_x = shift_x.clamp(-0.45, 0.45);
        let shift_y = shift_y.clamp(-0.45, 0.45);
        if shift_x.abs() < 1e-9 && shift_y.abs() < 1e-9 {
            return;
        }

        let dx = self.width * shift_x;
        let dy = self.height * shift_y;
        self.min_x += dx;
        self.max_x += dx;
        self.min_y += dy;
        self.max_y += dy;
    }

    /// Pad the bounding box so its aspect ratio matches the target output dimensions.
    /// This prevents orbit distortion (stretching) when the orbit shape doesn't
    /// match the output aspect ratio.
    pub fn apply_aspect_correction(&mut self, target_width: u32, target_height: u32) {
        if target_height == 0 || self.height < 1e-12 {
            return;
        }
        let target_ar = f64::from(target_width) / f64::from(target_height);
        let bbox_ar = self.width / self.height;

        if bbox_ar < target_ar {
            let new_width = self.height * target_ar;
            let pad = (new_width - self.width) * 0.5;
            self.min_x -= pad;
            self.max_x += pad;
            self.width = new_width;
        } else if bbox_ar > target_ar {
            let new_height = self.width / target_ar;
            let pad = (new_height - self.height) * 0.5;
            self.min_y -= pad;
            self.max_y += pad;
            self.height = new_height;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_positions(points: &[(f64, f64)]) -> Vec<Vec<Vector3<f64>>> {
        (0..3).map(|_| points.iter().map(|&(x, y)| Vector3::new(x, y, 0.0)).collect()).collect()
    }

    #[test]
    fn test_bounding_box_from_positions() {
        let positions = make_positions(&[(0.0, 0.0), (10.0, 5.0)]);
        let bbox = BoundingBox::from_positions(&positions);
        assert!(bbox.min_x < 0.0);
        assert!(bbox.max_x > 10.0);
        assert!(bbox.width > 10.0);
        assert!(bbox.height > 5.0);
    }

    #[test]
    fn test_world_to_pixel_corners() {
        let positions = make_positions(&[(0.0, 0.0), (100.0, 100.0)]);
        let bbox = BoundingBox::from_positions(&positions);
        let (px, py) = bbox.world_to_pixel(bbox.min_x, bbox.min_y, 1920, 1080);
        assert!(px.abs() < 1.0, "top-left should map near pixel (0,0)");
        assert!(py.abs() < 1.0, "top-left should map near pixel (0,0)");
    }

    #[test]
    fn test_normalize_center() {
        let positions = make_positions(&[(0.0, 0.0), (10.0, 10.0)]);
        let bbox = BoundingBox::from_positions(&positions);
        let cx = f64::midpoint(bbox.min_x, bbox.max_x);
        let cy = f64::midpoint(bbox.min_y, bbox.max_y);
        let (nx, ny) = bbox.normalize(cx, cy);
        assert!((nx - 0.5).abs() < 0.01, "center should normalize to ~0.5");
        assert!((ny - 0.5).abs() < 0.01, "center should normalize to ~0.5");
    }

    #[test]
    fn test_aspect_correction_wide_orbit() {
        let mut bbox = BoundingBox {
            min_x: 0.0,
            max_x: 100.0,
            min_y: 0.0,
            max_y: 50.0,
            width: 100.0,
            height: 50.0,
        };
        bbox.apply_aspect_correction(1920, 1080);
        let ar = bbox.width / bbox.height;
        let target_ar = 1920.0 / 1080.0;
        assert!(
            (ar - target_ar).abs() < 0.01,
            "corrected AR {ar:.3} should match target {target_ar:.3}"
        );
        assert!((bbox.width - 100.0).abs() < 0.01);
        assert!(bbox.height > 50.0);
    }

    #[test]
    fn test_aspect_correction_tall_orbit() {
        let mut bbox = BoundingBox {
            min_x: 0.0,
            max_x: 50.0,
            min_y: 0.0,
            max_y: 100.0,
            width: 50.0,
            height: 100.0,
        };
        bbox.apply_aspect_correction(1920, 1080);
        let ar = bbox.width / bbox.height;
        let target_ar = 1920.0 / 1080.0;
        assert!(
            (ar - target_ar).abs() < 0.01,
            "corrected AR {ar:.3} should match target {target_ar:.3}"
        );
        assert!((bbox.height - 100.0).abs() < 0.01);
        assert!(bbox.width > 50.0);
    }

    #[test]
    fn test_aspect_correction_already_matching() {
        let mut bbox = BoundingBox {
            min_x: 0.0,
            max_x: 160.0,
            min_y: 0.0,
            max_y: 90.0,
            width: 160.0,
            height: 90.0,
        };
        let orig_width = bbox.width;
        let orig_height = bbox.height;
        bbox.apply_aspect_correction(1920, 1080);
        assert!((bbox.width - orig_width).abs() < 0.01, "should not change matching bbox");
        assert!((bbox.height - orig_height).abs() < 0.01, "should not change matching bbox");
    }

    #[test]
    fn test_aspect_correction_centers_padding() {
        let mut bbox = BoundingBox {
            min_x: 10.0,
            max_x: 110.0,
            min_y: 20.0,
            max_y: 70.0,
            width: 100.0,
            height: 50.0,
        };
        let cy_before = f64::midpoint(bbox.min_y, bbox.max_y);
        bbox.apply_aspect_correction(1920, 1080);
        let cy_after = f64::midpoint(bbox.min_y, bbox.max_y);
        assert!((cy_before - cy_after).abs() < 0.01, "center Y should be preserved");
    }

    #[test]
    fn test_aspect_correction_zero_height() {
        let mut bbox = BoundingBox {
            min_x: 0.0,
            max_x: 100.0,
            min_y: 50.0,
            max_y: 50.0,
            width: 100.0,
            height: 0.0,
        };
        bbox.apply_aspect_correction(1920, 1080);
        assert!(bbox.width.is_finite());
        assert!(bbox.height.is_finite());
    }

    #[test]
    fn test_render_context_with_aspect_correction() {
        let positions = make_positions(&[(0.0, 0.0), (100.0, 50.0)]);
        let ctx = RenderContext::new(1920, 1080, &positions, true);
        let ar = ctx.bounds().width / ctx.bounds().height;
        let target_ar = 1920.0 / 1080.0;
        assert!(
            (ar - target_ar).abs() < 0.05,
            "aspect-corrected context AR {ar:.3} should be near {target_ar:.3}"
        );
    }

    #[test]
    fn test_render_context_without_aspect_correction() {
        let positions = make_positions(&[(0.0, 0.0), (100.0, 50.0)]);
        let ctx = RenderContext::new(1920, 1080, &positions, false);
        let ar = ctx.bounds().width / ctx.bounds().height;
        assert!(ar > 1.5, "uncorrected AR should reflect orbit shape");
    }

    #[test]
    fn test_pixel_count() {
        let positions = make_positions(&[(0.0, 0.0), (10.0, 10.0)]);
        let ctx = RenderContext::new(1920, 1080, &positions, false);
        assert_eq!(ctx.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_framing_zoom_identity_is_noop() {
        let mut bbox = BoundingBox {
            min_x: 0.0,
            max_x: 100.0,
            min_y: 0.0,
            max_y: 50.0,
            width: 100.0,
            height: 50.0,
        };
        bbox.apply_framing_zoom(1.0);
        assert!((bbox.width - 100.0).abs() < 1e-9);
        assert!((bbox.height - 50.0).abs() < 1e-9);
        assert!((bbox.min_x - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_framing_zoom_inflates_symmetrically() {
        let mut bbox = BoundingBox {
            min_x: 0.0,
            max_x: 100.0,
            min_y: 0.0,
            max_y: 50.0,
            width: 100.0,
            height: 50.0,
        };
        let center_x_before = f64::midpoint(bbox.min_x, bbox.max_x);
        let center_y_before = f64::midpoint(bbox.min_y, bbox.max_y);

        bbox.apply_framing_zoom(1.5);

        let center_x_after = f64::midpoint(bbox.min_x, bbox.max_x);
        let center_y_after = f64::midpoint(bbox.min_y, bbox.max_y);
        assert!((bbox.width - 150.0).abs() < 1e-6);
        assert!((bbox.height - 75.0).abs() < 1e-6);
        assert!((center_x_before - center_x_after).abs() < 1e-9);
        assert!((center_y_before - center_y_after).abs() < 1e-9);
    }

    #[test]
    fn test_framing_zoom_clamps_outside_range() {
        let mut bbox = BoundingBox {
            min_x: 0.0,
            max_x: 100.0,
            min_y: 0.0,
            max_y: 50.0,
            width: 100.0,
            height: 50.0,
        };

        bbox.apply_framing_zoom(0.5);
        assert!((bbox.width - 100.0).abs() < 1e-9);

        bbox.apply_framing_zoom(8.0);
        assert!((bbox.width - 200.0).abs() < 1e-6);
    }

    #[test]
    fn test_render_context_new_with_framing_adds_padding() {
        let positions = make_positions(&[(0.0, 0.0), (100.0, 100.0)]);
        let tight = RenderContext::new_with_framing(800, 800, &positions, false, 1.0);
        let zoomed = RenderContext::new_with_framing(800, 800, &positions, false, 1.25);
        assert!(zoomed.bounds().width > tight.bounds().width);
        assert!(zoomed.bounds().height > tight.bounds().height);
    }

    #[test]
    fn test_framing_shift_moves_center_without_resizing() {
        let positions = make_positions(&[(0.0, 0.0), (100.0, 100.0)]);
        let centered = RenderContext::new_with_composition(800, 800, &positions, false, 1.0, 0.0, 0.0);
        let shifted = RenderContext::new_with_composition(800, 800, &positions, false, 1.0, 0.12, -0.08);
        let centered_bounds = centered.bounds();
        let shifted_bounds = shifted.bounds();
        assert!((centered_bounds.width - shifted_bounds.width).abs() < 1e-9);
        assert!((centered_bounds.height - shifted_bounds.height).abs() < 1e-9);
        assert!(shifted_bounds.min_x > centered_bounds.min_x);
        assert!(shifted_bounds.min_y < centered_bounds.min_y);
    }
}
