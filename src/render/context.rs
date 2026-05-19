//! Common rendering context and utilities
//!
//! This module provides the core rendering context that manages coordinate transformations
//! and pixel operations, as well as utilities for iterating through animation frames.

use crate::render::error::{RenderError, Result};
use nalgebra::Vector3;

/// Type alias for pixel buffers used throughout the pipeline.
///
/// Format: `(R, G, B, A)` with premultiplied alpha.
/// Color channels may be linear or display-space depending on the render stage.
pub type PixelBuffer = Vec<(f64, f64, f64, f64)>;

fn validate_positions(positions: &[Vec<Vector3<f64>>]) -> Result<()> {
    let mut has_position = false;

    for (body_idx, body_positions) in positions.iter().enumerate() {
        for (step_idx, point) in body_positions.iter().enumerate() {
            for (axis, value) in [("x", point[0]), ("y", point[1]), ("z", point[2])] {
                if !value.is_finite() {
                    return Err(RenderError::InvalidScene {
                        reason: format!(
                            "position body {body_idx}, step {step_idx}, axis {axis} must be finite"
                        ),
                    });
                }
            }

            has_position = true;
        }
    }

    if !has_position {
        return Err(RenderError::InvalidScene {
            reason: "position data must contain at least one sample".to_string(),
        });
    }

    Ok(())
}

fn padded_position_bounds(positions: &[Vec<Vector3<f64>>]) -> Result<(f64, f64, f64, f64)> {
    validate_positions(positions)?;
    Ok(crate::utils::bounding_box(positions))
}

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

/// Art-directed framing applied after scene bounds and optional aspect correction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FramingConfig {
    /// Values > 1 zoom in; values < 1 create more negative space.
    pub zoom: f64,
    /// Horizontal offset as a fraction of the current bound width.
    pub offset_x: f64,
    /// Vertical offset as a fraction of the current bound height.
    pub offset_y: f64,
}

impl Default for FramingConfig {
    fn default() -> Self {
        Self { zoom: 1.0, offset_x: 0.0, offset_y: 0.0 }
    }
}

impl RenderContext {
    fn validate_dimensions(width: u32, height: u32) -> Result<(usize, usize, usize)> {
        if width == 0 || height == 0 {
            return Err(RenderError::InvalidDimensions { width, height });
        }

        let pixel_count = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|count| usize::try_from(count).ok())
            .ok_or(RenderError::InvalidDimensions { width, height })?;
        let width_usize =
            usize::try_from(width).map_err(|_| RenderError::InvalidDimensions { width, height })?;
        let height_usize = usize::try_from(height)
            .map_err(|_| RenderError::InvalidDimensions { width, height })?;

        Ok((width_usize, height_usize, pixel_count))
    }

    /// Try to create a new render context from position data.
    ///
    /// # Errors
    ///
    /// Returns an error when either dimension is zero or the pixel count cannot
    /// fit into the host address space, or when position data is empty or
    /// contains non-finite coordinates.
    pub fn try_new(
        width: u32,
        height: u32,
        positions: &[Vec<Vector3<f64>>],
        aspect_correction: bool,
    ) -> Result<Self> {
        let (width_usize, height_usize, pixel_count) = Self::validate_dimensions(width, height)?;
        let mut bounds = BoundingBox::try_from_positions(positions)?;
        if aspect_correction {
            bounds.apply_aspect_correction(width, height);
        }
        bounds.apply_framing(FramingConfig::default());

        let context = Self { width, height, width_usize, height_usize, bounds };
        debug_assert_eq!(context.width_usize * context.height_usize, pixel_count);
        Ok(context)
    }

    /// Try to create a render context with art-directed framing.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::try_new`].
    pub fn try_new_with_framing(
        width: u32,
        height: u32,
        positions: &[Vec<Vector3<f64>>],
        aspect_correction: bool,
        framing: FramingConfig,
    ) -> Result<Self> {
        let (width_usize, height_usize, pixel_count) = Self::validate_dimensions(width, height)?;
        let mut bounds = BoundingBox::try_from_positions(positions)?;
        if aspect_correction {
            bounds.apply_aspect_correction(width, height);
        }
        bounds.apply_framing(framing);

        let context = Self { width, height, width_usize, height_usize, bounds };
        debug_assert_eq!(context.width_usize * context.height_usize, pixel_count);
        Ok(context)
    }

    /// Creates a new render context from already-validated position data.
    ///
    /// Prefer [`Self::try_new`] when dimensions come from a caller or other
    /// untrusted boundary.
    ///
    /// # Panics
    ///
    /// Panics when either dimension is zero or the pixel count cannot fit into
    /// the host address space, or when position data is empty or contains
    /// non-finite coordinates.
    #[must_use]
    pub fn new(
        width: u32,
        height: u32,
        positions: &[Vec<Vector3<f64>>],
        aspect_correction: bool,
    ) -> Self {
        Self::try_new(width, height, positions, aspect_correction)
            .expect("render context inputs should be valid")
    }

    /// Convert world coordinates to pixel coordinates
    #[must_use]
    #[inline]
    pub fn to_pixel(&self, x: f64, y: f64) -> (f32, f32) {
        self.bounds.world_to_pixel(x, y, self.width, self.height)
    }

    /// Try to get the total pixel count.
    ///
    /// # Errors
    ///
    /// Returns an error if the public `width_usize` or `height_usize` fields
    /// have been mutated so their product no longer fits in `usize`.
    pub fn try_pixel_count(&self) -> Result<usize> {
        self.width_usize.checked_mul(self.height_usize).ok_or_else(|| RenderError::InvalidScene {
            reason: format!(
                "render context dimensions overflow usize: {}x{}",
                self.width_usize, self.height_usize
            ),
        })
    }

    /// Get total pixel count.
    ///
    /// Prefer [`Self::try_pixel_count`] when the context may have crossed a
    /// public API boundary.
    ///
    /// # Panics
    ///
    /// Panics if the public `width_usize` or `height_usize` fields have been
    /// mutated so their product no longer fits in `usize`.
    #[must_use]
    #[inline]
    pub fn pixel_count(&self) -> usize {
        self.try_pixel_count().expect("render context dimensions should stay valid")
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
    /// Try to create a new bounding box from position data.
    ///
    /// # Errors
    ///
    /// Returns an error when the position data contains no samples or includes
    /// non-finite coordinates.
    pub fn try_from_positions(positions: &[Vec<Vector3<f64>>]) -> Result<Self> {
        let (min_x, max_x, min_y, max_y) = padded_position_bounds(positions)?;
        Ok(Self {
            min_x,
            max_x,
            min_y,
            max_y,
            width: (max_x - min_x).max(1e-12),
            height: (max_y - min_y).max(1e-12),
        })
    }

    /// Create a new bounding box from already-validated position data.
    ///
    /// Prefer [`Self::try_from_positions`] when position data comes from a
    /// caller or other untrusted boundary.
    ///
    /// # Panics
    ///
    /// Panics when the position data contains no samples or includes non-finite
    /// coordinates.
    #[must_use]
    pub fn from_positions(positions: &[Vec<Vector3<f64>>]) -> Self {
        Self::try_from_positions(positions).expect("position data should contain finite samples")
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

    /// Apply a tasteful zoom/offset framing transform around the current bounds.
    pub fn apply_framing(&mut self, framing: FramingConfig) {
        let zoom = if framing.zoom.is_finite() { framing.zoom.clamp(0.65, 1.35) } else { 1.0 };
        let offset_x =
            if framing.offset_x.is_finite() { framing.offset_x.clamp(-0.18, 0.18) } else { 0.0 };
        let offset_y =
            if framing.offset_y.is_finite() { framing.offset_y.clamp(-0.18, 0.18) } else { 0.0 };

        let center_x = f64::midpoint(self.min_x, self.max_x) + self.width * offset_x;
        let center_y = f64::midpoint(self.min_y, self.max_y) + self.height * offset_y;
        let new_width = (self.width / zoom).max(1e-12);
        let new_height = (self.height / zoom).max(1e-12);

        self.min_x = center_x - new_width * 0.5;
        self.max_x = center_x + new_width * 0.5;
        self.min_y = center_y - new_height * 0.5;
        self.max_y = center_y + new_height * 0.5;
        self.width = new_width;
        self.height = new_height;
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
    fn test_bounding_box_try_from_positions_rejects_empty_positions() {
        let positions = vec![Vec::new(), Vec::new(), Vec::new()];
        let err = BoundingBox::try_from_positions(&positions)
            .expect_err("empty position data should fail");

        assert!(matches!(err, RenderError::InvalidScene { .. }));
        assert!(err.to_string().contains("at least one sample"));
    }

    #[test]
    fn test_bounding_box_try_from_positions_rejects_non_finite_positions() {
        let positions = vec![vec![Vector3::new(0.0, f64::NAN, 0.0)]];
        let err = BoundingBox::try_from_positions(&positions)
            .expect_err("non-finite position data should fail");

        assert!(matches!(err, RenderError::InvalidScene { .. }));
        assert!(err.to_string().contains("axis y must be finite"));
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
        let ctx = RenderContext::try_new(1920, 1080, &positions, true)
            .expect("valid dimensions should create render context");
        let ar = ctx.bounds().width / ctx.bounds().height;
        let target_ar = 1920.0 / 1080.0;
        assert!(
            (ar - target_ar).abs() < 0.05,
            "aspect-corrected context AR {ar:.3} should be near {target_ar:.3}"
        );
    }

    #[test]
    fn test_framing_zoom_in_tightens_bounds_around_center() {
        let positions = make_positions(&[(0.0, 0.0), (100.0, 100.0)]);
        let base = RenderContext::try_new(1000, 1000, &positions, false)
            .expect("valid context should build");
        let framed = RenderContext::try_new_with_framing(
            1000,
            1000,
            &positions,
            false,
            FramingConfig { zoom: 1.25, offset_x: 0.0, offset_y: 0.0 },
        )
        .expect("valid framed context should build");

        assert!(framed.bounds().width < base.bounds().width);
        assert!(framed.bounds().height < base.bounds().height);
        assert!(
            (f64::midpoint(framed.bounds().min_x, framed.bounds().max_x)
                - f64::midpoint(base.bounds().min_x, base.bounds().max_x))
            .abs()
                < 1e-9
        );
    }

    #[test]
    fn test_framing_negative_space_expands_bounds_and_offsets_center() {
        let positions = make_positions(&[(0.0, 0.0), (100.0, 100.0)]);
        let base = RenderContext::try_new(1000, 1000, &positions, false)
            .expect("valid context should build");
        let framed = RenderContext::try_new_with_framing(
            1000,
            1000,
            &positions,
            false,
            FramingConfig { zoom: 0.75, offset_x: 0.10, offset_y: -0.08 },
        )
        .expect("valid framed context should build");

        assert!(framed.bounds().width > base.bounds().width);
        assert!(framed.bounds().height > base.bounds().height);
        assert!(
            f64::midpoint(framed.bounds().min_x, framed.bounds().max_x)
                > f64::midpoint(base.bounds().min_x, base.bounds().max_x)
        );
        assert!(
            f64::midpoint(framed.bounds().min_y, framed.bounds().max_y)
                < f64::midpoint(base.bounds().min_y, base.bounds().max_y)
        );
    }

    #[test]
    fn test_render_context_try_new_rejects_zero_dimensions() {
        let positions = make_positions(&[(0.0, 0.0), (100.0, 50.0)]);
        let err =
            RenderContext::try_new(0, 1080, &positions, true).expect_err("zero width should fail");

        assert!(matches!(err, RenderError::InvalidDimensions { width: 0, height: 1080 }));
    }

    #[test]
    fn test_render_context_try_new_rejects_empty_positions() {
        let positions = vec![Vec::new(), Vec::new(), Vec::new()];
        let err = RenderContext::try_new(1920, 1080, &positions, true)
            .expect_err("empty positions should fail");

        assert!(matches!(err, RenderError::InvalidScene { .. }));
        assert!(err.to_string().contains("at least one sample"));
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
        assert_eq!(ctx.try_pixel_count().expect("pixel count should fit"), 1920 * 1080);
    }

    #[test]
    fn test_try_pixel_count_reports_mutated_overflow() {
        let positions = make_positions(&[(0.0, 0.0), (10.0, 10.0)]);
        let mut ctx = RenderContext::new(1920, 1080, &positions, false);
        ctx.width_usize = usize::MAX;
        ctx.height_usize = 2;

        let err = ctx.try_pixel_count().expect_err("overflow should fail");

        assert!(matches!(err, RenderError::InvalidScene { .. }));
        assert!(err.to_string().contains("render context dimensions overflow"));
    }
}
