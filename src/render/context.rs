//! Common rendering context and utilities
//!
//! This module provides the core rendering context that manages coordinate transformations
//! and pixel operations, as well as utilities for iterating through animation frames.

use super::camera_path::PerspectiveCamera;
use super::framing_fit::fit_to_ink_camera;
use super::pipeline_flags;
use nalgebra::Vector3;

/// Shared thread-local store used to pass a pre-computed camera focal offset
/// to `RenderContext::new` without threading it through every call site.
///
/// Set by the CLI (or a test harness) before constructing a render context;
/// consumed once and cleared on read. A `None` offset means "no rule-of-thirds
/// shift".
static FOCAL_OFFSET_BITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FOCAL_OFFSET_X: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FOCAL_OFFSET_Y: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Install a rule-of-thirds focal offset for the next
/// [`RenderContext`] constructed with `with_framing`.
pub fn set_focal_offset(offset: Vector3<f64>) {
    FOCAL_OFFSET_X.store(offset.x.to_bits(), std::sync::atomic::Ordering::Relaxed);
    FOCAL_OFFSET_Y.store(offset.y.to_bits(), std::sync::atomic::Ordering::Relaxed);
    FOCAL_OFFSET_BITS.store(1, std::sync::atomic::Ordering::Relaxed);
}

/// Read the currently-installed rule-of-thirds focal offset, or zero if
/// none has been set. Does **not** clear the value — multiple render
/// contexts in a single run share the same composition offset.
#[must_use]
pub fn current_focal_offset() -> Vector3<f64> {
    if FOCAL_OFFSET_BITS.load(std::sync::atomic::Ordering::Relaxed) == 0 {
        return Vector3::zeros();
    }
    let x = f64::from_bits(FOCAL_OFFSET_X.load(std::sync::atomic::Ordering::Relaxed));
    let y = f64::from_bits(FOCAL_OFFSET_Y.load(std::sync::atomic::Ordering::Relaxed));
    Vector3::new(x, y, 0.0)
}

/// Clear the installed focal offset (test-only helper).
#[cfg(test)]
pub fn clear_focal_offset() {
    FOCAL_OFFSET_BITS.store(0, std::sync::atomic::Ordering::Relaxed);
    FOCAL_OFFSET_X.store(0, std::sync::atomic::Ordering::Relaxed);
    FOCAL_OFFSET_Y.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// Type alias for pixel buffers used throughout the pipeline.
///
/// Format: `(R, G, B, A)` with premultiplied alpha.
/// Color channels may be linear or display-space depending on the render stage.
pub type PixelBuffer = Vec<(f64, f64, f64, f64)>;

/// How the camera frames the trajectory.
#[derive(Clone, Copy, Debug)]
pub enum FramingMode {
    /// Tight fit to the bbox corners: percentile-trimmed axis-aligned bbox
    /// projected with `orbit_fit` so the bbox (not the actual ink) covers
    /// `fill` of the frame.
    AutoFill {
        /// Fraction of the frame the bbox should occupy (0.5..=0.98).
        fill: f64,
        /// Fraction of points to keep when trimming outliers (e.g. 0.99).
        pct: f64,
    },
    /// Tight fit to the projected ink: build a proxy projection, find the
    /// percentile-trimmed pixel rectangle containing the visible trail, and
    /// re-zoom/re-center the camera so **that** rectangle covers `fill` of
    /// the frame. This is the default — the axis-aligned bbox typically
    /// wastes most of the frame on empty corners for a thin 3-body curve.
    FitToInk {
        /// Fraction of the frame the ink envelope should occupy (0.5..=0.98).
        fill: f64,
        /// Fraction of projected ink samples kept when trimming outliers.
        ink_pct: f64,
    },
    /// Legacy behaviour: raw min/max bbox + fixed 55 FOV `orbit_default`.
    Classic,
}

impl Default for FramingMode {
    fn default() -> Self {
        Self::FitToInk { fill: 0.95, ink_pct: 0.98 }
    }
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
    /// Optional perspective camera; when `None`, `to_pixel` uses the bounding-box orthographic map.
    pub perspective: Option<PerspectiveCamera>,
}

impl RenderContext {
    /// Creates a new render context from position data using the globally-configured
    /// framing mode (see [`pipeline_flags::current_framing_mode`]).
    /// Prefer [`RenderContext::with_framing`] for explicit control.
    #[must_use]
    pub fn new(
        width: u32,
        height: u32,
        positions: &[Vec<Vector3<f64>>],
        aspect_correction: bool,
    ) -> Self {
        Self::with_framing(
            width,
            height,
            positions,
            aspect_correction,
            pipeline_flags::current_framing_mode(),
        )
    }

    /// Creates a new render context with an explicit framing mode.
    #[must_use]
    pub fn with_framing(
        width: u32,
        height: u32,
        positions: &[Vec<Vector3<f64>>],
        aspect_correction: bool,
        framing: FramingMode,
    ) -> Self {
        let mut bounds = match framing {
            FramingMode::AutoFill { pct, .. } => {
                BoundingBox::from_positions_percentile(positions, pct)
            }
            FramingMode::FitToInk { ink_pct, .. } => {
                BoundingBox::from_positions_percentile(positions, ink_pct)
            }
            FramingMode::Classic => BoundingBox::from_positions(positions),
        };
        if aspect_correction {
            bounds.apply_aspect_correction(width, height);
        }

        let focal_offset = current_focal_offset();

        let perspective = if pipeline_flags::perspective_camera_enabled() {
            let cam = match framing {
                FramingMode::AutoFill { fill, .. } => PerspectiveCamera::orbit_fit_with_offset(
                    &bounds,
                    width,
                    height,
                    fill,
                    focal_offset,
                ),
                FramingMode::FitToInk { fill, ink_pct } => fit_to_ink_camera(
                    positions,
                    &bounds,
                    width,
                    height,
                    fill,
                    ink_pct,
                    focal_offset,
                ),
                FramingMode::Classic => PerspectiveCamera::orbit_classic(&bounds, width, height),
            };
            Some(cam)
        } else {
            None
        };

        Self {
            width,
            height,
            width_usize: width as usize,
            height_usize: height as usize,
            bounds,
            perspective,
        }
    }

    /// Convert world coordinates to pixel coordinates
    #[must_use]
    #[inline]
    pub fn to_pixel(&self, x: f64, y: f64) -> (f32, f32) {
        self.bounds.world_to_pixel(x, y, self.width, self.height)
    }

    /// Project a full 3D world point to pixels; third return is circle-of-confusion radius in pixels.
    #[must_use]
    #[inline]
    pub fn to_pixel_world(&self, p: Vector3<f64>) -> (f32, f32, f32) {
        if let Some(cam) = &self.perspective {
            cam.project(self.width, self.height, p)
        } else {
            let (px, py) = self.bounds.world_to_pixel(p.x, p.y, self.width, self.height);
            let coc = (p.z as f32 * 0.05).abs();
            (px, py, coc)
        }
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

/// Compute min/max Z extent across all bodies and timesteps.
#[inline]
fn z_extent(positions: &[Vec<Vector3<f64>>]) -> (f64, f64) {
    let mut min_z = f64::INFINITY;
    let mut max_z = f64::NEG_INFINITY;
    for body in positions {
        for p in body {
            if p[2] < min_z {
                min_z = p[2];
            }
            if p[2] > max_z {
                max_z = p[2];
            }
        }
    }
    if !min_z.is_finite() || !max_z.is_finite() {
        return (-1.0, 1.0);
    }
    if (max_z - min_z).abs() < 1e-12 {
        return (
            min_z - crate::render::constants::BOUNDING_BOX_PADDING,
            max_z + crate::render::constants::BOUNDING_BOX_PADDING,
        );
    }
    (min_z, max_z)
}

/// Bounding box for coordinate transformations (now also carries Z extent for
/// perspective-camera fitting).
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
    /// Minimum z-coordinate of the bounding region.
    pub min_z: f64,
    /// Maximum z-coordinate of the bounding region.
    pub max_z: f64,
    /// Horizontal span (`max_x - min_x`), clamped to avoid division by zero.
    pub width: f64,
    /// Vertical span (`max_y - min_y`), clamped to avoid division by zero.
    pub height: f64,
    /// Depth span (`max_z - min_z`), clamped to avoid division by zero.
    pub depth: f64,
}

impl BoundingBox {
    /// Create a new bounding box from position data (raw min/max, legacy).
    #[must_use]
    pub fn from_positions(positions: &[Vec<Vector3<f64>>]) -> Self {
        let (min_x, max_x, min_y, max_y) = crate::utils::bounding_box(positions);
        let (min_z, max_z) = z_extent(positions);
        Self {
            min_x,
            max_x,
            min_y,
            max_y,
            min_z,
            max_z,
            width: (max_x - min_x).max(1e-12),
            height: (max_y - min_y).max(1e-12),
            depth: (max_z - min_z).max(1e-12),
        }
    }

    /// Create a percentile-trimmed bounding box so drift-inflated outlier
    /// timesteps do not back the camera off from the interesting action.
    #[must_use]
    pub fn from_positions_percentile(positions: &[Vec<Vector3<f64>>], pct: f64) -> Self {
        let (min_x, max_x, min_y, max_y, min_z, max_z) =
            crate::utils::bounding_box_percentile(positions, pct);
        Self {
            min_x,
            max_x,
            min_y,
            max_y,
            min_z,
            max_z,
            width: (max_x - min_x).max(1e-12),
            height: (max_y - min_y).max(1e-12),
            depth: (max_z - min_z).max(1e-12),
        }
    }

    /// The eight axis-aligned corners of the box.
    #[must_use]
    pub fn corners(&self) -> [Vector3<f64>; 8] {
        [
            Vector3::new(self.min_x, self.min_y, self.min_z),
            Vector3::new(self.max_x, self.min_y, self.min_z),
            Vector3::new(self.min_x, self.max_y, self.min_z),
            Vector3::new(self.max_x, self.max_y, self.min_z),
            Vector3::new(self.min_x, self.min_y, self.max_z),
            Vector3::new(self.max_x, self.min_y, self.max_z),
            Vector3::new(self.min_x, self.max_y, self.max_z),
            Vector3::new(self.max_x, self.max_y, self.max_z),
        ]
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

    fn test_bbox(min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> BoundingBox {
        BoundingBox {
            min_x,
            max_x,
            min_y,
            max_y,
            min_z: -1.0,
            max_z: 1.0,
            width: (max_x - min_x).max(1e-12),
            height: (max_y - min_y).max(1e-12),
            depth: 2.0,
        }
    }

    #[test]
    fn test_aspect_correction_wide_orbit() {
        let mut bbox = test_bbox(0.0, 100.0, 0.0, 50.0);
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
        let mut bbox = test_bbox(0.0, 50.0, 0.0, 100.0);
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
        let mut bbox = test_bbox(0.0, 160.0, 0.0, 90.0);
        let orig_width = bbox.width;
        let orig_height = bbox.height;
        bbox.apply_aspect_correction(1920, 1080);
        assert!((bbox.width - orig_width).abs() < 0.01, "should not change matching bbox");
        assert!((bbox.height - orig_height).abs() < 0.01, "should not change matching bbox");
    }

    #[test]
    fn test_aspect_correction_centers_padding() {
        let mut bbox = test_bbox(10.0, 110.0, 20.0, 70.0);
        let cy_before = f64::midpoint(bbox.min_y, bbox.max_y);
        bbox.apply_aspect_correction(1920, 1080);
        let cy_after = f64::midpoint(bbox.min_y, bbox.max_y);
        assert!((cy_before - cy_after).abs() < 0.01, "center Y should be preserved");
    }

    #[test]
    fn test_aspect_correction_zero_height() {
        let mut bbox = test_bbox(0.0, 100.0, 50.0, 50.0);
        bbox.height = 0.0;
        bbox.apply_aspect_correction(1920, 1080);
        assert!(bbox.width.is_finite());
        assert!(bbox.height.is_finite());
    }

    #[test]
    fn test_bounding_box_carries_z_extent() {
        let positions = vec![vec![Vector3::new(0.0, 0.0, -3.0), Vector3::new(0.0, 0.0, 4.0)]];
        let bbox = BoundingBox::from_positions(&positions);
        assert!(bbox.min_z <= -3.0, "min_z should include the negative extreme");
        assert!(bbox.max_z >= 4.0, "max_z should include the positive extreme");
        assert!(bbox.depth >= 7.0, "depth should span the z-extent");
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
}
