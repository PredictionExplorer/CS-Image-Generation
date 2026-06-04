//! Batched drawing operations for improved cache locality and performance
//!
//! This module provides optimized batch drawing functions that process multiple
//! line segments together, improving CPU cache utilization and instruction pipelining.

use super::color::OklabColor;
use super::constants::CRISP_TRIANGLE_EDGE_STRENGTH;
use super::drawing::{
    LineVertex, SpectralLineSegment, draw_line_segment_aa_spectral_rows,
    draw_triangle_fill_spectral_rows,
};
use crate::spectrum::NUM_BINS;
use nalgebra::Vector3;

/// Triangle vertex data for batch processing
pub type TriangleVertex = LineVertex;

pub(crate) struct BatchDrawParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) row_start: usize,
    pub(crate) row_end: usize,
    pub(crate) vertices: [TriangleVertex; 3],
    pub(crate) hdr_multipliers: [f64; 3],
    pub(crate) hdr_scale: f64,
}

/// Draw a complete triangle (3 line segments) in a batch for better performance
#[inline]
pub fn draw_triangle_batch_spectral(
    accum: &mut [[f64; NUM_BINS]],
    width: u32,
    height: u32,
    vertices: [TriangleVertex; 3],
    hdr_multipliers: [f64; 3],
    hdr_scale: f64,
) {
    draw_triangle_batch_spectral_rows(
        accum,
        &BatchDrawParams {
            width,
            height,
            row_start: 0,
            row_end: height as usize,
            vertices,
            hdr_multipliers,
            hdr_scale,
        },
    );
}

/// Draw a complete triangle into an owned row band of the destination buffer.
pub(crate) fn draw_triangle_batch_spectral_rows(
    accum: &mut [[f64; NUM_BINS]],
    params: &BatchDrawParams,
) {
    let [v0, v1, v2] = params.vertices;
    let [hdr_mult_01, hdr_mult_12, hdr_mult_20] = params.hdr_multipliers;

    draw_triangle_fill_spectral_rows(
        accum,
        params.width,
        params.height,
        params.row_start,
        params.row_end,
        params.vertices,
        params.hdr_scale,
    );

    draw_line_segment_aa_spectral_rows(
        accum,
        params.width,
        params.height,
        params.row_start,
        params.row_end,
        SpectralLineSegment {
            start: v0,
            end: v1,
            hdr_scale: params.hdr_scale * CRISP_TRIANGLE_EDGE_STRENGTH * hdr_mult_01,
        },
    );

    draw_line_segment_aa_spectral_rows(
        accum,
        params.width,
        params.height,
        params.row_start,
        params.row_end,
        SpectralLineSegment {
            start: v1,
            end: v2,
            hdr_scale: params.hdr_scale * CRISP_TRIANGLE_EDGE_STRENGTH * hdr_mult_12,
        },
    );

    draw_line_segment_aa_spectral_rows(
        accum,
        params.width,
        params.height,
        params.row_start,
        params.row_end,
        SpectralLineSegment {
            start: v2,
            end: v0,
            hdr_scale: params.hdr_scale * CRISP_TRIANGLE_EDGE_STRENGTH * hdr_mult_20,
        },
    );
}

/// Return the maximum 2D pixel motion between two triangle samples.
#[must_use]
#[inline]
pub(crate) fn max_triangle_vertex_motion_px(
    start: [TriangleVertex; 3],
    end: [TriangleVertex; 3],
) -> f32 {
    start
        .iter()
        .zip(end.iter())
        .map(|(a, b)| {
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            (dx * dx + dy * dy).sqrt()
        })
        .fold(0.0, f32::max)
}

/// Linearly interpolate a triangle sample in pixel/OkLab space.
#[must_use]
#[inline]
pub(crate) fn interpolate_triangle_vertices(
    start: [TriangleVertex; 3],
    end: [TriangleVertex; 3],
    t: f32,
) -> [TriangleVertex; 3] {
    std::array::from_fn(|idx| interpolate_vertex(start[idx], end[idx], t))
}

#[inline]
fn interpolate_vertex(start: TriangleVertex, end: TriangleVertex, t: f32) -> TriangleVertex {
    let t64 = f64::from(t);
    let inv_t64 = 1.0 - t64;
    TriangleVertex {
        x: start.x + (end.x - start.x) * t,
        y: start.y + (end.y - start.y) * t,
        z: start.z + (end.z - start.z) * t,
        color: (
            start.color.0 * inv_t64 + end.color.0 * t64,
            start.color.1 * inv_t64 + end.color.1 * t64,
            start.color.2 * inv_t64 + end.color.2 * t64,
        ),
        alpha: start.alpha * inv_t64 + end.alpha * t64,
    }
}

/// Prepare triangle vertices from position data for batched drawing
#[must_use]
#[inline]
pub fn prepare_triangle_vertices(
    positions: &[Vec<Vector3<f64>>],
    colors: &[Vec<OklabColor>],
    body_alphas: &[f64; 3],
    step: usize,
    ctx: &super::context::RenderContext,
) -> [TriangleVertex; 3] {
    let p0 = positions[0][step];
    let p1 = positions[1][step];
    let p2 = positions[2][step];

    let (x0, y0) = ctx.to_pixel(p0[0], p0[1]);
    let (x1, y1) = ctx.to_pixel(p1[0], p1[1]);
    let (x2, y2) = ctx.to_pixel(p2[0], p2[1]);

    [
        TriangleVertex {
            x: x0,
            y: y0,
            z: p0[2] as f32,
            color: colors[0][step],
            alpha: body_alphas[0],
        },
        TriangleVertex {
            x: x1,
            y: y1,
            z: p1[2] as f32,
            color: colors[1][step],
            alpha: body_alphas[1],
        },
        TriangleVertex {
            x: x2,
            y: y2,
            z: p2[2] as f32,
            color: colors[2][step],
            alpha: body_alphas[2],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_vertex_creation() {
        let vertex =
            TriangleVertex { x: 100.0, y: 200.0, z: 0.0, color: (0.5, 0.1, 0.1), alpha: 0.5 };

        assert_eq!(vertex.x, 100.0);
        assert_eq!(vertex.y, 200.0);
        assert_eq!(vertex.alpha, 0.5);
    }

    #[test]
    fn test_prepare_triangle_vertices() {
        use crate::render::context::RenderContext;
        use nalgebra::Vector3;

        let positions = vec![
            vec![Vector3::new(0.0, 0.0, 0.0)],
            vec![Vector3::new(10.0, 0.0, 0.0)],
            vec![Vector3::new(5.0, 10.0, 0.0)],
        ];

        let colors = vec![vec![(0.5, 0.1, 0.1)], vec![(0.5, -0.1, 0.0)], vec![(0.5, 0.0, -0.1)]];

        let body_alphas = [0.5, 0.6, 0.7];

        let ctx = RenderContext::new(1920, 1080, &positions, false);
        let vertices = prepare_triangle_vertices(&positions, &colors, &body_alphas, 0, &ctx);

        assert_eq!(vertices.len(), 3);
        assert_eq!(vertices[0].alpha, 0.5);
        assert_eq!(vertices[1].alpha, 0.6);
        assert_eq!(vertices[2].alpha, 0.7);
    }

    #[test]
    fn test_triangle_interpolation_blends_positions_colors_and_alpha() {
        let start = [
            TriangleVertex { x: 0.0, y: 2.0, z: 4.0, color: (0.4, 0.1, -0.2), alpha: 0.2 },
            TriangleVertex { x: 10.0, y: 12.0, z: 14.0, color: (0.5, 0.2, -0.1), alpha: 0.4 },
            TriangleVertex { x: 20.0, y: 22.0, z: 24.0, color: (0.6, 0.3, 0.0), alpha: 0.6 },
        ];
        let end = [
            TriangleVertex { x: 2.0, y: 4.0, z: 6.0, color: (0.8, -0.1, 0.2), alpha: 0.6 },
            TriangleVertex { x: 12.0, y: 14.0, z: 16.0, color: (0.9, -0.2, 0.1), alpha: 0.8 },
            TriangleVertex { x: 22.0, y: 24.0, z: 26.0, color: (1.0, -0.3, 0.0), alpha: 1.0 },
        ];

        let blended = interpolate_triangle_vertices(start, end, 0.25);

        assert_eq!(blended[0].x, 0.5);
        assert_eq!(blended[0].y, 2.5);
        assert_eq!(blended[0].z, 4.5);
        assert!((blended[0].color.0 - 0.5).abs() < 1e-12);
        assert!((blended[0].color.1 - 0.05).abs() < 1e-12);
        assert!((blended[0].color.2 - -0.1).abs() < 1e-12);
        assert!((blended[0].alpha - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_triangle_motion_reports_max_screen_delta() {
        let start = [
            TriangleVertex { x: 0.0, y: 0.0, z: 0.0, color: (0.0, 0.0, 0.0), alpha: 1.0 },
            TriangleVertex { x: 10.0, y: 0.0, z: 0.0, color: (0.0, 0.0, 0.0), alpha: 1.0 },
            TriangleVertex { x: 0.0, y: 10.0, z: 0.0, color: (0.0, 0.0, 0.0), alpha: 1.0 },
        ];
        let end = [
            TriangleVertex { x: 3.0, y: 4.0, z: 0.0, color: (0.0, 0.0, 0.0), alpha: 1.0 },
            TriangleVertex { x: 11.0, y: 0.0, z: 0.0, color: (0.0, 0.0, 0.0), alpha: 1.0 },
            TriangleVertex { x: 0.0, y: 12.0, z: 0.0, color: (0.0, 0.0, 0.0), alpha: 1.0 },
        ];

        assert_eq!(max_triangle_vertex_motion_px(start, end), 5.0);
    }
}
