//! Batched drawing operations for improved cache locality and performance
//!
//! This module provides optimized batch drawing functions that process multiple
//! line segments together, improving CPU cache utilization and instruction pipelining.

use super::color::OklabColor;
use super::drawing::{LineVertex, SpectralLineSegment, draw_line_segment_aa_spectral};
use crate::spectrum::NUM_BINS;
use nalgebra::Vector3;

/// Triangle vertex data for batch processing
pub type TriangleVertex = LineVertex;

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
    let [v0, v1, v2] = vertices;
    let [hdr_mult_01, hdr_mult_12, hdr_mult_20] = hdr_multipliers;

    draw_line_segment_aa_spectral(
        accum,
        width,
        height,
        SpectralLineSegment { start: v0, end: v1, hdr_scale: hdr_scale * hdr_mult_01 },
    );

    draw_line_segment_aa_spectral(
        accum,
        width,
        height,
        SpectralLineSegment { start: v1, end: v2, hdr_scale: hdr_scale * hdr_mult_12 },
    );

    draw_line_segment_aa_spectral(
        accum,
        width,
        height,
        SpectralLineSegment { start: v2, end: v0, hdr_scale: hdr_scale * hdr_mult_20 },
    );
}

/// Prepare triangle vertices from position data for batched drawing
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
}
