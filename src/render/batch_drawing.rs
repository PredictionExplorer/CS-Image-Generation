//! Batched drawing operations for improved cache locality and performance
//!
//! This module provides optimized batch drawing functions that process multiple
//! line segments together, improving CPU cache utilization and instruction pipelining.

use super::color::OklabColor;
use super::drawing::{LineVertex, SpectralLineSegment, draw_line_segment_aa_spectral_rows};
use crate::kinematics::KinematicTrajectories;
use crate::render::pipeline_flags;
use crate::spectrum::NUM_BINS;
use nalgebra::Vector3;

/// Triangle vertex data for batch processing
pub type TriangleVertex = LineVertex;

pub(crate) struct BatchDrawParams<'a> {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) row_start: usize,
    pub(crate) row_end: usize,
    pub(crate) vertices: [TriangleVertex; 3],
    pub(crate) hdr_multipliers: [f64; 3],
    pub(crate) hdr_scale: f64,
    pub(crate) kinematics: Option<&'a KinematicTrajectories>,
    pub(crate) step: usize,
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
            kinematics: None,
            step: 0,
        },
    );
}

#[inline]
fn doppler_mul(kin: Option<&KinematicTrajectories>, step: usize, a: usize, b: usize) -> f64 {
    let c = pipeline_flags::doppler_c_art();
    if !c.is_finite() || c <= 0.0 || c > 1.0e12 {
        return 1.0;
    }
    let Some(k) = kin else {
        return 1.0;
    };
    if step >= k.velocity[0].len() {
        return 1.0;
    }
    let v = (k.velocity[a][step].norm() + k.velocity[b][step].norm()) * 0.5;
    (1.0 + v / c).clamp(0.2, 5.0)
}

/// Draw a complete triangle into an owned row band of the destination buffer.
pub(crate) fn draw_triangle_batch_spectral_rows(
    accum: &mut [[f64; NUM_BINS]],
    params: &BatchDrawParams<'_>,
) {
    let [v0, v1, v2] = params.vertices;
    let [hdr_mult_01, hdr_mult_12, hdr_mult_20] = params.hdr_multipliers;
    let st = params.step;
    let kin = params.kinematics;
    let dm01 = doppler_mul(kin, st, 0, 1);
    let dm12 = doppler_mul(kin, st, 1, 2);
    let dm20 = doppler_mul(kin, st, 2, 0);

    draw_line_segment_aa_spectral_rows(
        accum,
        params.width,
        params.height,
        params.row_start,
        params.row_end,
        SpectralLineSegment {
            start: v0,
            end: v1,
            hdr_scale: params.hdr_scale * hdr_mult_01,
            doppler_mul: dm01,
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
            hdr_scale: params.hdr_scale * hdr_mult_12,
            doppler_mul: dm12,
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
            hdr_scale: params.hdr_scale * hdr_mult_20,
            doppler_mul: dm20,
        },
    );
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
    prepare_triangle_vertices_at([p0, p1, p2], colors, body_alphas, step, ctx)
}

/// Like [`prepare_triangle_vertices`] but with explicit corner positions (for motion blur substeps).
#[must_use]
pub fn prepare_triangle_vertices_at(
    corners: [Vector3<f64>; 3],
    colors: &[Vec<OklabColor>],
    body_alphas: &[f64; 3],
    step: usize,
    ctx: &super::context::RenderContext,
) -> [TriangleVertex; 3] {
    let p0 = corners[0];
    let p1 = corners[1];
    let p2 = corners[2];

    let (x0, y0, _) = ctx.to_pixel_world(p0);
    let (x1, y1, _) = ctx.to_pixel_world(p1);
    let (x2, y2, _) = ctx.to_pixel_world(p2);

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

/// Linearly interpolate `pos[step]` toward `pos[step+1]` by `frac ∈ [0,1]`.
#[inline]
pub(crate) fn lerp_body_step(pos: &[Vector3<f64>], step: usize, frac: f64) -> Vector3<f64> {
    if step + 1 < pos.len() {
        pos[step] * (1.0 - frac) + pos[step + 1] * frac
    } else {
        pos[step]
    }
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
