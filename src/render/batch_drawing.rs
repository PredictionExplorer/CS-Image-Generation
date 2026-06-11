//! Batched drawing operations for improved cache locality and performance
//!
//! This module provides optimized batch drawing functions that process multiple
//! line segments together, improving CPU cache utilization and instruction
//! pipelining, plus the stroke-level symmetry replication (mirror, k-fold
//! rotational, dihedral) used by kaleidoscopic seeds.

use super::color::OklabColor;
use super::drawing::{LineVertex, SpectralLineSegment, draw_line_segment_aa_spectral_rows};
use super::velocity_hdr::SegmentDynamics;
use super::visual_profile::SymmetryOp;
use crate::spectrum::NUM_BINS;
use nalgebra::Vector3;
use smallvec::SmallVec;

/// Triangle vertex data for batch processing
pub type TriangleVertex = LineVertex;

pub(crate) struct BatchDrawParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) row_start: usize,
    pub(crate) row_end: usize,
    pub(crate) vertices: [TriangleVertex; 3],
    pub(crate) edge_dynamics: [SegmentDynamics; 3],
    /// Per-edge alpha weight; 0 skips the edge entirely (duet modes).
    pub(crate) edge_weights: [f64; 3],
    /// Seed-level global line weight multiplier.
    pub(crate) line_weight: f64,
    pub(crate) hdr_scale: f64,
    /// Stroke-level symmetry replication (mirror / rotational / dihedral).
    pub(crate) symmetry: SymmetryOp,
}

/// One rigid 2D symmetry copy: rotate by (cos, sin) about the frame center,
/// optionally mirroring x first, with a uniform scale that keeps rotated
/// content inside the frame.
#[derive(Clone, Copy, Debug)]
struct SymmetryTransform {
    cos: f32,
    sin: f32,
    scale: f32,
    mirror_x: bool,
}

impl SymmetryTransform {
    const IDENTITY: Self = Self { cos: 1.0, sin: 0.0, scale: 1.0, mirror_x: false };

    #[inline]
    fn apply_point(self, x: f32, y: f32, cx: f32, cy: f32) -> (f32, f32) {
        let mut dx = x - cx;
        let dy = y - cy;
        if self.mirror_x {
            dx = -dx;
        }
        let rx = (dx * self.cos - dy * self.sin) * self.scale;
        let ry = (dx * self.sin + dy * self.cos) * self.scale;
        (cx + rx, cy + ry)
    }

    #[inline]
    fn apply_segment(self, segment: SpectralLineSegment, cx: f32, cy: f32) -> SpectralLineSegment {
        let mut out = segment;
        let (sx, sy) = self.apply_point(segment.start.x, segment.start.y, cx, cy);
        let (ex, ey) = self.apply_point(segment.end.x, segment.end.y, cx, cy);
        out.start.x = sx;
        out.start.y = sy;
        out.end.x = ex;
        out.end.y = ey;
        out
    }
}

/// Uniform shrink that keeps a full-frame composition inside the frame under
/// arbitrary rotation: the frame diagonal must fit inside the short edge.
#[inline]
fn rotational_fit_scale(width: u32, height: u32) -> f32 {
    let w = width as f32;
    let h = height as f32;
    let diagonal = (w * w + h * h).sqrt();
    if diagonal <= 0.0 { 1.0 } else { (w.min(h) / diagonal).min(1.0) }
}

/// Expand a symmetry op into its concrete per-copy transforms for this frame.
///
/// Total deposited energy is preserved by callers dividing per-copy energy by
/// `transforms.len()`, so exposure stays stable across symmetry classes.
fn symmetry_transforms(
    symmetry: SymmetryOp,
    width: u32,
    height: u32,
) -> SmallVec<[SymmetryTransform; 12]> {
    let mut transforms: SmallVec<[SymmetryTransform; 12]> = SmallVec::new();
    match symmetry {
        SymmetryOp::None => transforms.push(SymmetryTransform::IDENTITY),
        SymmetryOp::MirrorX => {
            transforms.push(SymmetryTransform::IDENTITY);
            transforms.push(SymmetryTransform { mirror_x: true, ..SymmetryTransform::IDENTITY });
        }
        SymmetryOp::Rotational { k } => {
            let k = usize::from(k.max(1));
            let scale = rotational_fit_scale(width, height);
            for i in 0..k {
                let theta = std::f64::consts::TAU * i as f64 / k as f64;
                transforms.push(SymmetryTransform {
                    cos: theta.cos() as f32,
                    sin: theta.sin() as f32,
                    scale,
                    mirror_x: false,
                });
            }
        }
        SymmetryOp::Dihedral { k } => {
            let k = usize::from(k.max(1));
            let scale = rotational_fit_scale(width, height);
            for mirror_x in [false, true] {
                for i in 0..k {
                    let theta = std::f64::consts::TAU * i as f64 / k as f64;
                    transforms.push(SymmetryTransform {
                        cos: theta.cos() as f32,
                        sin: theta.sin() as f32,
                        scale,
                        mirror_x,
                    });
                }
            }
        }
    }
    transforms
}

/// Draw a segment once per symmetry copy, dividing energy by the fold count.
#[inline]
pub(crate) fn draw_segment_rows_symmetric(
    accum: &mut [[f64; NUM_BINS]],
    width: u32,
    height: u32,
    row_start: usize,
    row_end: usize,
    segment: SpectralLineSegment,
    symmetry: SymmetryOp,
) {
    if matches!(symmetry, SymmetryOp::None) {
        draw_line_segment_aa_spectral_rows(accum, width, height, row_start, row_end, segment);
        return;
    }

    let transforms = symmetry_transforms(symmetry, width, height);
    // usize→f64: fold counts are at most 12.
    let energy_share = 1.0 / transforms.len() as f64;
    // u32→f32 precision loss is irrelevant at raster scale.
    let cx = width as f32 * 0.5;
    let cy = height as f32 * 0.5;
    for transform in &transforms {
        let mut copy = transform.apply_segment(segment, cx, cy);
        copy.hdr_scale = segment.hdr_scale * energy_share;
        draw_line_segment_aa_spectral_rows(accum, width, height, row_start, row_end, copy);
    }
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
            edge_dynamics: hdr_multipliers
                .map(|m| SegmentDynamics { hdr_multiplier: m, thickness_factor: 1.0 }),
            edge_weights: [1.0; 3],
            line_weight: 1.0,
            hdr_scale,
            symmetry: SymmetryOp::None,
        },
    );
}

/// Draw a complete triangle into an owned row band of the destination buffer.
pub(crate) fn draw_triangle_batch_spectral_rows(
    accum: &mut [[f64; NUM_BINS]],
    params: &BatchDrawParams,
) {
    let [v0, v1, v2] = params.vertices;
    let edges = [(v0, v1), (v1, v2), (v2, v0)];

    for (edge_idx, (start, end)) in edges.into_iter().enumerate() {
        let weight = params.edge_weights[edge_idx];
        if weight <= 0.0 {
            continue;
        }
        let dynamics = params.edge_dynamics[edge_idx];
        draw_segment_rows_symmetric(
            accum,
            params.width,
            params.height,
            params.row_start,
            params.row_end,
            SpectralLineSegment {
                start,
                end,
                hdr_scale: params.hdr_scale * dynamics.hdr_multiplier * weight,
                thickness_factor: dynamics.thickness_factor * params.line_weight,
            },
            params.symmetry,
        );
    }
}

/// Draw one per-body trail stroke from the current vertex to `target_vertices`,
/// with the deposited energy scaled by `energy_scale`.
///
/// Used for ribbon strokes (`step -> step + 1`, scale 1) and for time-lagged
/// chord/echo strokes (`step -> step + lag`, reduced scale).
#[inline]
pub(crate) fn draw_body_trail_segment_rows(
    accum: &mut [[f64; NUM_BINS]],
    params: &BatchDrawParams,
    body: usize,
    target_vertices: [TriangleVertex; 3],
    energy_scale: f64,
) {
    let dynamics = params.edge_dynamics[body];
    draw_segment_rows_symmetric(
        accum,
        params.width,
        params.height,
        params.row_start,
        params.row_end,
        SpectralLineSegment {
            start: params.vertices[body],
            end: target_vertices[body],
            hdr_scale: params.hdr_scale * dynamics.hdr_multiplier * energy_scale,
            thickness_factor: dynamics.thickness_factor * params.line_weight,
        },
        params.symmetry,
    );
}

/// Draw the three body-to-centroid spokes for the current step.
#[inline]
pub(crate) fn draw_spoke_segments_rows(accum: &mut [[f64; NUM_BINS]], params: &BatchDrawParams) {
    let [v0, v1, v2] = params.vertices;
    let centroid_x = (v0.x + v1.x + v2.x) / 3.0;
    let centroid_y = (v0.y + v1.y + v2.y) / 3.0;
    let centroid_z = (v0.z + v1.z + v2.z) / 3.0;

    for body in 0..3 {
        let vertex = params.vertices[body];
        let centroid = TriangleVertex {
            x: centroid_x,
            y: centroid_y,
            z: centroid_z,
            color: vertex.color,
            alpha: vertex.alpha,
        };
        let dynamics = params.edge_dynamics[body];
        draw_segment_rows_symmetric(
            accum,
            params.width,
            params.height,
            params.row_start,
            params.row_end,
            SpectralLineSegment {
                start: vertex,
                end: centroid,
                hdr_scale: params.hdr_scale * dynamics.hdr_multiplier,
                thickness_factor: dynamics.thickness_factor * params.line_weight,
            },
            params.symmetry,
        );
    }
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
pub(crate) fn interpolate_vertex(
    start: TriangleVertex,
    end: TriangleVertex,
    t: f32,
) -> TriangleVertex {
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

    fn buffer_energy(buf: &[[f64; NUM_BINS]]) -> f64 {
        buf.iter().flat_map(|bins| bins.iter()).sum()
    }

    fn test_segment() -> SpectralLineSegment {
        SpectralLineSegment {
            start: TriangleVertex { x: 20.0, y: 32.0, z: 0.0, color: (0.7, 0.2, 0.0), alpha: 1.0 },
            end: TriangleVertex { x: 44.0, y: 36.0, z: 0.0, color: (0.7, -0.1, 0.2), alpha: 1.0 },
            hdr_scale: 1.0,
            thickness_factor: 1.0,
        }
    }

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

    #[test]
    fn test_triangle_batch_does_not_fill_interior_by_default() {
        let width = 48usize;
        let height = 48usize;
        let mut accum = vec![[0.0; NUM_BINS]; width * height];
        let vertices = [
            TriangleVertex { x: 4.0, y: 4.0, z: 0.0, color: (0.7, 0.2, 0.0), alpha: 1.0 },
            TriangleVertex { x: 44.0, y: 4.0, z: 0.0, color: (0.7, -0.1, 0.2), alpha: 1.0 },
            TriangleVertex { x: 4.0, y: 44.0, z: 0.0, color: (0.7, 0.0, -0.2), alpha: 1.0 },
        ];

        draw_triangle_batch_spectral(
            &mut accum,
            width as u32,
            height as u32,
            vertices,
            [1.0, 1.0, 1.0],
            1.0,
        );

        let interior_energy: f64 = accum[16 * width + 16].iter().sum();
        assert_eq!(interior_energy, 0.0, "default triangle batch must remain edge-only");
    }

    #[test]
    fn symmetry_transform_counts_match_fold_counts() {
        for symmetry in [
            SymmetryOp::None,
            SymmetryOp::MirrorX,
            SymmetryOp::Rotational { k: 2 },
            SymmetryOp::Rotational { k: 5 },
            SymmetryOp::Dihedral { k: 3 },
            SymmetryOp::Dihedral { k: 6 },
        ] {
            let transforms = symmetry_transforms(symmetry, 640, 360);
            assert_eq!(
                transforms.len(),
                symmetry.fold_count(),
                "transform count must equal fold count for {symmetry:?}"
            );
        }
    }

    #[test]
    fn symmetric_draw_approximately_preserves_total_energy() {
        let width = 64usize;
        let height = 64usize;
        let segment = test_segment();

        let mut reference = vec![[0.0; NUM_BINS]; width * height];
        draw_segment_rows_symmetric(
            &mut reference,
            width as u32,
            height as u32,
            0,
            height,
            segment,
            SymmetryOp::None,
        );
        let reference_energy = buffer_energy(&reference);
        assert!(reference_energy > 0.0);

        for symmetry in [
            SymmetryOp::MirrorX,
            SymmetryOp::Rotational { k: 3 },
            SymmetryOp::Rotational { k: 6 },
            SymmetryOp::Dihedral { k: 4 },
        ] {
            let mut accum = vec![[0.0; NUM_BINS]; width * height];
            draw_segment_rows_symmetric(
                &mut accum,
                width as u32,
                height as u32,
                0,
                height,
                segment,
                symmetry,
            );
            let energy = buffer_energy(&accum);
            // Rotational copies are scaled to fit the frame, which shortens
            // segments and concentrates ink; allow a generous band rather than
            // exact equality.
            assert!(
                energy > reference_energy * 0.4 && energy < reference_energy * 2.5,
                "energy drifted too far under {symmetry:?}: {energy} vs {reference_energy}"
            );
        }
    }

    #[test]
    fn rotational_copies_land_at_rotated_positions() {
        let width = 100u32;
        let height = 100u32;
        // A dot just right of center: under Rotational k=4 copies must appear
        // above, left, and below center too.
        let dot = SpectralLineSegment {
            start: TriangleVertex { x: 70.0, y: 50.0, z: 0.0, color: (0.7, 0.2, 0.0), alpha: 1.0 },
            end: TriangleVertex { x: 70.0, y: 50.0, z: 0.0, color: (0.7, 0.2, 0.0), alpha: 1.0 },
            hdr_scale: 1.0,
            thickness_factor: 1.0,
        };
        let mut accum = vec![[0.0; NUM_BINS]; 100 * 100];
        draw_segment_rows_symmetric(
            &mut accum,
            width,
            height,
            0,
            100,
            dot,
            SymmetryOp::Rotational { k: 4 },
        );

        let scale = rotational_fit_scale(width, height);
        let offset = 20.0 * scale; // dot is 20px right of center
        let energy_at = |x: f64, y: f64| -> f64 {
            let xi = x.round() as usize;
            let yi = y.round() as usize;
            let mut total = 0.0;
            for dy in -2i64..=2 {
                for dx in -2i64..=2 {
                    let px = (xi as i64 + dx).clamp(0, 99) as usize;
                    let py = (yi as i64 + dy).clamp(0, 99) as usize;
                    total += accum[py * 100 + px].iter().sum::<f64>();
                }
            }
            total
        };

        assert!(energy_at(50.0 + f64::from(offset), 50.0) > 0.0, "0 deg copy missing");
        assert!(energy_at(50.0, 50.0 + f64::from(offset)) > 0.0, "90 deg copy missing");
        assert!(energy_at(50.0 - f64::from(offset), 50.0) > 0.0, "180 deg copy missing");
        assert!(energy_at(50.0, 50.0 - f64::from(offset)) > 0.0, "270 deg copy missing");
    }

    #[test]
    fn mirror_matches_legacy_vertical_axis_reflection() {
        let width = 64usize;
        let height = 32usize;
        let segment = test_segment();
        let mut accum = vec![[0.0; NUM_BINS]; width * height];
        draw_segment_rows_symmetric(
            &mut accum,
            width as u32,
            height as u32,
            0,
            height,
            segment,
            SymmetryOp::MirrorX,
        );

        // Energy must appear on both sides of the vertical center line.
        let left: f64 = (0..height)
            .flat_map(|y| (0..width / 2).map(move |x| (x, y)))
            .map(|(x, y)| accum[y * width + x].iter().sum::<f64>())
            .sum();
        let right: f64 = (0..height)
            .flat_map(|y| (width / 2..width).map(move |x| (x, y)))
            .map(|(x, y)| accum[y * width + x].iter().sum::<f64>())
            .sum();
        assert!(left > 0.0 && right > 0.0, "mirror must paint both halves");
        assert!(
            (left - right).abs() / (left + right) < 0.05,
            "mirrored halves should carry near-equal energy: {left} vs {right}"
        );
    }

    #[test]
    fn rotational_fit_scale_keeps_diagonal_inside_short_edge() {
        let scale = rotational_fit_scale(3456, 2234);
        let diagonal = (3456.0f32 * 3456.0 + 2234.0 * 2234.0).sqrt();
        assert!(diagonal * scale <= 2234.0 + 1.0);
        assert!(scale > 0.3, "fit scale should not collapse the composition");
        assert_eq!(rotational_fit_scale(100, 100), 100.0 / (2.0f32).sqrt() / 100.0);
    }
}
