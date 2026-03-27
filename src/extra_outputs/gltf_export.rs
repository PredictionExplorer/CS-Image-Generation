//! 3D glTF export — writes the orbital trajectories as line-strip geometry
//! in glTF 2.0 binary (.glb) format, viewable in any 3D viewer or AR app.

use crate::render::OklabColor;
use nalgebra::Vector3;
use std::io::Write;
use tracing::info;

pub fn export_gltf(
    positions: &[Vec<Vector3<f64>>],
    colors: &[Vec<OklabColor>],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Exporting 3D model (glTF binary)...");

    let downsample = compute_downsample_factor(positions);
    let (vertices, vertex_colors, line_indices) =
        build_geometry(positions, colors, downsample);

    let glb = build_glb(&vertices, &vertex_colors, &line_indices)?;

    let mut file = std::fs::File::create(output_path)?;
    file.write_all(&glb)?;
    info!(
        "   Saved 3D model ({} vertices, {} line segments) => {}",
        vertices.len() / 3,
        line_indices.len() / 2,
        output_path
    );
    Ok(())
}

fn compute_downsample_factor(positions: &[Vec<Vector3<f64>>]) -> usize {
    let total_points: usize = positions.iter().map(|p| p.len()).sum();
    let target = 50_000;
    (total_points / target).max(1)
}

fn build_geometry(
    positions: &[Vec<Vector3<f64>>],
    colors: &[Vec<OklabColor>],
    downsample: usize,
) -> (Vec<f32>, Vec<f32>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut vertex_colors = Vec::new();
    let mut indices = Vec::new();
    let mut vertex_offset = 0u32;

    for (body_idx, body_positions) in positions.iter().enumerate() {
        let body_colors = &colors[body_idx.min(colors.len() - 1)];
        let sampled: Vec<usize> = (0..body_positions.len()).step_by(downsample).collect();

        for &i in &sampled {
            let p = body_positions[i];
            vertices.push(p.x as f32);
            vertices.push(p.y as f32);
            vertices.push(p.z as f32);

            let color_idx = i.min(body_colors.len() - 1);
            let (l, a, b) = body_colors[color_idx];
            let (r, g, b_val) = oklab_to_srgb(l, a, b);
            vertex_colors.push(r);
            vertex_colors.push(g);
            vertex_colors.push(b_val);
            vertex_colors.push(1.0);
        }

        for j in 0..(sampled.len().saturating_sub(1)) {
            indices.push(vertex_offset + j as u32);
            indices.push(vertex_offset + j as u32 + 1);
        }
        vertex_offset += sampled.len() as u32;
    }

    (vertices, vertex_colors, indices)
}

fn oklab_to_srgb(l: f64, a: f64, b: f64) -> (f32, f32, f32) {
    let l_ = l + 0.3963377774 * a + 0.2158037573 * b;
    let m_ = l - 0.1055613458 * a - 0.0638541728 * b;
    let s_ = l - 0.0894841775 * a - 1.2914855480 * b;

    let l3 = l_ * l_ * l_;
    let m3 = m_ * m_ * m_;
    let s3 = s_ * s_ * s_;

    let r = 4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3;
    let g = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3;
    let b_out = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3;

    let to_srgb = |v: f64| -> f32 {
        let c = v.clamp(0.0, 1.0);
        if c <= 0.0031308 {
            (c * 12.92) as f32
        } else {
            (1.055 * c.powf(1.0 / 2.4) - 0.055) as f32
        }
    };

    (to_srgb(r), to_srgb(g), to_srgb(b_out))
}

fn build_glb(
    vertices: &[f32],
    colors: &[f32],
    indices: &[u32],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let vertex_count = vertices.len() / 3;
    let index_count = indices.len();

    let vert_bytes: Vec<u8> = vertices.iter().flat_map(|v| v.to_le_bytes()).collect();
    let color_bytes: Vec<u8> = colors.iter().flat_map(|v| v.to_le_bytes()).collect();
    let index_bytes: Vec<u8> = indices.iter().flat_map(|v| v.to_le_bytes()).collect();

    let vert_len = vert_bytes.len();
    let color_len = color_bytes.len();
    let index_len = index_bytes.len();
    let color_offset = vert_len;
    let index_offset = vert_len + color_len;
    let total_bin_len = vert_len + color_len + index_len;

    let mut min_pos = [f32::MAX; 3];
    let mut max_pos = [f32::MIN; 3];
    for i in 0..vertex_count {
        for c in 0..3 {
            let v = vertices[i * 3 + c];
            min_pos[c] = min_pos[c].min(v);
            max_pos[c] = max_pos[c].max(v);
        }
    }

    let json = format!(
        r#"{{"asset":{{"version":"2.0","generator":"cosmic-signature"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0,"name":"orbits"}}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"COLOR_0":1}},"indices":2,"mode":1}}]}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":{vc},"type":"VEC3","max":[{maxx},{maxy},{maxz}],"min":[{minx},{miny},{minz}]}},{{"bufferView":1,"componentType":5126,"count":{vc},"type":"VEC4"}},{{"bufferView":2,"componentType":5125,"count":{ic},"type":"SCALAR"}}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":{vl},"target":34962}},{{"buffer":0,"byteOffset":{co},"byteLength":{cl},"target":34962}},{{"buffer":0,"byteOffset":{io},"byteLength":{il},"target":34963}}],"buffers":[{{"byteLength":{tl}}}]}}"#,
        vc = vertex_count,
        ic = index_count,
        vl = vert_len,
        co = color_offset,
        cl = color_len,
        io = index_offset,
        il = index_len,
        tl = total_bin_len,
        minx = min_pos[0],
        miny = min_pos[1],
        minz = min_pos[2],
        maxx = max_pos[0],
        maxy = max_pos[1],
        maxz = max_pos[2],
    );

    let json_bytes = json.as_bytes();
    let json_padded_len = (json_bytes.len() + 3) & !3;
    let bin_padded_len = (total_bin_len + 3) & !3;
    let total_length = 12 + 8 + json_padded_len + 8 + bin_padded_len;

    let mut glb = Vec::with_capacity(total_length);

    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&(total_length as u32).to_le_bytes());

    glb.extend_from_slice(&(json_padded_len as u32).to_le_bytes());
    glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes());
    glb.extend_from_slice(json_bytes);
    glb.extend(std::iter::repeat_n(b' ', json_padded_len - json_bytes.len()));

    glb.extend_from_slice(&(bin_padded_len as u32).to_le_bytes());
    glb.extend_from_slice(&0x004E4942u32.to_le_bytes());
    glb.extend_from_slice(&vert_bytes);
    glb.extend_from_slice(&color_bytes);
    glb.extend_from_slice(&index_bytes);
    glb.extend(std::iter::repeat_n(0u8, bin_padded_len - total_bin_len));

    Ok(glb)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_positions() -> Vec<Vec<Vector3<f64>>> {
        vec![
            vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(2.0, 1.0, 0.0),
            ],
            vec![
                Vector3::new(5.0, 0.0, 0.0),
                Vector3::new(5.0, 1.0, 0.0),
                Vector3::new(5.0, 2.0, 0.0),
            ],
            vec![
                Vector3::new(0.0, 5.0, 0.0),
                Vector3::new(1.0, 5.0, 0.0),
                Vector3::new(2.0, 5.0, 0.0),
            ],
        ]
    }

    fn sample_colors() -> Vec<Vec<(f64, f64, f64)>> {
        vec![
            vec![(0.7, 0.1, 0.05); 3],
            vec![(0.6, -0.1, 0.15); 3],
            vec![(0.5, 0.0, -0.2); 3],
        ]
    }

    #[test]
    fn test_compute_downsample_small_returns_one() {
        let pos = sample_positions();
        assert_eq!(compute_downsample_factor(&pos), 1);
    }

    #[test]
    fn test_compute_downsample_large_proportional() {
        let pos = vec![vec![Vector3::zeros(); 100_000]; 3];
        let factor = compute_downsample_factor(&pos);
        assert!(factor >= 5, "300k points should downsample by at least 5, got {factor}");
    }

    #[test]
    fn test_build_geometry_vertex_count() {
        let pos = sample_positions();
        let colors = sample_colors();
        let (vertices, vertex_colors, _) = build_geometry(&pos, &colors, 1);
        assert_eq!(vertices.len() / 3, 9);
        assert_eq!(vertex_colors.len() / 4, 9);
    }

    #[test]
    fn test_build_geometry_index_pairs() {
        let pos = sample_positions();
        let colors = sample_colors();
        let (_, _, indices) = build_geometry(&pos, &colors, 1);
        assert_eq!(indices.len() % 2, 0, "indices should come in pairs for line strips");
        // 3 bodies * (3-1) segments = 6 line segments = 12 indices
        assert_eq!(indices.len(), 12);
    }

    #[test]
    fn test_build_geometry_with_downsample() {
        let pos: Vec<Vec<Vector3<f64>>> = (0..3)
            .map(|_| (0..10).map(|i| Vector3::new(i as f64, 0.0, 0.0)).collect())
            .collect();
        let colors = vec![vec![(0.5, 0.0, 0.0); 10]; 3];
        let (vertices, _, _) = build_geometry(&pos, &colors, 3);
        let verts_per_body = (0..10).step_by(3).count();
        assert_eq!(vertices.len() / 3, verts_per_body * 3);
    }

    #[test]
    fn test_oklab_to_srgb_black() {
        let (r, g, b) = oklab_to_srgb(0.0, 0.0, 0.0);
        assert!(r < 0.01 && g < 0.01 && b < 0.01, "L=0 should be near black");
    }

    #[test]
    fn test_oklab_to_srgb_white() {
        let (r, g, b) = oklab_to_srgb(1.0, 0.0, 0.0);
        assert!(r > 0.95 && g > 0.95 && b > 0.95, "L=1 should be near white, got ({r}, {g}, {b})");
    }

    #[test]
    fn test_oklab_to_srgb_clamped() {
        let (r, g, b) = oklab_to_srgb(0.7, 0.2, 0.1);
        assert!((0.0..=1.0).contains(&r), "r out of range: {r}");
        assert!((0.0..=1.0).contains(&g), "g out of range: {g}");
        assert!((0.0..=1.0).contains(&b), "b out of range: {b}");
    }

    #[test]
    fn test_build_glb_magic_bytes() {
        let pos = sample_positions();
        let colors = sample_colors();
        let (v, c, i) = build_geometry(&pos, &colors, 1);
        let glb = build_glb(&v, &c, &i).unwrap();

        assert_eq!(&glb[0..4], b"glTF", "GLB must start with magic 'glTF'");
        let version = u32::from_le_bytes([glb[4], glb[5], glb[6], glb[7]]);
        assert_eq!(version, 2, "GLB version must be 2");
    }

    #[test]
    fn test_build_glb_total_length_matches_header() {
        let pos = sample_positions();
        let colors = sample_colors();
        let (v, c, i) = build_geometry(&pos, &colors, 1);
        let glb = build_glb(&v, &c, &i).unwrap();

        let declared_len = u32::from_le_bytes([glb[8], glb[9], glb[10], glb[11]]) as usize;
        assert_eq!(glb.len(), declared_len, "actual length must match header declaration");
    }

    #[test]
    fn test_build_glb_json_chunk_type() {
        let pos = sample_positions();
        let colors = sample_colors();
        let (v, c, i) = build_geometry(&pos, &colors, 1);
        let glb = build_glb(&v, &c, &i).unwrap();

        let chunk_type = u32::from_le_bytes([glb[16], glb[17], glb[18], glb[19]]);
        assert_eq!(chunk_type, 0x4E4F534A, "first chunk must be JSON (0x4E4F534A)");
    }
}
