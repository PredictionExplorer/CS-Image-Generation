//! Render-only Loop subdivision. The original simulation cache is never changed.
use super::super::{Mesh, V3};
use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::TAU;

pub(super) fn smooth(mesh: &Mesh, positions: Vec<V3>, levels: u32) -> Mesh {
    let mut result = Mesh { positions, triangles: mesh.triangles.clone(), uv: mesh.uv.clone() };
    for _ in 0..levels {
        result = step(&result);
    }
    result
}

fn step(mesh: &Mesh) -> Mesh {
    let count = mesh.positions.len();
    let mut edges: BTreeMap<(u32, u32), Vec<u32>> = BTreeMap::new();
    let mut neighbors = vec![BTreeSet::new(); count];
    for &[a, b, c] in &mesh.triangles {
        for (i, j, k) in [(a, b, c), (b, c, a), (c, a, b)] {
            edges.entry((i.min(j), i.max(j))).or_default().push(k);
            neighbors[i as usize].insert(j);
            neighbors[j as usize].insert(i);
        }
    }
    let mut boundary = vec![Vec::new(); count];
    for (&(a, b), opposites) in &edges {
        if opposites.len() == 1 {
            boundary[a as usize].push(b);
            boundary[b as usize].push(a);
        }
    }
    let mut positions = Vec::with_capacity(count + edges.len());
    let mut uv = Vec::with_capacity(count + edges.len());
    for i in 0..count {
        let (selected, beta): (Vec<u32>, f64) = if boundary[i].len() == 2 {
            (boundary[i].clone(), 0.125)
        } else {
            let n = neighbors[i].len();
            let beta = if n > 2 {
                (0.625 - (0.375 + 0.25 * (TAU / n as f64).cos()).powi(2)) / n as f64
            } else {
                0.0
            };
            (neighbors[i].iter().copied().collect(), beta)
        };
        let base = 1.0 - beta * selected.len() as f64;
        let mut position = mesh.positions[i] * base;
        let mut texcoord = [mesh.uv[i][0] * base, mesh.uv[i][1] * base];
        for j in selected {
            position += mesh.positions[j as usize] * beta;
            texcoord[0] += mesh.uv[j as usize][0] * beta;
            texcoord[1] += mesh.uv[j as usize][1] * beta;
        }
        positions.push(position);
        uv.push(texcoord);
    }
    let mut midpoint = BTreeMap::new();
    for (&(a, b), opposites) in &edges {
        midpoint.insert((a, b), positions.len() as u32);
        let mut p = (mesh.positions[a as usize] + mesh.positions[b as usize]) * 0.5;
        let mut t = [
            (mesh.uv[a as usize][0] + mesh.uv[b as usize][0]) * 0.5,
            (mesh.uv[a as usize][1] + mesh.uv[b as usize][1]) * 0.5,
        ];
        if opposites.len() == 2 {
            p *= 0.75;
            t[0] *= 0.75;
            t[1] *= 0.75;
            for &i in opposites {
                p += mesh.positions[i as usize] * 0.125;
                t[0] += mesh.uv[i as usize][0] * 0.125;
                t[1] += mesh.uv[i as usize][1] * 0.125;
            }
        }
        positions.push(p);
        uv.push(t);
    }
    let edge = |a: u32, b: u32| midpoint[&(a.min(b), a.max(b))];
    let mut triangles = Vec::with_capacity(mesh.triangles.len() * 4);
    for &[a, b, c] in &mesh.triangles {
        let (ab, bc, ca) = (edge(a, b), edge(b, c), edge(c, a));
        triangles.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
    }
    Mesh { positions, triangles, uv }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn subdivision_keeps_uvs_affine_and_topology_repeatable() {
        let mesh = Mesh {
            positions: vec![
                V3::new(0.0, 0.0, 0.0),
                V3::new(1.0, 0.0, 0.0),
                V3::new(1.0, 1.0, 0.0),
                V3::new(0.0, 1.0, 0.0),
            ],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            uv: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        };
        let a = smooth(&mesh, mesh.positions.clone(), 2);
        let b = smooth(&mesh, mesh.positions.clone(), 2);
        assert_eq!(a.triangles.len(), 32);
        assert_eq!(a.positions, b.positions);
        assert_eq!(a.triangles, b.triangles);
        for (p, t) in a.positions.iter().zip(&a.uv) {
            assert!((p.x - t[0]).abs() < 1e-12 && (p.y - t[1]).abs() < 1e-12);
        }
        assert_eq!(
            mesh.positions[0],
            V3::ZERO,
            "render subdivision must not alter the source cache"
        );
    }
}
