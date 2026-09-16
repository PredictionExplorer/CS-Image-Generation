//! Deterministic zero-isosurface extraction and closed-surface topology audits.
//!
//! Every cube uses the same six tetrahedra around its 0--7 body diagonal. The
//! resulting face diagonals agree between neighboring cubes. Intersections are
//! shared by their global grid-edge identities, including face/body diagonals;
//! exact zero endpoints use the grid-node identity itself. Interpolated roots
//! within a recorded float32-scale tolerance of a grid endpoint also use that
//! canonical node, bounding the displacement without general vertex welding.
//! No artificial caps or selection/merging of connected components occurs.
//!
//! Exact zero faces separating negative tetrahedra cancel in opposite pairs.
//! Collapsed zero triangles are omitted, then unused vertices are compacted in
//! insertion order. A remaining zero-contact pinch is rejected by the topology
//! audit rather than resolved by moving geometry. Audits do not establish the
//! absence of general geometric self-intersections in arbitrary input meshes.
//! A separate strict audit rejects triangles that collapse after native f32
//! coordinate conversion. Canonical mesh coordinates remain f64.

use super::Grid;
use crate::silk::{SilkResult, V3};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

const TETRAHEDRA: [[usize; 4]; 6] =
    [[0, 5, 1, 7], [0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7], [0, 6, 4, 7], [0, 4, 5, 7]];

/// Explicit oriented triangle surface, with negative material on its inward side.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Mesh {
    /// World-coordinate vertices after the recorded grid-endpoint conditioning.
    pub vertices: Vec<V3>,
    /// Counterclockwise triangles as seen from outside the surviving material.
    pub triangles: Vec<[u32; 3]>,
}

/// Measured topology and geometry; surface components include inner cavity walls.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeshDiagnostics {
    /// Axis-aligned world bounds, or None for an empty vertex list.
    pub bounds: Option<[V3; 2]>,
    /// Sum of oriented component volumes; inward-facing cavity walls subtract.
    pub signed_volume: f64,
    /// Number of components connected through triangle edges, not solid regions.
    pub components: usize,
    /// Signed volumes in stable lowest-triangle-index component order.
    pub component_signed_volumes: Vec<f64>,
    /// Number of edges incident to exactly one triangle.
    pub boundary_edges: usize,
    /// Number of edges incident to more than two triangles.
    pub nonmanifold_edges: usize,
    /// Two-face edges whose incident triangles traverse them in the same direction.
    pub orientation_conflicts: usize,
    /// Repeated-index or exactly zero-area triangles.
    pub degenerate_triangles: usize,
    /// Triangles repeating a prior triangle's three vertex identities.
    pub duplicate_triangles: usize,
    /// Used vertices whose incident-triangle link is not one closed cycle.
    pub nonmanifold_vertices: usize,
    /// Vertices referenced by no triangle.
    pub unreferenced_vertices: usize,
    /// V-E+F for the referenced surface, useful for independent genus checks.
    pub euler_characteristic: i64,
    /// False: this audit does not perform geometric triangle/triangle tests.
    pub self_intersections_checked: bool,
}

/// Explicit numerical conditioning used to keep native f32 imports nondegenerate.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ExtractionDiagnostics {
    /// Maximum permitted world-space displacement to an existing grid endpoint.
    pub endpoint_snap_tolerance: f64,
    /// Number of distinct intersected grid/tetrahedral edges mapped to an endpoint.
    pub snapped_edge_roots: usize,
    /// Largest measured displacement from an unsnapped interpolated root.
    pub maximum_endpoint_displacement: f64,
    /// Vertices that are nonfinite after native f32 conversion; zero on success.
    pub float32_nonfinite_vertices: usize,
    /// Triangles with exactly zero area after f32 conversion; zero on success.
    pub float32_collapsed_triangles: usize,
}

#[derive(Clone, Copy)]
struct Corner {
    point: V3,
    value: f64,
    id: usize,
}

struct FaceRecord {
    slot: usize,
    orientation: i8,
    cancelled: bool,
}

struct Builder {
    vertices: Vec<V3>,
    faces: Vec<Option<[u32; 3]>>,
    edges: HashMap<(usize, usize), u32>,
    face_records: HashMap<[u32; 3], FaceRecord>,
    endpoint_snap_tolerance: f64,
    snapped_edge_roots: usize,
    maximum_endpoint_displacement: f64,
}

fn length(v: V3) -> f64 {
    v.x.hypot(v.y).hypot(v.z)
}

fn checked_grid(grid: &Grid) -> SilkResult<()> {
    if grid.dims.iter().any(|&n| n < 2)
        || !grid.origin.is_finite()
        || !grid.spacing.is_finite()
        || grid.spacing <= 0.0
    {
        return Err("remaining-form grid requires finite coordinates, positive spacing and at least two nodes per axis".into());
    }
    let count = grid
        .dims
        .into_iter()
        .try_fold(1usize, usize::checked_mul)
        .ok_or("remaining-form grid dimensions overflow")?;
    if count != grid.values.len() || grid.values.iter().any(|v| !v.is_finite()) {
        return Err("remaining-form grid has invalid scalar count or nonfinite values".into());
    }
    let extent = grid.dims.map(|n| (n - 1) as f64 * grid.spacing);
    let maximum_extent = extent.into_iter().fold(0.0, f64::max);
    if !maximum_extent.powi(3).is_finite()
        || maximum_extent.powi(3) == 0.0
        || grid.spacing.powi(3) == 0.0
    {
        return Err("remaining-form grid scale exceeds reliable f64 area/volume arithmetic".into());
    }
    for (axis, &span) in extent.iter().enumerate() {
        let start = grid.origin.axis(axis);
        let end = start + span;
        let previous = start + (grid.dims[axis] - 2) as f64 * grid.spacing;
        if !end.is_finite() || start + grid.spacing <= start || end <= previous {
            return Err(
                "remaining-form grid spacing is not representable at its world origin".into()
            );
        }
    }
    let [nx, ny, nz] = grid.dims;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if (x == 0 || y == 0 || z == 0 || x + 1 == nx || y + 1 == ny || z + 1 == nz)
                    && grid.values[(z * ny + y) * nx + x] <= 0.0
                {
                    return Err("remaining-form material reaches the grid boundary; enlarge the sampled domain rather than cap the surface".into());
                }
            }
        }
    }
    Ok(())
}

/// Extract a closed oriented zero-isosurface without changing the supplied grid.
///
/// Outer grid nodes must be strictly positive, so an under-sized domain cannot
/// silently generate an open or artificially capped object. Ambiguous zero-node
/// contacts that remain nonmanifold are errors. No self-intersection audit is
/// claimed beyond the conforming tetrahedral construction and topology checks.
pub fn extract(grid: &Grid) -> SilkResult<Mesh> {
    extract_with_diagnostics(grid).map(|(mesh, _)| mesh)
}

/// Extract with bounded grid-endpoint conditioning and strict native-f32 auditing.
///
/// The tolerance is four f32 epsilons at the grid's coordinate magnitude, capped
/// at one thousandth of a grid cell. Only roots near an existing endpoint move.
/// Topology defects caused or exposed by this conditioning are errors, never
/// repaired by discarding components or weakening the manifold checks.
pub fn extract_with_diagnostics(grid: &Grid) -> SilkResult<(Mesh, ExtractionDiagnostics)> {
    checked_grid(grid)?;
    let mut coordinate_scale = 1.0_f64;
    for axis in 0..3 {
        coordinate_scale = coordinate_scale
            .max(grid.origin.axis(axis).abs())
            .max((grid.origin.axis(axis) + (grid.dims[axis] - 1) as f64 * grid.spacing).abs());
    }
    let tolerance = (4.0 * f64::from(f32::EPSILON) * coordinate_scale).min(grid.spacing * 0.001);
    let (mesh, mut diagnostics) = extract_conditioned(grid, tolerance)?;
    let (invalid_vertices, collapsed_faces) = float32_failures(&mesh);
    diagnostics.float32_nonfinite_vertices = invalid_vertices;
    diagnostics.float32_collapsed_triangles = collapsed_faces;
    if invalid_vertices != 0 || collapsed_faces != 0 {
        return Err(format!(
            "remaining-form mesh fails native f32 export precision: {diagnostics:?}; no triangles or components were discarded to bypass this check"
        ).into());
    }
    Ok((mesh, diagnostics))
}

/// The caller validates the grid; the unconditioned path is used only by tests.
fn extract_conditioned(grid: &Grid, tolerance: f64) -> SilkResult<(Mesh, ExtractionDiagnostics)> {
    let [nx, ny, nz] = grid.dims;
    let mut builder = Builder {
        vertices: Vec::new(),
        faces: Vec::new(),
        edges: HashMap::new(),
        face_records: HashMap::new(),
        endpoint_snap_tolerance: tolerance,
        snapped_edge_roots: 0,
        maximum_endpoint_displacement: 0.0,
    };
    for z in 0..nz - 1 {
        for y in 0..ny - 1 {
            for x in 0..nx - 1 {
                let corners: [Corner; 8] = std::array::from_fn(|bit| {
                    let ix = x + (bit & 1);
                    let iy = y + ((bit >> 1) & 1);
                    let iz = z + ((bit >> 2) & 1);
                    let id = (iz * ny + iy) * nx + ix;
                    Corner {
                        id,
                        value: grid.values[id],
                        point: grid.origin
                            + V3::new(ix as f64, iy as f64, iz as f64) * grid.spacing,
                    }
                });
                if corners.iter().all(|p| p.value >= 0.0) || corners.iter().all(|p| p.value < 0.0) {
                    continue;
                }
                for tetrahedron in TETRAHEDRA {
                    builder.tetrahedron(tetrahedron.map(|i| corners[i]))?;
                }
            }
        }
    }
    let diagnostics = ExtractionDiagnostics {
        endpoint_snap_tolerance: tolerance,
        snapped_edge_roots: builder.snapped_edge_roots,
        maximum_endpoint_displacement: builder.maximum_endpoint_displacement,
        ..ExtractionDiagnostics::default()
    };
    let mesh = builder.finish()?;
    mesh.validate()?;
    Ok((mesh, diagnostics))
}

impl Builder {
    fn vertex(&mut self, a: Corner, b: Corner) -> SilkResult<u32> {
        let (a, b) = if a.id <= b.id { (a, b) } else { (b, a) };
        let edge_key = (a.id, b.id);
        if let Some(&vertex) = self.edges.get(&edge_key) {
            return Ok(vertex);
        }
        let (point, key) = if a.value == 0.0 {
            (a.point, (a.id, a.id))
        } else if b.value == 0.0 {
            (b.point, (b.id, b.id))
        } else {
            let av = a.value.abs();
            let bv = b.value.abs();
            // Opposite-sign interpolation without overflowing av+bv or a-b.
            let t = if av >= bv {
                1.0 / (1.0 + bv / av)
            } else {
                let ratio = av / bv;
                ratio / (1.0 + ratio)
            };
            let original = a.point + (b.point - a.point) * t;
            let distance_a = length(original - a.point);
            let distance_b = length(original - b.point);
            let (endpoint, distance) =
                if distance_a <= distance_b { (a, distance_a) } else { (b, distance_b) };
            if self.endpoint_snap_tolerance > 0.0 && distance <= self.endpoint_snap_tolerance {
                self.snapped_edge_roots += 1;
                self.maximum_endpoint_displacement =
                    self.maximum_endpoint_displacement.max(distance);
                (endpoint.point, (endpoint.id, endpoint.id))
            } else {
                (original, edge_key)
            }
        };
        if !point.is_finite() {
            return Err("remaining-form edge intersection is not finite".into());
        }
        if let Some(&vertex) = self.edges.get(&key) {
            self.edges.insert(edge_key, vertex);
            return Ok(vertex);
        }
        let index = u32::try_from(self.vertices.len())
            .map_err(|_| "remaining-form vertex index exceeds u32")?;
        self.vertices.try_reserve(1)?;
        self.edges.try_reserve(1)?;
        self.vertices.push(point);
        self.edges.insert(key, index);
        self.edges.insert(edge_key, index);
        Ok(index)
    }

    fn triangle(&mut self, mut triangle: [u32; 3], outward: V3) -> SilkResult<()> {
        if triangle[0] == triangle[1] || triangle[1] == triangle[2] || triangle[2] == triangle[0] {
            return Ok(());
        }
        let [a, b, c] = triangle.map(|i| self.vertices[i as usize]);
        let normal = (b - a).cross(c - a);
        if !normal.is_finite() {
            return Err("remaining-form triangle normal overflow".into());
        }
        if length(normal) == 0.0 {
            return Ok(());
        }
        let orientation = normal.dot(outward);
        if !orientation.is_finite() || orientation == 0.0 {
            return Err("remaining-form triangle orientation is numerically indeterminate".into());
        }
        if orientation < 0.0 {
            triangle.swap(1, 2);
        }
        let key = sorted_triangle(triangle);
        let sign = triangle_orientation(triangle);
        if let Some(existing) = self.face_records.get_mut(&key) {
            if existing.cancelled || existing.orientation == sign {
                return Err("remaining-form zero face has inconsistent duplicate incidence".into());
            }
            self.faces[existing.slot] = None;
            existing.cancelled = true;
            return Ok(());
        }
        self.faces.try_reserve(1)?;
        self.face_records.try_reserve(1)?;
        self.face_records.insert(
            key,
            FaceRecord { slot: self.faces.len(), orientation: sign, cancelled: false },
        );
        self.faces.push(Some(triangle));
        Ok(())
    }

    fn tetrahedron(&mut self, corners: [Corner; 4]) -> SilkResult<()> {
        let mut inside = [0usize; 4];
        let mut outside = [0usize; 4];
        let (mut ni, mut no) = (0, 0);
        for (index, corner) in corners.iter().enumerate() {
            if corner.value < 0.0 {
                inside[ni] = index;
                ni += 1;
            } else {
                outside[no] = index;
                no += 1;
            }
        }
        if ni == 0 || ni == 4 {
            return Ok(());
        }
        let outward = tetrahedron_gradient(corners)?;
        match ni {
            1 => {
                let a = self.vertex(corners[inside[0]], corners[outside[0]])?;
                let b = self.vertex(corners[inside[0]], corners[outside[1]])?;
                let c = self.vertex(corners[inside[0]], corners[outside[2]])?;
                self.triangle([a, b, c], outward)?;
            }
            3 => {
                let a = self.vertex(corners[inside[0]], corners[outside[0]])?;
                let b = self.vertex(corners[inside[1]], corners[outside[0]])?;
                let c = self.vertex(corners[inside[2]], corners[outside[0]])?;
                self.triangle([a, b, c], outward)?;
            }
            2 => {
                let a = self.vertex(corners[inside[0]], corners[outside[0]])?;
                let b = self.vertex(corners[inside[0]], corners[outside[1]])?;
                let c = self.vertex(corners[inside[1]], corners[outside[0]])?;
                let d = self.vertex(corners[inside[1]], corners[outside[1]])?;
                self.triangle([a, b, d], outward)?;
                self.triangle([a, d, c], outward)?;
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    fn finish(self) -> SilkResult<Mesh> {
        let mut triangles: Vec<[u32; 3]> = self.faces.into_iter().flatten().collect();
        let mut used = vec![false; self.vertices.len()];
        for triangle in &triangles {
            for &v in triangle {
                used[v as usize] = true;
            }
        }
        let mut remap = vec![0u32; used.len()];
        let mut vertices = Vec::new();
        vertices.try_reserve_exact(used.iter().filter(|&&v| v).count())?;
        for (index, point) in self.vertices.into_iter().enumerate() {
            if used[index] {
                remap[index] = u32::try_from(vertices.len())
                    .map_err(|_| "remaining-form vertex index overflow")?;
                vertices.push(point);
            }
        }
        for triangle in &mut triangles {
            for v in triangle {
                *v = remap[*v as usize];
            }
        }
        Ok(Mesh { vertices, triangles })
    }
}

/// Direction of increasing affine scalar value within this tetrahedron.
/// Scaling coordinates and values avoids overflow and makes orientation robust
/// when a conditioned root coincides with the formerly negative reference corner.
fn tetrahedron_gradient(corners: [Corner; 4]) -> SilkResult<V3> {
    let edges = [
        corners[1].point - corners[0].point,
        corners[2].point - corners[0].point,
        corners[3].point - corners[0].point,
    ];
    let coordinate_scale =
        edges.iter().flat_map(|v| [v.x.abs(), v.y.abs(), v.z.abs()]).fold(0.0, f64::max);
    let value_scale = corners.iter().map(|corner| corner.value.abs()).fold(0.0, f64::max);
    if coordinate_scale == 0.0 || value_scale == 0.0 {
        return Err("remaining-form tetrahedron has an indeterminate scalar gradient".into());
    }
    let [u, v, w] = edges.map(|edge| edge / coordinate_scale);
    let determinant = u.dot(v.cross(w));
    let base = corners[0].value / value_scale;
    let differences = [
        corners[1].value / value_scale - base,
        corners[2].value / value_scale - base,
        corners[3].value / value_scale - base,
    ];
    let gradient =
        (v.cross(w) * differences[0] + w.cross(u) * differences[1] + u.cross(v) * differences[2])
            * determinant.signum();
    let magnitude = gradient.x.abs().max(gradient.y.abs()).max(gradient.z.abs());
    if !determinant.is_finite() || determinant == 0.0 || !gradient.is_finite() || magnitude == 0.0 {
        return Err("remaining-form tetrahedron gradient is numerically indeterminate".into());
    }
    Ok(gradient / magnitude)
}

/// Match native import precision, then promote before subtraction/cross products.
/// This detects actual collapsed geometry without float32 intermediate cancellation.
fn float32_failures(mesh: &Mesh) -> (usize, usize) {
    let points: Vec<V3> = mesh
        .vertices
        .iter()
        .map(|point| {
            V3::new(f64::from(point.x as f32), f64::from(point.y as f32), f64::from(point.z as f32))
        })
        .collect();
    let nonfinite = points.iter().filter(|point| !point.is_finite()).count();
    let collapsed = mesh
        .triangles
        .iter()
        .filter(|triangle| {
            let [a, b, c] = triangle.map(|index| points[index as usize]);
            a.is_finite() && b.is_finite() && c.is_finite() && length((b - a).cross(c - a)) == 0.0
        })
        .count();
    (nonfinite, collapsed)
}

fn sorted_triangle(mut triangle: [u32; 3]) -> [u32; 3] {
    triangle.sort_unstable();
    triangle
}

fn triangle_orientation(triangle: [u32; 3]) -> i8 {
    let inversions = usize::from(triangle[0] > triangle[1])
        + usize::from(triangle[0] > triangle[2])
        + usize::from(triangle[1] > triangle[2]);
    if inversions.is_multiple_of(2) { 1 } else { -1 }
}

struct UnionFind(Vec<usize>);
impl UnionFind {
    fn new(count: usize) -> Self {
        Self((0..count).collect())
    }
    fn root(&mut self, mut i: usize) -> usize {
        while self.0[i] != i {
            self.0[i] = self.0[self.0[i]];
            i = self.0[i];
        }
        i
    }
    fn join(&mut self, a: usize, b: usize) {
        let a = self.root(a);
        let b = self.root(b);
        self.0[a.max(b)] = a.min(b);
    }
}

#[derive(Default)]
struct Sum {
    value: f64,
    correction: f64,
}
impl Sum {
    fn add(&mut self, term: f64) {
        let next = term - self.correction;
        let value = self.value + next;
        self.correction = (value - self.value) - next;
        self.value = value;
    }
}

fn valid_link(links: &[[u32; 2]]) -> bool {
    let mut neighbors: Vec<u32> = links.iter().flatten().copied().collect();
    neighbors.sort_unstable();
    neighbors.dedup();
    if neighbors.len() < 3 {
        return false;
    }
    let mut degree = vec![0usize; neighbors.len()];
    let mut sets = UnionFind::new(neighbors.len());
    for &[a, b] in links {
        if a == b {
            return false;
        }
        let a = neighbors.binary_search(&a).expect("link vertex was collected");
        let b = neighbors.binary_search(&b).expect("link vertex was collected");
        degree[a] += 1;
        degree[b] += 1;
        sets.join(a, b);
    }
    degree.into_iter().all(|n| n == 2) && (0..neighbors.len()).all(|i| sets.root(i) == 0)
}

impl Mesh {
    /// Report topology defects without claiming general self-intersection freedom.
    pub fn audit(&self) -> SilkResult<MeshDiagnostics> {
        if self.vertices.iter().any(|p| !p.is_finite())
            || self.triangles.iter().flatten().any(|&v| v as usize >= self.vertices.len())
        {
            return Err("remaining-form mesh has invalid coordinates or triangle indices".into());
        }
        let bounds = self.vertices.first().map(|&first| {
            self.vertices.iter().fold([first, first], |b, &p| [b[0].min(p), b[1].max(p)])
        });
        let reference = bounds.map_or(V3::ZERO, |b| b[0] * 0.5 + b[1] * 0.5);
        let mut edges: HashMap<(u32, u32), (usize, i64, usize)> = HashMap::new();
        let mut links = vec![Vec::<[u32; 2]>::new(); self.vertices.len()];
        let mut sets = UnionFind::new(self.triangles.len());
        let mut faces = HashSet::new();
        let mut degenerate_triangles = 0;
        let mut duplicate_triangles = 0;
        let mut volumes = Vec::new();
        volumes.try_reserve_exact(self.triangles.len())?;
        for (index, &triangle) in self.triangles.iter().enumerate() {
            let [a, b, c] = triangle.map(|i| self.vertices[i as usize] - reference);
            let normal = (b - a).cross(c - a);
            let volume = a.dot(b.cross(c)) / 6.0;
            if !normal.is_finite() || !volume.is_finite() {
                return Err("remaining-form mesh area or volume arithmetic overflow".into());
            }
            degenerate_triangles += usize::from(
                length(normal) == 0.0
                    || triangle[0] == triangle[1]
                    || triangle[1] == triangle[2]
                    || triangle[2] == triangle[0],
            );
            duplicate_triangles += usize::from(!faces.insert(sorted_triangle(triangle)));
            volumes.push(volume);
            for (v, a, b) in [
                (triangle[0], triangle[1], triangle[2]),
                (triangle[1], triangle[2], triangle[0]),
                (triangle[2], triangle[0], triangle[1]),
            ] {
                links[v as usize].push([a, b]);
                let key = (v.min(a), v.max(a));
                let entry = edges.entry(key).or_insert((0, 0, index));
                entry.0 += 1;
                entry.1 += if v < a { 1 } else { -1 };
                sets.join(index, entry.2);
            }
        }
        let mut components: BTreeMap<usize, Sum> = BTreeMap::new();
        let mut signed_volume = Sum::default();
        for (i, volume) in volumes.into_iter().enumerate() {
            components.entry(sets.root(i)).or_default().add(volume);
            signed_volume.add(volume);
        }
        let unreferenced_vertices = links.iter().filter(|link| link.is_empty()).count();
        let vertices = i64::try_from(self.vertices.len() - unreferenced_vertices)?;
        let edge_count = i64::try_from(edges.len())?;
        let triangles = i64::try_from(self.triangles.len())?;
        Ok(MeshDiagnostics {
            bounds,
            signed_volume: signed_volume.value,
            components: components.len(),
            component_signed_volumes: components.into_values().map(|v| v.value).collect(),
            boundary_edges: edges.values().filter(|e| e.0 == 1).count(),
            nonmanifold_edges: edges.values().filter(|e| e.0 > 2).count(),
            orientation_conflicts: edges.values().filter(|e| e.0 == 2 && e.1 != 0).count(),
            degenerate_triangles,
            duplicate_triangles,
            nonmanifold_vertices: links
                .iter()
                .filter(|link| !link.is_empty() && !valid_link(link))
                .count(),
            unreferenced_vertices,
            euler_characteristic: vertices - edge_count + triangles,
            self_intersections_checked: false,
        })
    }

    /// Require a nonempty consistently oriented closed manifold with positive volume.
    ///
    /// Separate negative-volume cavity wall components are valid. This is a
    /// topology/finite-geometry gate, not a general self-intersection certificate.
    pub fn validate(&self) -> SilkResult<MeshDiagnostics> {
        let result = self.audit()?;
        if self.triangles.is_empty()
            || result.signed_volume <= 0.0
            || !result.signed_volume.is_finite()
            || result.boundary_edges != 0
            || result.nonmanifold_edges != 0
            || result.orientation_conflicts != 0
            || result.degenerate_triangles != 0
            || result.duplicate_triangles != 0
            || result.nonmanifold_vertices != 0
            || result.unreferenced_vertices != 0
        {
            return Err(format!(
                "remaining-form surface failed its closed-manifold audit: {result:?}"
            )
            .into());
        }
        Ok(result)
    }

    /// Area-weighted outward vertex normals in deterministic triangle order.
    pub fn vertex_normals(&self) -> SilkResult<Vec<V3>> {
        self.validate()?;
        let mut normals = vec![V3::ZERO; self.vertices.len()];
        let mut first = vec![V3::ZERO; self.vertices.len()];
        for &triangle in &self.triangles {
            let [a, b, c] = triangle.map(|i| self.vertices[i as usize]);
            let normal = (b - a).cross(c - a);
            for v in triangle {
                normals[v as usize] += normal;
                if first[v as usize] == V3::ZERO {
                    first[v as usize] = normal;
                }
            }
        }
        for (normal, first) in normals.iter_mut().zip(first) {
            if length(*normal) == 0.0 {
                *normal = first;
            }
            *normal /= length(*normal);
            if !normal.is_finite() {
                return Err("remaining-form mesh normal is not finite".into());
            }
        }
        Ok(normals)
    }
}

/// Atomically publish a deterministic binary-little-endian PLY without overwriting.
///
/// Coordinates and area-weighted normals use f64; face indices use u32. The file
/// contains no timestamps, process state, or output-path-dependent comments.
pub fn write_ply(path: &Path, mesh: &Mesh) -> SilkResult<()> {
    let normals = mesh.vertex_normals()?;
    write_ply_with_normals(path, mesh, &normals)
}

/// Publish the same f64 positions and indexed faces with caller-supplied unit normals.
///
/// The normal count must match vertices; every normal must be finite and unit
/// length within 1e-8. Values are validated and written unchanged. This changes
/// shading attributes only and performs no geometry movement or re-extraction.
pub fn write_ply_with_normals(path: &Path, mesh: &Mesh, normals: &[V3]) -> SilkResult<()> {
    if normals.len() != mesh.vertices.len()
        || normals.iter().any(|normal| !normal.is_finite() || (length(*normal) - 1.0).abs() > 1e-8)
    {
        return Err("remaining-form PLY needs one finite unit normal per vertex".into());
    }
    mesh.validate()?;
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    if path.exists() {
        return Err("remaining-form PLY output already exists".into());
    }
    let mut temporary_name = path.as_os_str().to_owned();
    temporary_name.push(".partial");
    let temporary = Path::new(&temporary_name);
    let file = OpenOptions::new().write(true).create_new(true).open(temporary)?;
    let result = (|| -> SilkResult<()> {
        let mut writer = BufWriter::new(file);
        write!(
            writer,
            "ply\nformat binary_little_endian 1.0\ncomment Remaining Form oriented isosurface\nelement vertex {}\nproperty double x\nproperty double y\nproperty double z\nproperty double nx\nproperty double ny\nproperty double nz\nelement face {}\nproperty list uchar uint vertex_indices\nend_header\n",
            mesh.vertices.len(),
            mesh.triangles.len()
        )?;
        for (&p, n) in mesh.vertices.iter().zip(normals) {
            for value in [p.x, p.y, p.z, n.x, n.y, n.z] {
                writer.write_all(&value.to_le_bytes())?;
            }
        }
        for triangle in &mesh.triangles {
            writer.write_all(&[3])?;
            for v in triangle {
                writer.write_all(&v.to_le_bytes())?;
            }
        }
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        // Same-directory hard-link publication is atomic and refuses an existing destination.
        fs::hard_link(temporary, path)?;
        Ok(())
    })();
    let cleanup = fs::remove_file(temporary);
    result?;
    cleanup?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn grid(n: usize, half: f64, field: impl Fn(V3) -> f64) -> Grid {
        let spacing = 2.0 * half / (n - 1) as f64;
        let origin = V3::new(-half, -half, -half);
        let mut values = Vec::new();
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    values.push(field(origin + V3::new(x as f64, y as f64, z as f64) * spacing));
                }
            }
        }
        Grid { dims: [n; 3], origin, spacing, values }
    }

    #[test]
    fn sphere_is_closed_outward_and_handles_exact_zero_grid_nodes() {
        let sphere = grid(17, 2.0, |p| p.length_squared() - 1.0);
        let mesh = extract(&sphere).unwrap();
        let audit = mesh.validate().unwrap();
        assert_eq!(audit.components, 1);
        assert_eq!(audit.euler_characteristic, 2);
        assert!((audit.signed_volume - 4.0 * PI / 3.0).abs() < 0.2);
        for triangle in &mesh.triangles {
            let [a, b, c] = triangle.map(|i| mesh.vertices[i as usize]);
            assert!((b - a).cross(c - a).dot(a + b + c) > 0.0);
        }
        for p in &mesh.vertices {
            assert!(length(*p) <= 1.0 + 1e-12);
        }
        assert!(!audit.self_intersections_checked);
    }

    #[test]
    fn torus_preserves_its_genus_one_tunnel() {
        let torus = grid(33, 2.0, |p| (p.x.hypot(p.y) - 0.95).hypot(p.z) - 0.36);
        let audit = extract(&torus).unwrap().validate().unwrap();
        assert_eq!(audit.components, 1);
        assert_eq!(audit.euler_characteristic, 0);
        let expected = 2.0 * PI * PI * 0.95 * 0.36_f64.powi(2);
        assert!((audit.signed_volume - expected).abs() / expected < 0.08);
    }

    #[test]
    fn spherical_annulus_preserves_separate_inward_cavity_wall() {
        let shell = grid(33, 1.6, |p| (length(p) - 1.1).max(0.65 - length(p)));
        let audit = extract(&shell).unwrap().validate().unwrap();
        assert_eq!(audit.components, 2);
        assert_eq!(audit.euler_characteristic, 4);
        assert_eq!(audit.component_signed_volumes.iter().filter(|&&v| v < 0.0).count(), 1);
        let expected = 4.0 * PI / 3.0 * (1.1_f64.powi(3) - 0.65_f64.powi(3));
        assert!((audit.signed_volume - expected).abs() / expected < 0.04);
    }

    #[test]
    fn extraction_and_binary_ply_are_repeatable_without_overwrites() {
        let source = grid(15, 1.8, |p| length(p) - 1.0);
        let a = extract(&source).unwrap();
        let b = extract(&source).unwrap();
        assert_eq!(a.vertices, b.vertices);
        assert_eq!(a.triangles, b.triangles);
        let directory = tempfile::tempdir().unwrap();
        let first = directory.path().join("one.ply");
        let second = directory.path().join("two.ply");
        write_ply(&first, &a).unwrap();
        write_ply(&second, &b).unwrap();
        let bytes = fs::read(&first).unwrap();
        assert_eq!(bytes, fs::read(&second).unwrap());
        assert!(bytes.starts_with(b"ply\nformat binary_little_endian 1.0\n"));
        assert!(write_ply(&first, &a).is_err());
        assert_eq!(bytes, fs::read(&first).unwrap());
    }

    #[test]
    fn supplied_normal_export_is_stable_and_preserves_all_geometry_bytes() {
        let mesh = extract(&grid(15, 1.8, |p| length(p) - 1.0)).unwrap();
        let normals: Vec<V3> = mesh.vertices.iter().map(|p| *p / length(*p)).collect();
        let directory = tempfile::tempdir().unwrap();
        let baseline = directory.path().join("mesh.ply");
        let custom = directory.path().join("field.ply");
        let repeated = directory.path().join("repeated.ply");
        write_ply(&baseline, &mesh).unwrap();
        write_ply_with_normals(&custom, &mesh, &normals).unwrap();
        write_ply_with_normals(&repeated, &mesh, &normals).unwrap();
        let a = fs::read(baseline).unwrap();
        let b = fs::read(custom).unwrap();
        assert_eq!(b, fs::read(repeated).unwrap());
        let marker = b"end_header\n";
        let header =
            a.windows(marker.len()).position(|part| part == marker).unwrap() + marker.len();
        assert_eq!(&a[..header], &b[..header]);
        for (index, normal) in normals.iter().enumerate() {
            let start = header + index * 48;
            assert_eq!(&a[start..start + 24], &b[start..start + 24]);
            for (axis, value) in [normal.x, normal.y, normal.z].into_iter().enumerate() {
                assert_eq!(&b[start + 24 + axis * 8..start + 32 + axis * 8], &value.to_le_bytes());
            }
        }
        let faces = header + mesh.vertices.len() * 48;
        assert_eq!(&a[faces..], &b[faces..]);
    }

    #[test]
    fn supplied_normals_require_matching_count_and_finite_unit_vectors() {
        let mesh = extract(&grid(9, 2.0, |p| p.length_squared() - 1.0)).unwrap();
        let normals = mesh.vertex_normals().unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("invalid.ply");
        assert!(write_ply_with_normals(&path, &mesh, &normals[..normals.len() - 1]).is_err());
        for invalid in [V3::ZERO, V3::new(f64::NAN, 0.0, 1.0), V3::new(0.0, 0.0, 2.0)] {
            let mut bad = normals.clone();
            bad[0] = invalid;
            assert!(write_ply_with_normals(&path, &mesh, &bad).is_err());
            assert!(!path.exists());
            assert!(!directory.path().join("invalid.ply.partial").exists());
        }
    }

    #[test]
    fn bounded_endpoint_conditioning_prevents_real_float32_corner_collapse() {
        let mut sphere = grid(17, 2.0, |point| point.length_squared() - (1.0 + 1e-8));
        // Nonzero coordinates reproduce native precision loss around the
        // observed sculpture corner, rather than relying on arithmetic near zero.
        sphere.origin += V3::new(-0.6294820717131473, -1.1762948207171315, -0.8328685258964144);
        checked_grid(&sphere).unwrap();
        let (unconditioned, _) = extract_conditioned(&sphere, 0.0).unwrap();
        assert!(float32_failures(&unconditioned).1 > 0);
        let (conditioned, diagnostics) = extract_with_diagnostics(&sphere).unwrap();
        assert!(diagnostics.snapped_edge_roots > 0);
        assert!(diagnostics.maximum_endpoint_displacement <= diagnostics.endpoint_snap_tolerance);
        assert!(diagnostics.endpoint_snap_tolerance <= sphere.spacing * 0.001);
        assert_eq!(float32_failures(&conditioned), (0, 0));
        let audit = conditioned.validate().unwrap();
        assert_eq!(audit.components, 1);
        assert_eq!(audit.euler_characteristic, 2);
        let (again, evidence) = extract_with_diagnostics(&sphere).unwrap();
        assert_eq!(conditioned.vertices, again.vertices);
        assert_eq!(conditioned.triangles, again.triangles);
        assert_eq!(
            serde_json::to_vec(&diagnostics).unwrap(),
            serde_json::to_vec(&evidence).unwrap()
        );
    }

    #[test]
    fn snapping_an_inside_corner_uses_affine_gradient_for_orientation() {
        let base = V3::new(-0.6294820717131473, -1.1762948207171315, -0.8328685258964144);
        let step = 0.01593625498007968;
        let corners = [
            Corner { id: 0, point: base, value: -1e-6 },
            Corner { id: 1, point: base + V3::new(step, 0.0, 0.0), value: 1.0 },
            Corner { id: 2, point: base + V3::new(0.0, step, 0.0), value: 1e-6 },
            Corner { id: 3, point: base + V3::new(0.0, 0.0, step), value: 1e-6 },
        ];
        let mut builder = Builder {
            vertices: Vec::new(),
            faces: Vec::new(),
            edges: HashMap::new(),
            face_records: HashMap::new(),
            endpoint_snap_tolerance: 1e-6,
            snapped_edge_roots: 0,
            maximum_endpoint_displacement: 0.0,
        };
        builder.tetrahedron(corners).unwrap();
        assert_eq!(builder.snapped_edge_roots, 1);
        let gradient = tetrahedron_gradient(corners).unwrap();
        let mesh = builder.finish().unwrap();
        assert_eq!(mesh.triangles.len(), 1);
        assert!(mesh.vertices.contains(&base));
        let [a, b, c] = mesh.triangles[0].map(|index| mesh.vertices[index as usize]);
        assert!((b - a).cross(c - a).dot(gradient) > 0.0);
    }

    #[test]
    fn invalid_grids_and_unpadded_boundaries_are_rejected() {
        let mut source = grid(5, 2.0, |p| length(p) - 1.0);
        source.values[0] = f64::NAN;
        assert!(extract(&source).is_err());
        source.values[0] = -1.0;
        assert!(extract(&source).is_err());
        source.values[0] = 1.0;
        source.spacing = f64::INFINITY;
        assert!(extract(&source).is_err());
        source.spacing = 1.0;
        source.dims = [usize::MAX; 3];
        assert!(extract(&source).is_err());
        source.dims = [1, 5, 5];
        assert!(extract(&source).is_err());
        source.dims = [5; 3];
        source.origin = V3::new(1e100, 0.0, 0.0);
        assert!(extract(&source).is_err());
    }

    #[test]
    fn audit_catches_orientation_open_edges_and_vertex_only_pinches() {
        let mut mesh = extract(&grid(15, 1.8, |p| length(p) - 1.0)).unwrap();
        mesh.triangles[0].swap(0, 1);
        assert!(mesh.audit().unwrap().orientation_conflicts > 0);
        assert!(mesh.validate().is_err());
        mesh.triangles[0].swap(0, 1);
        mesh.triangles.pop();
        assert!(mesh.audit().unwrap().boundary_edges > 0);
        assert!(mesh.validate().is_err());
        let a = extract(&grid(9, 2.0, |p| p.length_squared() - 1.0)).unwrap();
        let mut pinched = a.clone();
        let shared = a.vertices.len() as u32;
        pinched.vertices.extend(a.vertices.iter().skip(1).map(|p| *p + V3::new(4.0, 0.0, 0.0)));
        for triangle in a.triangles {
            pinched.triangles.push(triangle.map(|v| if v == 0 { 0 } else { shared + v - 1 }));
        }
        let audit = pinched.audit().unwrap();
        assert_eq!(audit.boundary_edges, 0);
        assert!(audit.nonmanifold_vertices > 0);
        assert!(pinched.validate().is_err());
    }
}
