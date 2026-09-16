//! Quasi-static linear elasticity on a remotely clamped rectangular sheet.
//!
//! The finite-element domain is larger than the visible optical window. Its
//! entire outer rim has zero displacement: the displayed crystal outline is an
//! aperture into this sheet, **not a traction-free material boundary**. Young's
//! modulus and sheet thickness are one; the constitutive law is isotropic plane
//! stress. Two constant-strain triangles per rectangular cell supply the usual
//! `area * B^T D B` stiffness. Boundary degrees of freedom are eliminated before
//! assembling a symmetric positive-definite sparse system.
//!
//! Three compact, smooth force footprints follow a fixed projection of the
//! unchanged source. Each body pair applies equal and opposite attractive loads.
//! Optional footprint elongation follows the projected three-dimensional tangent
//! without normalizing its projection; motion out of the sheet is circular.
//! Their bounded strength uses the original three-dimensional separation in the
//! source's single normalized coordinate system, never projected separation.
//! This is an explicit artistic coupling, not gravitational loading of a real
//! specimen. Short memory is a finite quadrature of past forces; by linearity it
//! is equivalent to averaging their elastic responses. There is no viscoelastic
//! time integration and no dependence on the order in which frames are rendered.
//! The memory kernel is `exp(-3u) * (1-u)^2` for age `u` in [0, 1] relative to
//! the full configured window. Its value and slope vanish at the old edge, so
//! startup truncation does not add an abrupt relaxation-rate change. Once the
//! full window is available its mean age is about 0.163425 of the window, before
//! multiplying by the configured memory mixture weight.
//!
//! Optical sampling uses area-weighted nodal stress recovery followed by a
//! cubic B-spline reconstruction. This deliberately smooths over about one grid
//! cell; the displayed interpolant is not claimed to satisfy local equilibrium
//! exactly. The finite-element residual is checked before reconstruction.

use crate::atelier::{OrbitSeries, SilkResult, SourceFrame, V3};
use serde::{Deserialize, Serialize};

/// Deterministic positive-definite preconditioner for the unchanged FEM system.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreconditionerKind {
    /// Inverse stiffness diagonal; retained as an independent reference path.
    #[default]
    Jacobi,
    /// Sparse zero-fill incomplete Cholesky, with guarded diagonal-shift retries.
    IncompleteCholesky,
}

/// Mechanical domain, fixed source projection, load coupling and solver controls.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FieldConfig {
    /// Half width and half height of the remote clamped domain, centered at zero.
    pub domain_half_size: [f64; 2],
    /// Node counts, including the clamped perimeter, in x and y.
    pub grid: [usize; 2],
    /// Fixed orthonormal source directions supplying sheet x and y coordinates.
    pub source_axes: [V3; 2],
    /// Single positive scale from projected normalized source to sheet coordinates.
    pub motion_scale: f64,
    /// Isotropic Poisson ratio; Young's modulus and sheet thickness are both one.
    pub poisson_ratio: f64,
    /// Minor-axis support radius of each compact C2 force footprint.
    /// At elongation one this is the circular radius, in sheet coordinates.
    pub load_softness: f64,
    /// Maximum footprint aspect along the projected source tangent; one is circular.
    /// The unnormalized 3D tangent projection continuously weakens elongation
    /// when motion points out of the sheet. Total applied force is unchanged.
    pub load_elongation: f64,
    /// Maximum attractive force magnitude of each body pair, before projection.
    pub load_strength: f64,
    /// Positive three-dimensional source separation at which force halves.
    pub encounter_scale: f64,
    /// Width of the recent source-time interval, as a fraction of the recording.
    pub memory_fraction: f64,
    /// Mixture weight of the recent forcing average, from zero to one.
    pub memory_weight: f64,
    /// Fixed Gauss-Legendre quadrature count over the available recent source interval.
    pub memory_samples: usize,
    /// Required true Euclidean residual norm divided by the load-vector norm.
    pub cg_tolerance: f64,
    /// Maximum number of preconditioned conjugate-gradient iterations.
    pub cg_max_iterations: usize,
    /// Reusable preconditioner; this never changes the physical stiffness or tolerance.
    pub preconditioner: PreconditionerKind,
}

impl Default for FieldConfig {
    fn default() -> Self {
        Self {
            domain_half_size: [4.4, 3.4],
            grid: [65, 49],
            source_axes: [V3::new(1.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0)],
            motion_scale: 0.9,
            poisson_ratio: 0.3,
            load_softness: 0.55,
            load_elongation: 1.0,
            load_strength: 0.6,
            encounter_scale: 0.8,
            memory_fraction: 0.02,
            memory_weight: 0.28,
            memory_samples: 12,
            cg_tolerance: 1e-9,
            cg_max_iterations: 2500,
            preconditioner: PreconditionerKind::IncompleteCholesky,
        }
    }
}

impl FieldConfig {
    /// Reject unresolved footprints, invalid mechanics and unbounded allocations.
    pub fn validate(&self) -> SilkResult<()> {
        if !self.domain_half_size.iter().all(|x| x.is_finite() && *x > 0.0)
            || !self.grid.iter().all(|n| (5..=257).contains(n))
            || !self.motion_scale.is_finite()
            || self.motion_scale <= 0.0
            || !self.poisson_ratio.is_finite()
            || !(-0.95..0.49).contains(&self.poisson_ratio)
            || !self.load_softness.is_finite()
            || self.load_softness <= 0.0
            || !self.load_elongation.is_finite()
            || !(1.0..=4.0).contains(&self.load_elongation)
            || !self.load_strength.is_finite()
            || self.load_strength < 0.0
            || !self.encounter_scale.is_finite()
            || self.encounter_scale <= 0.0
            || !self.memory_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.memory_fraction)
            || !self.memory_weight.is_finite()
            || !(0.0..=1.0).contains(&self.memory_weight)
            || !(1..=128).contains(&self.memory_samples)
            || !self.cg_tolerance.is_finite()
            || !(1e-13..=1e-3).contains(&self.cg_tolerance)
            || !(1..=20_000).contains(&self.cg_max_iterations)
        {
            return Err("invalid polarized-crystal elastic field configuration".into());
        }
        let [a, b] = self.source_axes;
        if !a.is_finite()
            || !b.is_finite()
            || (a.length_squared() - 1.0).abs() > 1e-8
            || (b.length_squared() - 1.0).abs() > 1e-8
            || a.dot(b).abs() > 1e-8
        {
            return Err("crystal field source axes must be fixed orthonormal vectors".into());
        }
        let spacing = self.spacing();
        if self.load_softness < 1.5 * spacing[0].max(spacing[1])
            || self.load_softness * self.load_elongation
                >= self.domain_half_size[0].min(self.domain_half_size[1])
        {
            return Err(
                "crystal load support must span at least 1.5 grid cells and fit inside the domain"
                    .into(),
            );
        }
        Ok(())
    }

    fn spacing(&self) -> [f64; 2] {
        std::array::from_fn(|axis| 2.0 * self.domain_half_size[axis] / (self.grid[axis] - 1) as f64)
    }
}

/// Measured mechanical solve quality for one independent time sample.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FieldDiagnostics {
    /// Actual preconditioner used, including any guarded fallback.
    #[serde(default)]
    pub preconditioner: PreconditionerKind,
    /// Relative diagonal shift used only in the incomplete factor, never in K.
    #[serde(default)]
    pub preconditioner_diagonal_shift: f64,
    /// True only when requested incomplete Cholesky exhausted its guarded retries.
    #[serde(default)]
    pub preconditioner_fallback: bool,
    /// Conjugate-gradient iterations used by this solve; zero for a zero load.
    pub cg_iterations: usize,
    /// True residual norm divided by force norm, recomputed from `K u - f`.
    pub relative_residual: f64,
    /// Absolute Euclidean norm of the true unconstrained residual.
    pub residual_norm: f64,
    /// Euclidean norm of the assembled unconstrained load vector.
    pub force_norm: f64,
    /// Signed sum of all applied nodal force components, before support reactions.
    pub net_force: [f64; 2],
    /// Magnitude of net force divided by summed nodal force magnitudes.
    pub relative_force_imbalance: f64,
    /// Applied nodal moment about the domain center, including discretization error.
    pub applied_torque: f64,
    /// Elastic strain energy, `0.5 * u^T K u`, at unit sheet thickness.
    pub strain_energy: f64,
    /// Maximum displacement magnitude at a finite-element node.
    pub max_displacement: f64,
    /// Largest absolute recovered stress component before optical interpolation.
    pub max_stress_component: f64,
    /// Largest recovered principal-stress difference, useful for optical scaling.
    pub max_stress_difference: f64,
    /// Actual number of forcing evaluations, including the present when weighted.
    pub forcing_samples: usize,
    /// Actual oldest sampled interval endpoint; no unavailable history is invented.
    pub history_start: f64,
}

impl FieldDiagnostics {
    /// Validate finite, mutually consistent receipt diagnostics without solving.
    pub fn validate(&self) -> SilkResult<()> {
        let nonnegative = [
            self.relative_residual,
            self.residual_norm,
            self.force_norm,
            self.relative_force_imbalance,
            self.strain_energy,
            self.max_displacement,
            self.max_stress_component,
            self.max_stress_difference,
        ];
        if !nonnegative.iter().all(|value| value.is_finite() && *value >= 0.0)
            || !self.net_force.iter().all(|value| value.is_finite())
            || !self.applied_torque.is_finite()
            || !self.history_start.is_finite()
            || !(0.0..=1.0).contains(&self.history_start)
            || !(1..=129).contains(&self.forcing_samples)
            || self.cg_iterations > 20_000
            || self.relative_residual > 1e-3
            || self.relative_force_imbalance > 1e-10
        {
            return Err("invalid crystal field numerical diagnostics".into());
        }
        if !IC_DIAGONAL_SHIFTS.contains(&self.preconditioner_diagonal_shift)
            || (self.preconditioner == PreconditionerKind::Jacobi
                && self.preconditioner_diagonal_shift != 0.0)
            || (self.preconditioner == PreconditionerKind::IncompleteCholesky
                && self.preconditioner_fallback)
        {
            return Err("invalid crystal preconditioner diagnostics".into());
        }
        if self.force_norm == 0.0 {
            if self.cg_iterations != 0
                || self.residual_norm != 0.0
                || self.relative_residual != 0.0
                || self.strain_energy != 0.0
                || self.max_displacement != 0.0
                || self.max_stress_component != 0.0
                || self.max_stress_difference != 0.0
                || self.net_force != [0.0; 2]
                || self.applied_torque != 0.0
                || self.relative_force_imbalance != 0.0
            {
                return Err("zero-load crystal diagnostics must describe a zero response".into());
            }
        } else if self.cg_iterations == 0
            || self.strain_energy <= 0.0
            || self.max_displacement <= 0.0
            || (self.relative_residual - self.residual_norm / self.force_norm).abs() > 1e-12
        {
            return Err("inconsistent crystal field solve diagnostics".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Element {
    nodes: [usize; 3],
    /// Shape-function gradients in world x/y, one per triangle vertex.
    gradients: [[f64; 2]; 3],
    area: f64,
}

impl Element {
    fn new(nodes: [usize; 3], positions: &[[f64; 2]]) -> Self {
        let [a, b, c] = nodes.map(|node| positions[node]);
        let twice_area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
        debug_assert!(twice_area > 0.0);
        Self {
            nodes,
            gradients: [
                [(b[1] - c[1]) / twice_area, (c[0] - b[0]) / twice_area],
                [(c[1] - a[1]) / twice_area, (a[0] - c[0]) / twice_area],
                [(a[1] - b[1]) / twice_area, (b[0] - a[0]) / twice_area],
            ],
            area: twice_area * 0.5,
        }
    }

    /// A column of B, whose third row is engineering shear strain `gamma_xy`.
    fn strain_column(&self, local_dof: usize) -> [f64; 3] {
        let [dx, dy] = self.gradients[local_dof / 2];
        if local_dof.is_multiple_of(2) { [dx, 0.0, dy] } else { [0.0, dy, dx] }
    }

    fn stress(&self, displacement: &[[f64; 2]], poisson: f64) -> [f64; 3] {
        let mut strain = [0.0; 3];
        for local in 0..6 {
            let column = self.strain_column(local);
            let value = displacement[self.nodes[local / 2]][local % 2];
            for axis in 0..3 {
                strain[axis] += column[axis] * value;
            }
        }
        constitutive(strain, poisson)
    }
}

/// Plane-stress D times [`epsilon_xx`, `epsilon_yy`, engineering `gamma_xy`].
fn constitutive(strain: [f64; 3], poisson: f64) -> [f64; 3] {
    let factor = 1.0 / (1.0 - poisson * poisson);
    [
        factor * (strain[0] + poisson * strain[1]),
        factor * (poisson * strain[0] + strain[1]),
        strain[2] / (2.0 * (1.0 + poisson)),
    ]
}

#[derive(Debug)]
struct SparseMatrix {
    offsets: Vec<usize>,
    columns: Vec<usize>,
    values: Vec<f64>,
    inverse_diagonal: Vec<f64>,
}

impl SparseMatrix {
    fn from_rows(mut rows: Vec<Vec<(usize, f64)>>) -> SilkResult<Self> {
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        let mut columns = Vec::new();
        let mut values = Vec::new();
        let mut inverse_diagonal = Vec::with_capacity(rows.len());
        for (index, row) in rows.iter_mut().enumerate() {
            row.sort_by_key(|entry| entry.0);
            offsets.push(values.len());
            let mut diagonal = 0.0;
            let mut cursor = 0;
            while cursor < row.len() {
                let column = row[cursor].0;
                let mut value = 0.0;
                while cursor < row.len() && row[cursor].0 == column {
                    value += row[cursor].1;
                    cursor += 1;
                }
                if value != 0.0 {
                    columns.push(column);
                    values.push(value);
                }
                if column == index {
                    diagonal = value;
                }
            }
            if !diagonal.is_finite() || diagonal <= 0.0 {
                return Err("crystal FEM stiffness has an invalid diagonal".into());
            }
            inverse_diagonal.push(1.0 / diagonal);
        }
        offsets.push(values.len());
        Ok(Self { offsets, columns, values, inverse_diagonal })
    }

    fn multiply(&self, vector: &[f64], result: &mut [f64]) {
        for (row, target) in result.iter_mut().enumerate() {
            *target = (self.offsets[row]..self.offsets[row + 1])
                .map(|entry| self.values[entry] * vector[self.columns[entry]])
                .sum();
        }
    }
}

/// Only the preconditioner's diagonal is shifted; the CG operator stays exact.
const IC_DIAGONAL_SHIFTS: [f64; 7] = [0.0, 1e-8, 1e-6, 1e-4, 0.01, 0.1, 1.0];
const IC_MIN_RELATIVE_PIVOT: f64 = 1e-12;

/// Strictly lower CSR plus the inverse positive Cholesky diagonal. No fill outside
/// the original lower sparsity pattern is introduced; allocation remains O(nnz).
#[derive(Debug)]
struct IncompleteCholesky {
    offsets: Vec<usize>,
    columns: Vec<usize>,
    values: Vec<f64>,
    inverse_diagonal: Vec<f64>,
    shift: f64,
}

impl IncompleteCholesky {
    fn factor(matrix: &SparseMatrix, shift: f64) -> Option<Self> {
        let count = matrix.inverse_diagonal.len();
        let mut offsets = Vec::with_capacity(count + 1);
        let mut columns = Vec::with_capacity(matrix.columns.len() / 2);
        let mut values = Vec::with_capacity(matrix.values.len() / 2);
        let mut diagonal = vec![0.0; count];
        for (row, diagonal_value) in diagonal.iter_mut().enumerate() {
            offsets.push(values.len());
            for entry in matrix.offsets[row]..matrix.offsets[row + 1] {
                let column = matrix.columns[entry];
                if column < row {
                    columns.push(column);
                    values.push(matrix.values[entry]);
                } else if column == row {
                    *diagonal_value = matrix.values[entry];
                }
            }
        }
        offsets.push(values.len());
        for row in 0..count {
            let original_diagonal = diagonal[row];
            for entry in offsets[row]..offsets[row + 1] {
                let column = columns[entry];
                let mut value = values[entry];
                let mut left = offsets[row];
                let mut right = offsets[column];
                // Intersect sorted lower rows to obtain sum_k L_ik L_jk.
                // Previously computed entries of the current row are the only
                // dependencies; all of the earlier row is already factored.
                while left < entry && right < offsets[column + 1] {
                    match columns[left].cmp(&columns[right]) {
                        std::cmp::Ordering::Less => left += 1,
                        std::cmp::Ordering::Greater => right += 1,
                        std::cmp::Ordering::Equal => {
                            value -= values[left] * values[right];
                            left += 1;
                            right += 1;
                        }
                    }
                }
                value /= diagonal[column];
                if !value.is_finite() {
                    return None;
                }
                values[entry] = value;
            }
            let squared: f64 =
                values[offsets[row]..offsets[row + 1]].iter().map(|value| value * value).sum();
            let pivot = original_diagonal * (1.0 + shift) - squared;
            if !pivot.is_finite() || pivot <= original_diagonal * IC_MIN_RELATIVE_PIVOT {
                return None;
            }
            diagonal[row] = pivot.sqrt();
        }
        let inverse_diagonal = diagonal.into_iter().map(f64::recip).collect();
        Some(Self { offsets, columns, values, inverse_diagonal, shift })
    }

    /// Apply (L L^T)^-1 through sparse forward and backward substitution.
    /// The reverse scatter avoids a second transposed sparse allocation.
    fn apply(&self, rhs: &[f64], output: &mut [f64]) {
        for row in 0..rhs.len() {
            let mut value = rhs[row];
            for entry in self.offsets[row]..self.offsets[row + 1] {
                value -= self.values[entry] * output[self.columns[entry]];
            }
            output[row] = value * self.inverse_diagonal[row];
        }
        for row in (0..rhs.len()).rev() {
            let value = output[row] * self.inverse_diagonal[row];
            output[row] = value;
            for entry in self.offsets[row]..self.offsets[row + 1] {
                output[self.columns[entry]] -= self.values[entry] * value;
            }
        }
    }
}

#[derive(Debug)]
struct Preconditioner {
    factor: Option<IncompleteCholesky>,
    fallback: bool,
}

impl Preconditioner {
    fn new(matrix: &SparseMatrix, requested: PreconditionerKind) -> Self {
        if requested == PreconditionerKind::Jacobi {
            return Self { factor: None, fallback: false };
        }
        for shift in IC_DIAGONAL_SHIFTS {
            if let Some(factor) = IncompleteCholesky::factor(matrix, shift) {
                return Self { factor: Some(factor), fallback: false };
            }
        }
        Self { factor: None, fallback: true }
    }

    fn apply(&self, matrix: &SparseMatrix, rhs: &[f64], output: &mut [f64]) {
        if let Some(factor) = &self.factor {
            factor.apply(rhs, output);
        } else {
            for ((value, force), inverse_diagonal) in
                output.iter_mut().zip(rhs).zip(&matrix.inverse_diagonal)
            {
                *value = force * inverse_diagonal;
            }
        }
    }

    fn write_diagnostics(&self, diagnostics: &mut FieldDiagnostics) {
        diagnostics.preconditioner = if self.factor.is_some() {
            PreconditionerKind::IncompleteCholesky
        } else {
            PreconditionerKind::Jacobi
        };
        diagnostics.preconditioner_diagonal_shift =
            self.factor.as_ref().map_or(0.0, |factor| factor.shift);
        diagnostics.preconditioner_fallback = self.fallback;
    }
}

/// Prepared linear sheet stiffness, reusable across arbitrary source times.
#[derive(Debug)]
pub struct ElasticSheet {
    config: FieldConfig,
    nodes: Vec<[f64; 2]>,
    nodal_area: Vec<f64>,
    /// First free displacement degree of freedom, or None for the clamped rim.
    free_dof: Vec<Option<usize>>,
    elements: Vec<Element>,
    stiffness: SparseMatrix,
    preconditioner: Preconditioner,
}

impl ElasticSheet {
    /// Assemble the homogeneous, unit-thickness plane-stress sheet once.
    pub fn new(config: &FieldConfig) -> SilkResult<Self> {
        config.validate()?;
        let [nx, ny] = config.grid;
        let spacing = config.spacing();
        let mut nodes = Vec::with_capacity(nx * ny);
        let mut free_dof = Vec::with_capacity(nx * ny);
        let mut dofs = 0;
        for y in 0..ny {
            for x in 0..nx {
                nodes.push([
                    -config.domain_half_size[0] + x as f64 * spacing[0],
                    -config.domain_half_size[1] + y as f64 * spacing[1],
                ]);
                if x == 0 || y == 0 || x + 1 == nx || y + 1 == ny {
                    free_dof.push(None);
                } else {
                    free_dof.push(Some(dofs));
                    dofs += 2;
                }
            }
        }
        let mut elements = Vec::with_capacity(2 * (nx - 1) * (ny - 1));
        for y in 0..ny - 1 {
            for x in 0..nx - 1 {
                let a = y * nx + x;
                let b = a + 1;
                let c = a + nx;
                let d = c + 1;
                // Alternating diagonals avoid privileging one shear direction.
                let triangles =
                    if (x + y) % 2 == 0 { [[a, b, d], [a, d, c]] } else { [[a, b, c], [b, d, c]] };
                elements.extend(triangles.map(|triangle| Element::new(triangle, &nodes)));
            }
        }
        let mut rows = vec![Vec::with_capacity(24); dofs];
        let mut nodal_area = vec![0.0; nodes.len()];
        for element in &elements {
            for &node in &element.nodes {
                nodal_area[node] += element.area / 3.0;
            }
            for local_row in 0..6 {
                let Some(row_base) = free_dof[element.nodes[local_row / 2]] else { continue };
                let strain_row = element.strain_column(local_row);
                for local_column in 0..6 {
                    let Some(column_base) = free_dof[element.nodes[local_column / 2]] else {
                        continue;
                    };
                    let stress_column =
                        constitutive(element.strain_column(local_column), config.poisson_ratio);
                    let value = element.area * dot3(strain_row, stress_column);
                    rows[row_base + local_row % 2].push((column_base + local_column % 2, value));
                }
            }
        }
        let stiffness = SparseMatrix::from_rows(rows)?;
        let preconditioner = Preconditioner::new(&stiffness, config.preconditioner);
        Ok(Self {
            config: config.clone(),
            nodes,
            nodal_area,
            free_dof,
            elements,
            stiffness,
            preconditioner,
        })
    }

    /// Solve an independent source time with a deterministic recent-force memory.
    pub fn solve(&self, source: &OrbitSeries, time: f64) -> SilkResult<StressField> {
        if !(0.0..=1.0).contains(&time) || source.sample(time).is_none() {
            return Err("crystal field time is outside the verified source interval".into());
        }
        let mut forces = vec![[0.0; 2]; self.nodes.len()];
        let schedule = forcing_schedule(&self.config, 0.0, time);
        for &(sample_time, weight) in &schedule {
            let frame =
                source.sample(sample_time).ok_or("crystal memory requested unavailable history")?;
            self.add_frame_forces(&frame, weight, &mut forces)?;
        }
        let mut diagnostics = force_diagnostics(&self.nodes, &forces);
        diagnostics.forcing_samples = schedule.len();
        diagnostics.history_start = if self.config.memory_weight == 0.0 {
            time
        } else {
            (time - self.config.memory_fraction).max(0.0)
        };
        let rhs = self.free_vector(&forces);
        let displacement = self.solve_displacements(&rhs, &mut diagnostics)?;
        let recovered = self.recover_stress(&displacement);
        diagnostics.max_displacement =
            displacement.iter().map(|v| v[0].hypot(v[1])).fold(0.0, f64::max);
        diagnostics.max_stress_component =
            recovered.iter().flatten().map(|v| v.abs()).fold(0.0, f64::max);
        diagnostics.max_stress_difference = recovered
            .iter()
            .map(|stress| (stress[0] - stress[1]).hypot(2.0 * stress[2]))
            .fold(0.0, f64::max);
        diagnostics.validate()?;
        Ok(StressField {
            domain_half_size: self.config.domain_half_size,
            grid: self.config.grid,
            stress: recovered,
            diagnostics,
        })
    }

    fn add_frame_forces(
        &self,
        frame: &SourceFrame,
        weight: f64,
        forces: &mut [[f64; 2]],
    ) -> SilkResult<()> {
        if self.config.load_strength == 0.0 || weight == 0.0 {
            return Ok(());
        }
        let centers: [[f64; 2]; 3] = frame.bodies.map(|body| {
            self.config.source_axes.map(|axis| body.position.dot(axis) * self.config.motion_scale)
        });
        let footprints: Vec<Vec<(usize, f64)>> = centers
            .iter()
            .zip(&frame.bodies)
            .map(|(center, body)| {
                let tangent = self.config.source_axes.map(|axis| body.tangent.dot(axis));
                self.footprint(*center, tangent)
            })
            .collect::<SilkResult<_>>()?;
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            let delta = [centers[b][0] - centers[a][0], centers[b][1] - centers[a][1]];
            let separation = (frame.bodies[b].position - frame.bodies[a].position).length();
            let ratio = separation / self.config.encounter_scale;
            let strength = weight * self.config.load_strength / (1.0 + ratio * ratio);
            // Regularize the direction so a pair coinciding only in projection
            // fades continuously to zero rather than flipping a unit vector.
            let direction_length = delta[0].hypot(delta[1]).hypot(0.1 * self.config.load_softness);
            let force = delta.map(|component| strength * component / direction_length);
            for &(node, portion) in &footprints[a] {
                for axis in 0..2 {
                    forces[node][axis] += force[axis] * portion;
                }
            }
            for &(node, portion) in &footprints[b] {
                for axis in 0..2 {
                    forces[node][axis] -= force[axis] * portion;
                }
            }
        }
        Ok(())
    }

    fn footprint(&self, center: [f64; 2], tangent: [f64; 2]) -> SilkResult<Vec<(usize, f64)>> {
        let radius = self.config.load_softness;
        let support = radius * self.config.load_elongation;
        if (0..2).any(|axis| {
            !center[axis].is_finite()
                || center[axis].abs() + support >= self.config.domain_half_size[axis]
        }) {
            return Err("crystal force footprint reaches the clamped rim; enlarge the domain or reduce motion_scale".into());
        }
        let [nx, ny] = self.config.grid;
        let spacing = self.config.spacing();
        let lower: [usize; 2] = std::array::from_fn(|axis| {
            ((center[axis] - support + self.config.domain_half_size[axis]) / spacing[axis])
                .floor()
                .max(1.0) as usize
        });
        let upper: [usize; 2] = std::array::from_fn(|axis| {
            (((center[axis] + support + self.config.domain_half_size[axis]) / spacing[axis]).ceil()
                as usize)
                .min(self.config.grid[axis] - 2)
        });
        let mut result = Vec::new();
        let mut total = 0.0;
        for y in lower[1]..=upper[1].min(ny - 2) {
            for x in lower[0]..=upper[0].min(nx - 2) {
                let node = y * nx + x;
                let offset = [self.nodes[node][0] - center[0], self.nodes[node][1] - center[1]];
                let q = footprint_radius(offset, tangent, radius, self.config.load_elongation);
                if q < 1.0 {
                    let value = (1.0 - q).powi(4) * (1.0 + 4.0 * q) * self.nodal_area[node];
                    result.push((node, value));
                    total += value;
                }
            }
        }
        if total <= 0.0 || !total.is_finite() {
            return Err("crystal force footprint contains no resolved free nodes".into());
        }
        for (_, value) in &mut result {
            *value /= total;
        }
        Ok(result)
    }

    fn free_vector(&self, vectors: &[[f64; 2]]) -> Vec<f64> {
        let mut result = vec![0.0; self.stiffness.inverse_diagonal.len()];
        for (node, free) in self.free_dof.iter().enumerate() {
            if let Some(index) = *free {
                result[index] = vectors[node][0];
                result[index + 1] = vectors[node][1];
            }
        }
        result
    }

    fn solve_displacements(
        &self,
        rhs: &[f64],
        diagnostics: &mut FieldDiagnostics,
    ) -> SilkResult<Vec<[f64; 2]>> {
        self.preconditioner.write_diagnostics(diagnostics);
        let count = rhs.len();
        let force_norm = dot(rhs, rhs).sqrt();
        diagnostics.force_norm = force_norm;
        if !force_norm.is_finite() {
            return Err("crystal force vector is not finite".into());
        }
        if force_norm == 0.0 {
            if rhs.iter().any(|component| *component != 0.0) {
                return Err("crystal force norm underflowed for a nonzero load vector".into());
            }
            return Ok(vec![[0.0; 2]; self.nodes.len()]);
        }
        let target = self.config.cg_tolerance * force_norm;
        let mut solution = vec![0.0; count];
        let mut residual = rhs.to_vec();
        let mut preconditioned = vec![0.0; count];
        self.preconditioner.apply(&self.stiffness, &residual, &mut preconditioned);
        let mut direction = preconditioned.clone();
        let mut product = vec![0.0; count];
        let mut rho = dot(&residual, &preconditioned);
        let mut converged = false;
        for iteration in 1..=self.config.cg_max_iterations {
            self.stiffness.multiply(&direction, &mut product);
            let denominator = dot(&direction, &product);
            if !denominator.is_finite() || denominator <= 0.0 || !rho.is_finite() || rho <= 0.0 {
                return Err("crystal conjugate-gradient solver lost positive definiteness".into());
            }
            let alpha = rho / denominator;
            for i in 0..count {
                solution[i] += alpha * direction[i];
                residual[i] -= alpha * product[i];
            }
            diagnostics.cg_iterations = iteration;
            if dot(&residual, &residual).sqrt() <= target {
                // Recurrence residuals can drift: certify using the original K.
                self.stiffness.multiply(&solution, &mut product);
                for i in 0..count {
                    residual[i] = rhs[i] - product[i];
                }
                if dot(&residual, &residual).sqrt() <= target {
                    converged = true;
                    break;
                }
                // A reliable restart keeps an inaccurate recurrence from
                // reporting success and remains independent of frame order.
                self.preconditioner.apply(&self.stiffness, &residual, &mut preconditioned);
                direction.copy_from_slice(&preconditioned);
                rho = dot(&residual, &preconditioned);
                continue;
            }
            self.preconditioner.apply(&self.stiffness, &residual, &mut preconditioned);
            let next_rho = dot(&residual, &preconditioned);
            let beta = next_rho / rho;
            for i in 0..count {
                direction[i] = preconditioned[i] + beta * direction[i];
            }
            rho = next_rho;
        }
        self.stiffness.multiply(&solution, &mut product);
        diagnostics.residual_norm =
            rhs.iter().zip(&product).map(|(f, ku)| (f - ku).powi(2)).sum::<f64>().sqrt();
        diagnostics.relative_residual = diagnostics.residual_norm / force_norm;
        diagnostics.strain_energy = 0.5 * dot(&solution, &product);
        if !converged
            || !diagnostics.relative_residual.is_finite()
            || diagnostics.relative_residual > self.config.cg_tolerance
        {
            return Err(format!("crystal FEM failed to converge in {} iterations (relative residual {:.3e}, tolerance {:.3e})", diagnostics.cg_iterations, diagnostics.relative_residual, self.config.cg_tolerance).into());
        }
        let mut displacement = vec![[0.0; 2]; self.nodes.len()];
        for (node, free) in self.free_dof.iter().enumerate() {
            if let Some(index) = *free {
                displacement[node] = [solution[index], solution[index + 1]];
            }
        }
        Ok(displacement)
    }

    fn recover_stress(&self, displacement: &[[f64; 2]]) -> Vec<[f64; 3]> {
        let mut recovered = vec![[0.0; 3]; self.nodes.len()];
        for element in &self.elements {
            let stress = element.stress(displacement, self.config.poisson_ratio);
            for &node in &element.nodes {
                for component in 0..3 {
                    recovered[node][component] += element.area * stress[component] / 3.0;
                }
            }
        }
        for (stress, area) in recovered.iter_mut().zip(&self.nodal_area) {
            for component in stress {
                *component /= area;
            }
        }
        recovered
    }
}

fn footprint_radius(offset: [f64; 2], tangent: [f64; 2], radius: f64, elongation: f64) -> f64 {
    if elongation == 1.0 || tangent == [0.0; 2] {
        // Preserve the original circular arithmetic exactly at aspect one.
        return offset[0].hypot(offset[1]) / radius;
    }
    let along = offset[0] * tangent[0] + offset[1] * tangent[1];
    let anisotropy = 1.0 - 1.0 / (elongation * elongation);
    (offset[0] * offset[0] + offset[1] * offset[1] - anisotropy * along * along).max(0.0).sqrt()
        / radius
}

/// Recovered stress tensor and solve diagnostics for a fixed source time.
#[derive(Debug)]
pub struct StressField {
    domain_half_size: [f64; 2],
    grid: [usize; 2],
    stress: Vec<[f64; 3]>,
    /// Verified solve quality and actual applied-force balance.
    pub diagnostics: FieldDiagnostics,
}

impl StressField {
    /// Smoothly sample [`sigma_xx`, `sigma_yy`, `sigma_xy`] in sheet coordinates.
    ///
    /// Samples outside the mechanical domain return zero. The visible optical
    /// aperture must stay strictly inside the domain, away from this cutoff.
    pub fn sample(&self, x: f64, y: f64) -> [f64; 3] {
        if !x.is_finite()
            || !y.is_finite()
            || x.abs() > self.domain_half_size[0]
            || y.abs() > self.domain_half_size[1]
        {
            return [0.0; 3];
        }
        let coordinates = [x, y];
        let grid_coordinates: [f64; 2] = std::array::from_fn(|axis| {
            (coordinates[axis] + self.domain_half_size[axis]) * (self.grid[axis] - 1) as f64
                / (2.0 * self.domain_half_size[axis])
        });
        let base = grid_coordinates.map(|coordinate| coordinate.floor() as isize);
        let weights = std::array::from_fn::<_, 2, _>(|axis| {
            cubic_weights(grid_coordinates[axis] - base[axis] as f64)
        });
        let mut result = [0.0; 3];
        for j in 0..4 {
            let iy = (base[1] + j as isize - 1).clamp(0, self.grid[1] as isize - 1) as usize;
            for i in 0..4 {
                let ix = (base[0] + i as isize - 1).clamp(0, self.grid[0] as isize - 1) as usize;
                let weight = weights[0][i] * weights[1][j];
                let stress = self.stress[iy * self.grid[0] + ix];
                for component in 0..3 {
                    result[component] += weight * stress[component];
                }
            }
        }
        result
    }
}

fn cubic_weights(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        (1.0 - t).powi(3) / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
        t3 / 6.0,
    ]
}

fn forcing_schedule(config: &FieldConfig, history_start: f64, time: f64) -> Vec<(f64, f64)> {
    let available = config.memory_fraction.min((time - history_start).max(0.0));
    if available <= 0.0 || config.memory_weight == 0.0 {
        return vec![(time, 1.0)];
    }
    let mut result = Vec::with_capacity(config.memory_samples + 1);
    if config.memory_weight < 1.0 {
        result.push((time, 1.0 - config.memory_weight));
    }
    let mut total = 0.0;
    let mut past = Vec::with_capacity(config.memory_samples);
    for (node, quadrature_weight) in temporal_quadrature(config.memory_samples) {
        let age = available * node;
        // Age uses the full window, not the truncated startup interval. This
        // compact kernel and its first derivative vanish at the oldest edge,
        // so reaching the full window does not introduce a relaxation kink.
        // The common interval-width factor cancels in the normalization below.
        let normalized_age = age / config.memory_fraction;
        let weight =
            quadrature_weight * (-3.0 * normalized_age).exp() * (1.0 - normalized_age).powi(2);
        past.push((time - age, weight));
        total += weight;
    }
    result.extend(
        past.into_iter().map(|(sample, weight)| (sample, config.memory_weight * weight / total)),
    );
    result
}

/// Deterministic Gauss-Legendre nodes and weights on [0, 1]. High-order fixed
/// quadrature resolves the smooth startup kernel without midpoint-grid kinks.
/// Root iteration is bounded independently of render order or the source.
fn temporal_quadrature(count: usize) -> Vec<(f64, f64)> {
    let mut quadrature = vec![(0.0, 0.0); count];
    for index in 0..count.div_ceil(2) {
        let mut root = (std::f64::consts::PI * (index as f64 + 0.75) / (count as f64 + 0.5)).cos();
        for _ in 0..32 {
            let (value, derivative) = legendre_value_derivative(count, root);
            let next = root - value / derivative;
            let change = (next - root).abs();
            root = next;
            if change < 2e-15 {
                break;
            }
        }
        let (_, derivative) = legendre_value_derivative(count, root);
        let weight = 1.0 / ((1.0 - root * root) * derivative * derivative);
        quadrature[index] = ((1.0 - root) * 0.5, weight);
        quadrature[count - index - 1] = ((1.0 + root) * 0.5, weight);
    }
    quadrature
}

fn legendre_value_derivative(degree: usize, x: f64) -> (f64, f64) {
    let mut previous = 1.0;
    let mut current = x;
    for order in 2..=degree {
        let next =
            ((2 * order - 1) as f64 * x * current - (order - 1) as f64 * previous) / order as f64;
        previous = current;
        current = next;
    }
    (current, degree as f64 * (x * current - previous) / (x * x - 1.0))
}

fn force_diagnostics(nodes: &[[f64; 2]], forces: &[[f64; 2]]) -> FieldDiagnostics {
    let mut result = FieldDiagnostics::default();
    let mut magnitude = 0.0;
    for (point, force) in nodes.iter().zip(forces) {
        for (total, component) in result.net_force.iter_mut().zip(force) {
            *total += component;
        }
        result.applied_torque += point[0] * force[1] - point[1] * force[0];
        magnitude += force[0].hypot(force[1]);
    }
    result.relative_force_imbalance = if magnitude > 0.0 {
        result.net_force[0].hypot(result.net_force[1]) / magnitude
    } else {
        0.0
    };
    result
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silk::OrbitData;
    use std::f64::consts::PI;

    fn source() -> OrbitSeries {
        let samples = (0..65)
            .map(|index| {
                let t = f64::from(index) / 64.0;
                [
                    V3::new(-1.0 + 0.2 * t, -0.3 + 0.25 * t, 0.2),
                    V3::new(0.8 - 0.2 * t, -0.4, -0.1 + 0.1 * t),
                    V3::new(0.0, 0.7 - 0.1 * t, 0.3),
                ]
            })
            .collect();
        OrbitSeries::new(&OrbitData {
            seed: "0xc1".into(),
            dt: 0.01,
            masses: [1.0; 3],
            samples,
            provenance: serde_json::json!({}),
        })
        .unwrap()
    }

    fn config() -> FieldConfig {
        FieldConfig { grid: [33, 25], load_softness: 0.65, ..FieldConfig::default() }
    }

    #[test]
    fn zero_force_has_zero_stress_displacement_and_iterations() {
        let sheet = ElasticSheet::new(&FieldConfig { load_strength: 0.0, ..config() }).unwrap();
        let field = sheet.solve(&source(), 0.4).unwrap();
        assert_eq!(field.diagnostics.cg_iterations, 0);
        assert_eq!(field.diagnostics.max_displacement, 0.0);
        assert_eq!(field.diagnostics.strain_energy, 0.0);
        assert!(field.stress.iter().all(|stress| *stress == [0.0; 3]));
        assert_eq!(field.sample(0.31, -0.72), [0.0; 3]);
    }

    #[test]
    fn tiny_nonzero_force_is_rejected_when_its_norm_underflows() {
        let cfg = FieldConfig { load_strength: 1e-180, ..config() };
        cfg.validate().unwrap();
        let sheet = ElasticSheet::new(&cfg).unwrap();
        let error = sheet.solve(&source(), 0.4).unwrap_err().to_string();
        assert!(error.contains("underflow"), "{error}");
    }

    #[test]
    fn zero_force_diagnostics_reject_residual_applied_load_or_moment() {
        let baseline = FieldDiagnostics { forcing_samples: 1, ..FieldDiagnostics::default() };
        baseline.validate().unwrap();
        let mut invalid = baseline.clone();
        invalid.net_force = [1.0, 0.0];
        assert!(invalid.validate().is_err());
        let mut invalid = baseline.clone();
        invalid.applied_torque = 1.0;
        assert!(invalid.validate().is_err());
        let mut invalid = baseline;
        invalid.relative_force_imbalance = 1e-14;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn aspect_one_preserves_the_original_circle_exactly() {
        let sheet = ElasticSheet::new(&config()).unwrap();
        let reference = sheet.footprint([0.31, -0.13], [0.0; 2]).unwrap();
        assert_eq!(reference, sheet.footprint([0.31, -0.13], [1.0, 0.0]).unwrap());
        assert_eq!(reference, sheet.footprint([0.31, -0.13], [-0.2, 0.3]).unwrap());
        let offset = [0.23_f64, -0.16_f64];
        assert_eq!(
            footprint_radius(offset, [0.6, 0.8], 0.55, 1.0).to_bits(),
            (offset[0].hypot(offset[1]) / 0.55).to_bits()
        );
    }

    #[test]
    fn elongated_footprints_have_no_projected_tangent_singularity() {
        let offset = [0.23_f64, -0.16_f64];
        let circle = offset[0].hypot(offset[1]) / 0.55;
        assert_eq!(footprint_radius(offset, [0.0; 2], 0.55, 4.0), circle);
        assert!((footprint_radius(offset, [1e-9, 0.0], 0.55, 4.0) - circle).abs() < 1e-15);
        assert_eq!(
            footprint_radius(offset, [0.6, 0.8], 0.55, 3.0),
            footprint_radius(offset, [-0.6, -0.8], 0.55, 3.0)
        );
        assert!(
            footprint_radius([0.5, 0.0], [1.0, 0.0], 0.55, 3.0)
                < footprint_radius([0.5, 0.0], [0.5, 0.0], 0.55, 3.0)
        );
        let cfg = FieldConfig { load_elongation: 2.0, ..config() };
        let field = ElasticSheet::new(&cfg).unwrap().solve(&source(), 0.47).unwrap();
        assert!(field.diagnostics.relative_force_imbalance < 1e-13);
        assert!(field.diagnostics.relative_residual <= cfg.cg_tolerance);
    }

    #[test]
    fn affine_patch_has_exact_constant_stress_and_interior_equilibrium() {
        let sheet = ElasticSheet::new(&config()).unwrap();
        let displacement: Vec<[f64; 2]> = sheet
            .nodes
            .iter()
            .map(|&[x, y]| [0.02 * x + 0.03 * y + 0.7, -0.01 * x + 0.04 * y - 0.3])
            .collect();
        let expected = constitutive([0.02, 0.04, 0.02], sheet.config.poisson_ratio);
        let mut internal = vec![[0.0; 2]; sheet.nodes.len()];
        for element in &sheet.elements {
            let stress = element.stress(&displacement, sheet.config.poisson_ratio);
            for component in 0..3 {
                assert!((stress[component] - expected[component]).abs() < 2e-15);
            }
            for local in 0..6 {
                internal[element.nodes[local / 2]][local % 2] +=
                    element.area * dot3(element.strain_column(local), stress);
            }
        }
        assert!(sheet.free_vector(&internal).iter().all(|force| force.abs() < 2e-15));
    }

    #[test]
    fn assembled_stiffness_is_symmetric_and_has_positive_energy() {
        let sheet = ElasticSheet::new(&config()).unwrap();
        let size = sheet.stiffness.inverse_diagonal.len();
        let a: Vec<f64> = (0..size).map(|i| (i as f64 * 0.73).sin()).collect();
        let b: Vec<f64> = (0..size).map(|i| (i as f64 * 0.27).cos()).collect();
        let mut ka = vec![0.0; size];
        let mut kb = vec![0.0; size];
        sheet.stiffness.multiply(&a, &mut ka);
        sheet.stiffness.multiply(&b, &mut kb);
        assert!(dot(&a, &ka) > 0.0);
        assert!(dot(&b, &kb) > 0.0);
        assert!((dot(&a, &kb) - dot(&b, &ka)).abs() < 1e-10);
    }

    #[test]
    fn incomplete_cholesky_action_is_symmetric_and_positive() {
        let sheet = ElasticSheet::new(&config()).unwrap();
        assert!(sheet.preconditioner.factor.is_some());
        let count = sheet.stiffness.inverse_diagonal.len();
        let a: Vec<f64> = (0..count).map(|i| (i as f64 * 0.73).sin()).collect();
        let b: Vec<f64> = (0..count).map(|i| (i as f64 * 0.27).cos()).collect();
        let mut pa = vec![0.0; count];
        let mut pb = vec![0.0; count];
        sheet.preconditioner.apply(&sheet.stiffness, &a, &mut pa);
        sheet.preconditioner.apply(&sheet.stiffness, &b, &mut pb);
        let energy_a = dot(&a, &pa);
        let energy_b = dot(&b, &pb);
        assert!(energy_a > 0.0 && energy_b > 0.0);
        assert!((dot(&a, &pb) - dot(&b, &pa)).abs() <= 1e-11 * (energy_a * energy_b).sqrt());
    }

    fn sparse_from_dense<const N: usize>(dense: [[f64; N]; N]) -> SparseMatrix {
        SparseMatrix::from_rows(
            dense
                .iter()
                .map(|row| {
                    row.iter().copied().enumerate().filter(|(_, value)| *value != 0.0).collect()
                })
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn complete_small_pattern_factor_matches_the_exact_inverse_action() {
        let matrix = sparse_from_dense([[4.0, 1.0, 0.5], [1.0, 3.0, -0.3], [0.5, -0.3, 2.0]]);
        let factor = IncompleteCholesky::factor(&matrix, 0.0).unwrap();
        let rhs = [0.7, -1.3, 0.4];
        let mut solution = [0.0; 3];
        let mut product = [0.0; 3];
        factor.apply(&rhs, &mut solution);
        matrix.multiply(&solution, &mut product);
        for (actual, expected) in product.iter().zip(rhs) {
            assert!((actual - expected).abs() < 1e-14);
        }
    }

    #[test]
    fn shifted_factor_handles_a_positive_definite_zero_fill_breakdown() {
        // Eigenvalues are 1 +/- sqrt(2)*0.6, both positive. Zero-fill
        // elimination drops a crucial cancellation and its final pivot fails.
        let matrix = sparse_from_dense([
            [1.0, 0.6, 0.0, -0.6],
            [0.6, 1.0, 0.6, 0.0],
            [0.0, 0.6, 1.0, 0.6],
            [-0.6, 0.0, 0.6, 1.0],
        ]);
        assert!(IncompleteCholesky::factor(&matrix, 0.0).is_none());
        let before = matrix.values.clone();
        let preconditioner = Preconditioner::new(&matrix, PreconditionerKind::IncompleteCholesky);
        let factor = preconditioner.factor.as_ref().unwrap();
        assert!(factor.shift > 0.0);
        assert!(factor.inverse_diagonal.iter().all(|pivot| pivot.is_finite() && *pivot > 0.0));
        assert_eq!(matrix.values, before, "retry must never regularize the physical operator");
        let mut diagnostics =
            FieldDiagnostics { forcing_samples: 1, ..FieldDiagnostics::default() };
        preconditioner.write_diagnostics(&mut diagnostics);
        diagnostics.validate().unwrap();
        assert!(!diagnostics.preconditioner_fallback);
    }

    #[test]
    fn exhausted_factor_retries_report_jacobi_fallback() {
        // Deliberately non-SPD input exercises the bounded failure path only;
        // it is not a permissible elastic operator and is never passed to CG.
        let matrix = sparse_from_dense([[1.0, 4.0], [4.0, 1.0]]);
        let preconditioner = Preconditioner::new(&matrix, PreconditionerKind::IncompleteCholesky);
        assert!(preconditioner.factor.is_none());
        let mut diagnostics =
            FieldDiagnostics { forcing_samples: 1, ..FieldDiagnostics::default() };
        preconditioner.write_diagnostics(&mut diagnostics);
        assert_eq!(diagnostics.preconditioner, PreconditionerKind::Jacobi);
        assert!(diagnostics.preconditioner_fallback);
        assert_eq!(diagnostics.preconditioner_diagonal_shift, 0.0);
        diagnostics.validate().unwrap();
    }

    #[test]
    fn incomplete_factor_and_jacobi_recover_the_same_stress() {
        let cfg = FieldConfig { cg_tolerance: 1e-12, ..config() };
        let source = source();
        let fast = ElasticSheet::new(&cfg).unwrap().solve(&source, 0.47).unwrap();
        let reference = ElasticSheet::new(&FieldConfig {
            preconditioner: PreconditionerKind::Jacobi,
            ..cfg.clone()
        })
        .unwrap()
        .solve(&source, 0.47)
        .unwrap();
        assert!(fast.diagnostics.relative_residual <= cfg.cg_tolerance);
        assert!(reference.diagnostics.relative_residual <= cfg.cg_tolerance);
        let difference = fast
            .stress
            .iter()
            .flatten()
            .zip(reference.stress.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            difference < 5e-9 * reference.diagnostics.max_stress_component,
            "max stress difference={difference:e}"
        );
    }

    #[test]
    fn pair_loads_balance_and_cg_meets_true_residual() {
        let sheet = ElasticSheet::new(&config()).unwrap();
        let field = sheet.solve(&source(), 0.47).unwrap();
        assert!(field.diagnostics.relative_force_imbalance < 1e-13);
        assert!(field.diagnostics.relative_residual <= sheet.config.cg_tolerance);
        assert!(field.diagnostics.cg_iterations > 0);
        assert!(field.diagnostics.strain_energy > 0.0);
        assert!(field.diagnostics.max_stress_component > 0.0);
    }

    fn manufactured_error(grid: [usize; 2]) -> f64 {
        let cfg = FieldConfig { grid, load_softness: 1.2, ..FieldConfig::default() };
        let sheet = ElasticSheet::new(&cfg).unwrap();
        let [a, b] = cfg.domain_half_size;
        let kx = PI / (2.0 * a);
        let ky = PI / (2.0 * b);
        let mu = 1.0 / (2.0 * (1.0 + cfg.poisson_ratio));
        let lambda = cfg.poisson_ratio / (1.0 - cfg.poisson_ratio.powi(2));
        let forces: Vec<[f64; 2]> = sheet
            .nodes
            .iter()
            .zip(&sheet.nodal_area)
            .map(|(&[x, y], area)| {
                let (sx, cx) = (kx * (x + a)).sin_cos();
                let (sy, cy) = (ky * (y + b)).sin_cos();
                [
                    area * ((lambda + 2.0 * mu) * kx * kx + mu * ky * ky) * sx * sy,
                    -area * (lambda + mu) * kx * ky * cx * cy,
                ]
            })
            .collect();
        let rhs = sheet.free_vector(&forces);
        let displacement =
            sheet.solve_displacements(&rhs, &mut FieldDiagnostics::default()).unwrap();
        let error: f64 = sheet
            .nodes
            .iter()
            .zip(&displacement)
            .zip(&sheet.nodal_area)
            .map(|((&[x, y], u), area)| {
                let exact = (kx * (x + a)).sin() * (ky * (y + b)).sin();
                area * ((u[0] - exact).powi(2) + u[1].powi(2))
            })
            .sum();
        error.sqrt()
    }

    #[test]
    fn manufactured_smooth_solution_converges_under_mesh_refinement() {
        let coarse = manufactured_error([13, 11]);
        let fine = manufactured_error([25, 21]);
        assert!(fine < 0.65 * coarse, "coarse={coarse:e}, fine={fine:e}");
    }

    #[test]
    fn frame_order_and_memory_sampling_are_deterministic() {
        let cfg = config();
        let sheet = ElasticSheet::new(&cfg).unwrap();
        let source = source();
        let a = sheet.solve(&source, 0.43).unwrap();
        sheet.solve(&source, 0.81).unwrap();
        let b = sheet.solve(&source, 0.43).unwrap();
        assert_eq!(a.stress, b.stress);
        let schedule = forcing_schedule(&cfg, 0.0, 0.003);
        assert!(
            schedule.iter().all(|(time, weight)| *time >= 0.0 && *time <= 0.003 && *weight > 0.0)
        );
        assert!((schedule.iter().map(|entry| entry.1).sum::<f64>() - 1.0).abs() < 1e-14);
        assert_eq!(forcing_schedule(&cfg, 0.0, 0.0), vec![(0.0, 1.0)]);
        assert!(sheet.solve(&source, -0.1).is_err());
    }

    fn linear_forcing_average(config: &FieldConfig, time: f64) -> f64 {
        forcing_schedule(config, 0.0, time)
            .iter()
            .map(|(sample_time, weight)| sample_time * weight)
            .sum()
    }

    #[test]
    fn compact_memory_has_continuous_linear_response_slope_at_startup_cutoff() {
        let cfg = FieldConfig {
            memory_fraction: 0.2,
            memory_weight: 0.55,
            memory_samples: 12,
            ..config()
        };
        let time = cfg.memory_fraction;
        let h = time * 1e-5;
        let left = linear_forcing_average(&cfg, time - h);
        let center = linear_forcing_average(&cfg, time);
        let right = linear_forcing_average(&cfg, time + h);
        let left_slope = (center - left) / h;
        let right_slope = (right - center) / h;
        assert!(
            (left_slope - right_slope).abs() < 1e-8,
            "left={left_slope:.12}, right={right_slope:.12}"
        );
        assert!((right_slope - 1.0).abs() < 1e-9);
    }

    #[test]
    fn temporal_quadrature_converges_to_the_analytic_compact_kernel_lag() {
        // Integrals of exp(-3u)(1-u)^2 and u exp(-3u)(1-u)^2 are
        // (5-2 exp(-3))/27 and (1-4 exp(-3))/27 respectively.
        let mean_age_fraction = (1.0 - 4.0 * (-3.0_f64).exp()) / (5.0 - 2.0 * (-3.0_f64).exp());
        let cfg = FieldConfig { memory_fraction: 0.2, memory_weight: 0.55, ..config() };
        let time = 0.7;
        let exact = time - cfg.memory_weight * cfg.memory_fraction * mean_age_fraction;
        let coarse = FieldConfig { memory_samples: 4, ..cfg.clone() };
        let fine = FieldConfig { memory_samples: 8, ..cfg };
        let coarse_error = (linear_forcing_average(&coarse, time) - exact).abs();
        let fine_error = (linear_forcing_average(&fine, time) - exact).abs();
        assert!(fine_error < coarse_error * 1e-4, "coarse={coarse_error:e}, fine={fine_error:e}");
        assert!(fine_error < 1e-12);
        for count in [1, 4, 12, 24, 128] {
            let quadrature = temporal_quadrature(count);
            assert_eq!(quadrature.len(), count);
            assert!(
                quadrature.iter().all(|(node, weight)| *node > 0.0 && *node < 1.0 && *weight > 0.0)
            );
            assert!((quadrature.iter().map(|entry| entry.1).sum::<f64>() - 1.0).abs() < 1e-13);
        }
    }

    #[test]
    fn stress_interpolation_is_continuous_with_continuous_first_derivative() {
        let sheet = ElasticSheet::new(&config()).unwrap();
        let field = sheet.solve(&source(), 0.5).unwrap();
        let h = 1e-5;
        let left = field.sample(-h, 0.23);
        let center = field.sample(0.0, 0.23);
        let right = field.sample(h, 0.23);
        for axis in 0..3 {
            assert!((left[axis] + right[axis] - 2.0 * center[axis]).abs() / h < 1e-4);
        }
    }
}
