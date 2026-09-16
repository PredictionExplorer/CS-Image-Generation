//! The Remaining Form: source-driven cumulative excavation of a real 3D solid.
//!
//! Geometry, surface extraction, and photographic rendering are separate stages.
//! A field records nonnegative carving exposure through the unchanged source
//! timeline. It is an implicit material field, not a signed-distance field for
//! ray marching. Verified surface meshes are exported to a pinned offline renderer.
pub mod field;
pub mod mesh;

pub use crate::atelier::{OrbitSeries, SilkResult, V3};

/// A regular isotropic scalar grid; negative values identify surviving material.
///
/// Nodes use X-fastest order: `x + nx * (y + ny * z)`. Every dimension counts
/// nodes, not cells. `origin` is the position of node `[0,0,0]`.
#[derive(Clone, Debug)]
pub struct Grid {
    /// Number of nodes in X, Y and Z, each at least two.
    pub dims: [usize; 3],
    /// World-space position of the first grid node.
    pub origin: V3,
    /// Positive, uniform node spacing in all three directions.
    pub spacing: f64,
    /// X-fastest scalar values, finite and matching the dimensions exactly.
    pub values: Vec<f64>,
}
