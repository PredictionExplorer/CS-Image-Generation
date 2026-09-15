//! Tidal Silk: original three-body motion expressed through a continuous fabric.
//!
//! Orbit generation, cloth baking and lighting are independent stages. Stable
//! mesh identifiers and per-pixel sampling preserve repeatability across threads.

pub mod cache;
pub mod comparison;
/// Body-position guides for synchronized visualization comparisons.
pub mod guides;
mod math;
/// Replay the original accumulated-light rendering from a recorded source orbit.
pub mod normal;
pub mod orbit;
pub mod render;
pub mod simulation;

pub use math::V3;
use serde::{Deserialize, Serialize};

/// A uniformly sampled original Newtonian trajectory, before artistic transforms.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrbitData {
    /// Canonical source seed.
    pub seed: String,
    /// Simulation time between samples.
    pub dt: f64,
    /// Three source masses.
    pub masses: [f64; 3],
    /// Recorded physical positions, indexed by time and then body.
    pub samples: Vec<[V3; 3]>,
    /// Initial conditions and selection settings needed to reconstruct this orbit.
    pub provenance: serde_json::Value,
}

/// Fixed topology and material coordinates shared by every baked frame.
#[derive(Clone, Debug)]
pub struct Mesh {
    /// Initial vertex positions.
    pub positions: Vec<V3>,
    /// Consistently oriented triangle vertex indices.
    pub triangles: Vec<[u32; 3]>,
    /// Material coordinates that travel with each vertex.
    pub uv: Vec<[f64; 2]>,
}

/// Numerical diagnostics for a visible simulation frame.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FrameStats {
    /// Largest ratio between current and rest structural edge length.
    pub max_stretch: f64,
    /// Contact constraints resolved during the frame.
    pub contacts: usize,
    /// Maximum distance between an attachment and its target.
    pub pin_error: f64,
}

/// Reusable cloth geometry, independent of materials, lights and camera.
#[derive(Clone, Debug)]
pub struct ClothBake {
    /// Source mesh topology and material coordinates.
    pub mesh: Mesh,
    /// Visible vertex positions at every frame.
    pub frames: Vec<Vec<V3>>,
    /// Frames per second.
    pub fps: u32,
    /// Per-frame simulation diagnostics.
    pub stats: Vec<FrameStats>,
    /// Versioned source, normalization and cloth configuration.
    pub recipe: serde_json::Value,
}

/// Errors at a silk pipeline boundary.
pub type SilkResult<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
