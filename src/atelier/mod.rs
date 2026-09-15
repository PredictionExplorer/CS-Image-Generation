//! Six artistic interpretations of one recorded three-body motion.
//!
//! Physical trajectories remain frozen. Designed geometry, optics and graphic
//! fields turn their motion into distinct artworks using deterministic CPU work.

pub mod aurora;
pub mod calligraphy;
pub mod loom;
pub mod render;
pub mod source;

pub use crate::silk::{SilkResult, V3};
use serde::{Deserialize, Serialize};
pub use source::{BodySample, OrbitSeries, SourceFrame};

/// A smooth surface vertex in the shared, fixed world coordinate system.
#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    /// Position after the one fixed source normalization.
    pub position: V3,
    /// Oriented unit normal; its sign identifies the fabric's front face.
    pub normal: V3,
    /// Unit fiber direction along the surface.
    pub tangent: V3,
    /// Material coordinates along and across the band.
    pub uv: [f64; 2],
}

/// A material-indexed triangle; all indices address the scene's shared vertices.
#[derive(Clone, Copy, Debug)]
pub struct Triangle {
    /// Three counterclockwise vertex indices.
    pub indices: [u32; 3],
    /// Index in the scene material array.
    pub material: usize,
}

/// A fine continuous fiber with physical world-space radii.
#[derive(Clone, Debug)]
pub struct Strand {
    /// Ordered control points; adjacent points define a segment.
    pub points: Vec<V3>,
    /// Radius at each point, interpolated along each segment.
    pub radii: Vec<f64>,
    /// Index in the scene material array.
    pub material: usize,
}

/// Thin dyed fabric or fiber optics, evaluated in linear light.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Material {
    /// Front-face reflected color in linear RGB.
    pub front_color: V3,
    /// Reverse-face reflected color in linear RGB.
    pub back_color: V3,
    /// Per-channel absorption at normal incidence (Beer--Lambert optical depth).
    pub optical_depth: V3,
    /// Broadness of the satin reflection lobe.
    pub roughness: f64,
    /// Directionality of highlights along the fiber tangent.
    pub anisotropy: f64,
    /// Strength of grazing fiber reflection.
    pub sheen: f64,
    /// Optional emitted radiance, for explicitly luminous interpretations.
    pub emission: V3,
    /// Density of fine material fibers across one unit of transverse UV.
    pub fiber_frequency: f64,
    /// Restrained strength of material fiber modulation.
    pub fiber_strength: f64,
    /// Conductor fraction: zero preserves dielectric fabric, one is opaque metal.
    /// Omitted at zero so existing serialized Calligraphy recipes retain their hash.
    #[serde(skip_serializing_if = "metallic_is_zero")]
    pub metallic: f64,
    /// Optional continuous luminous-gas color and density profile.
    /// Its density and edge envelope fade both emission and extinction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emission_profile: Option<EmissionProfile>,
}

fn metallic_is_zero(value: &f64) -> bool {
    *value == 0.0
}

impl Default for Material {
    fn default() -> Self {
        Self {
            front_color: V3::new(0.72, 0.68, 0.57),
            back_color: V3::new(0.24, 0.22, 0.40),
            optical_depth: V3::new(0.28, 0.31, 0.37),
            roughness: 0.31,
            anisotropy: 0.72,
            sheen: 0.45,
            emission: V3::ZERO,
            fiber_frequency: 420.0,
            fiber_strength: 0.12,
            metallic: 0.0,
            emission_profile: None,
        }
    }
}

/// One control point of a continuous linear-light emission and density curve.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmissionStop {
    /// Normalized position along the profile's world-space axis.
    pub position: f64,
    /// Additional linear HDR emission before density and edge fading.
    pub emission: V3,
    /// Amount of emitting and absorbing material, from zero to one.
    pub density: f64,
}

/// Smooth fades measured inward from the edges of a unit material-UV square.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UvFeather {
    /// Fade widths at u=0 and u=1; zero disables the corresponding edge fade.
    pub u: [f64; 2],
    /// Fade widths at v=0 and v=1; zero disables the corresponding edge fade.
    pub v: [f64; 2],
}

/// A smooth world-space emission profile shared by sheet and strand geometry.
///
/// Coordinates are `dot(point-origin, normalized(axis))/extent`. Ordered stops
/// use bounded cubic interpolation and clamp to the end stops outside their
/// range. Density scales optical depth before Beer--Lambert attenuation and
/// scales the sum of base material emission and profile emission. Give the end
/// stops zero density to make the surrounding space entirely clear and dark.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EmissionProfile {
    /// World-space position of profile coordinate zero.
    pub origin: V3,
    /// Fixed world-space axis; normalized before evaluation.
    pub axis: V3,
    /// Positive world-space distance corresponding to one profile unit.
    pub extent: f64,
    /// At least two strictly ordered stops, with positions in zero to one.
    pub stops: Vec<EmissionStop>,
    /// Optional material-UV edge fades for sheets; leave absent for strands.
    pub uv_feather: Option<UvFeather>,
}

impl Default for EmissionProfile {
    fn default() -> Self {
        Self {
            origin: V3::ZERO,
            axis: V3::new(0.0, 1.0, 0.0),
            extent: 1.0,
            stops: [(0.0, 0.0), (0.2, 1.0), (0.8, 1.0), (1.0, 0.0)]
                .map(|(position, density)| EmissionStop {
                    position,
                    emission: V3::new(1.0, 1.0, 1.0),
                    density,
                })
                .into(),
            uv_feather: None,
        }
    }
}

/// Complete geometry for one frame, independent of its camera and pixel count.
#[derive(Clone, Debug, Default)]
pub struct Scene {
    /// Shared smooth surface vertices.
    pub vertices: Vec<Vertex>,
    /// Surface triangles.
    pub triangles: Vec<Triangle>,
    /// Fine strands rendered without constructing large tube meshes.
    pub strands: Vec<Strand>,
    /// Surface and fiber materials.
    pub materials: Vec<Material>,
}

/// Fixed orthographic composition, shared by all frames of a film.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Camera {
    /// Viewing location; distance only establishes the depth origin.
    pub position: V3,
    /// Point at the center of the image.
    pub target: V3,
    /// Approximate image-up axis, orthogonalized against the viewing direction.
    pub up: V3,
    /// World-space height visible in the image.
    pub orthographic_height: f64,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: V3::new(1.7, 1.1, 7.0),
            target: V3::ZERO,
            up: V3::new(0.0, 1.0, 0.0),
            orthographic_height: 4.8,
        }
    }
}

/// High-resolution CPU film settings; geometry density is configured separately.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RenderConfig {
    /// Encoded output width.
    pub width: u32,
    /// Encoded output height.
    pub height: u32,
    /// Number of stratified coverage samples on each pixel axis.
    pub aa: u32,
    /// Exposure offset in stops before the highlight shoulder and sRGB encoding.
    pub exposure: f64,
    /// Quiet background color in linear RGB.
    pub background: V3,
    /// Strength of the broad frontal strip light.
    pub key_strength: f64,
    /// Strength of cool grazing light.
    pub rim_strength: f64,
    /// Strength of broad ambient fill.
    pub fill_strength: f64,
    /// Angle in degrees rotating studio lights around the view axis.
    pub light_rotation_degrees: f64,
    /// Small optical glow around unusually bright highlights.
    pub bloom_strength: f64,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            width: 3840,
            height: 2160,
            aa: 2,
            exposure: 0.35,
            background: V3::new(0.0022, 0.0030, 0.0055),
            key_strength: 1.0,
            rim_strength: 1.2,
            fill_strength: 0.20,
            light_rotation_degrees: 20.0,
            bloom_strength: 0.045,
        }
    }
}
