//! The main edition's view of the orbit, followed step for step by the ember edition.
//!
//! The main edition does not draw the raw orbit. It draws it in the seed's projection space,
//! rotated to the best-composed of several viewing angles, carried along a drift path, and framed
//! by an aspect-corrected bounding box. The ember edition shows the same motion: at every recorded
//! step its three bodies sit where the main edition draws the head of each trail.
//!
//! # Why the view is recorded and re-applied
//!
//! The main pipeline computes its view with the platform's `sin`, `cos` and `ln` and chooses the
//! viewing angle with a platform-dependent score, so its transformed positions are not
//! reproducible bit for bit on another CPU or operating system, while everything the ember
//! edition renders must be (docs/ember-design.md §0.1). The main pipeline therefore *captures*
//! the view it resolved, as exact bit patterns, in a [`View`]: the projection mode, the viewing
//! rotation as a matrix, the drift as the quantities it actually added, and the frame. The ember
//! edition *re-applies* that view to the raw orbit it re-simulates, with exactly rounded
//! arithmetic and the portable `math` functions only. The two tracks agree to a few units in
//! the last place (far below a millionth of a pixel), and the ember track is a pure function of
//! the certificate's inputs on every CPU.
//!
//! # Recipe (docs/ember-design.md §3.1–§3.5)
//!
//! With raw positions `p_b[k]` (body `b`, recorded step `k`, `N` steps) and recorded step `dt`:
//!
//! 1. **Projection space.** Velocities by forward difference, the last step repeated:
//!    `v[k] = (p[min(k+1, N-1)] - p[min(k+1, N-1) - 1]) / dt`. With `ext(q) = max(max q - min q,
//!    10⁻¹²)` over every body and step, `velocity_scale = max(ext(p.x), ext(p.y)) /
//!    max(ext(v.x), ext(v.y))` (1 if the denominator is at most `10⁻¹²`) and `w = v ·
//!    velocity_scale`. Then `Position: (p.x, p.y, p.z)`, `PhasePortrait: (p.x, w.x, p.y)`,
//!    `CrossBraid: (p.x, p'.y, p.z)` with `p'` the next body, `Hodograph: (w.x, w.y, p.z)`.
//! 2. **Viewing rotation.** `q = R·s`, each component `(R_i0·s.x + R_i1·s.y) + R_i2·s.z`.
//! 3. **Drift.** `q += o[k]`, the same offset for the three bodies: none; linear
//!    `o = (V·k)·dt`; or elliptical `o = D·(a·(cos E - e), b·sin E, 0)` with `E` the eccentric
//!    anomaly of the mean anomaly `M = wrap(M₀ + n·(k·dt))`: the root of `E - e·sin E = M` by
//!    Newton's iteration from `E = M`, at most 8 steps, stopping at a step below `10⁻¹²`.
//! 4. **Frame.** `nx = (q.x - min_x)/width`, `ny = (q.y - min_y)/height`, and on the canvas
//!    `[-aspect, aspect] × [-1, 1]` (y up, whereas the main image's y points down):
//!    `x = aspect·(2·nx - 1)·scale`, `y = (1 - 2·ny)·scale`.
//!
//! Only `x` and `y` of `q` reach the frame, so the third component is never formed.

use std::f64::consts::{PI, TAU};

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

use super::error::{EmberError, EmberResult};
use super::math::{self, max, min};

/// Number of bodies of a three-body orbit.
const BODIES: usize = 3;

/// Smallest extent used when the projection space rescales velocities (the main edition's
/// guard against dividing by a collapsed extent).
const MIN_EXTENT: f64 = 1e-12;

/// Most Newton iterations of [`eccentric_anomaly`].
const KEPLER_ITERATIONS: usize = 8;

/// Newton step below which [`eccentric_anomaly`] stops.
const KEPLER_TOLERANCE: f64 = 1e-12;

/// Relative slack of the check that the frame has the canvas's aspect ratio.
const FRAME_ASPECT_TOLERANCE: f64 = 1e-9;

/// Relative slack of the check that every body stays on the canvas (rounding at the frame edge).
const CANVAS_TOLERANCE: f64 = 1e-9;

/// The main edition's view of the orbit: what the ember edition must re-apply so that its bodies
/// follow the same on-screen motion. Every number is the value the main pipeline resolved,
/// recorded as an exact bit pattern in the certificate (`inputs.view`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct View {
    /// The seed's projection space.
    pub projection: ViewProjection,
    /// Viewing rotation, row-major: the rotation the main edition chose among its candidates.
    #[serde(with = "bits::matrix")]
    pub rotation: [[f64; 3]; 3],
    /// The drift the main edition added.
    pub drift: ViewDrift,
    /// How the main edition frames the transformed orbit.
    pub frame: ViewFrame,
}

/// The seed's projection space (the main edition's `ProjectionMode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViewProjection {
    /// Plain positions `(x, y, z)`.
    Position,
    /// Per-body phase portrait `(x, v_x, y)`.
    PhasePortrait,
    /// Cross-body braid: body `i` takes its `y` from body `(i + 1) % 3`.
    CrossBraid,
    /// Velocity space `(v_x, v_y, z)`.
    Hodograph,
}

/// The offset the main edition's drift added to all three bodies at every step.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum ViewDrift {
    /// No drift. (A variant without fields would be read leniently: serde ignores unknown keys
    /// next to the tag of a unit variant, and the certificate's reader must reject them.)
    None {},
    /// Constant velocity: `offset[k] = (velocity·k)·dt`.
    Linear {
        /// Drift velocity.
        #[serde(with = "bits::vector")]
        velocity: [f64; 3],
    },
    /// A Keplerian arc: `offset[k] = rotation·(a·(cos E - e), b·sin E, 0)`.
    Elliptical {
        /// Orientation of the drift ellipse, row-major.
        #[serde(with = "bits::matrix")]
        rotation: [[f64; 3]; 3],
        /// Mean anomaly `M₀` at step 0, in radians.
        #[serde(with = "bits::scalar")]
        mean_anomaly: f64,
        /// Mean motion `n`, in radians per unit of simulation time.
        #[serde(with = "bits::scalar")]
        mean_motion: f64,
        /// Eccentricity `e ∈ [0, 1)`.
        #[serde(with = "bits::scalar")]
        eccentricity: f64,
        /// Semi-major axis `a`.
        #[serde(with = "bits::scalar")]
        semi_major: f64,
        /// Semi-minor axis `b`.
        #[serde(with = "bits::scalar")]
        semi_minor: f64,
    },
}

/// The main edition's frame: its aspect-corrected bounding box, and the scale about the frame
/// centre that its symmetric seeds apply to the primary copy of every stroke (1 otherwise).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViewFrame {
    /// Left edge of the bounding box.
    #[serde(with = "bits::scalar")]
    pub min_x: f64,
    /// The bounding box's edge that maps to the top row of the image.
    #[serde(with = "bits::scalar")]
    pub min_y: f64,
    /// Width of the bounding box.
    #[serde(with = "bits::scalar")]
    pub width: f64,
    /// Height of the bounding box.
    #[serde(with = "bits::scalar")]
    pub height: f64,
    /// Scale about the frame centre, in `(0, 1]`.
    #[serde(with = "bits::scalar")]
    pub scale: f64,
}

impl View {
    /// Rejects a view the ember edition cannot follow: a non-finite number, a frame without a
    /// positive size, a scale outside `(0, 1]`, or a drift ellipse that is not an ellipse.
    ///
    /// # Errors
    ///
    /// [`EmberError::InvalidView`] naming the offending part.
    pub fn validate(&self) -> EmberResult<()> {
        let finite = |values: &[f64], what: &str| {
            if values.iter().all(|value| value.is_finite()) {
                Ok(())
            } else {
                invalid(format!("{what} is not finite ({values:?})"))
            }
        };
        finite(self.rotation.as_flattened(), "the viewing rotation")?;
        let frame = self.frame;
        finite(&[frame.min_x, frame.min_y, frame.width, frame.height, frame.scale], "the frame")?;
        if !(frame.width > 0.0 && frame.height > 0.0) {
            return invalid(format!("the frame is {} by {}", frame.width, frame.height));
        }
        if !(frame.scale > 0.0 && frame.scale <= 1.0) {
            return invalid(format!("the frame scale {} is outside (0, 1]", frame.scale));
        }
        match self.drift {
            ViewDrift::None {} => Ok(()),
            ViewDrift::Linear { velocity } => finite(&velocity, "the drift velocity"),
            ViewDrift::Elliptical {
                rotation,
                mean_anomaly,
                mean_motion,
                eccentricity,
                semi_major,
                semi_minor,
            } => {
                finite(rotation.as_flattened(), "the drift orientation")?;
                finite(
                    &[mean_anomaly, mean_motion, eccentricity, semi_major, semi_minor],
                    "the drift ellipse",
                )?;
                if !(0.0..1.0).contains(&eccentricity) {
                    return invalid(format!(
                        "the drift eccentricity {eccentricity} is outside [0, 1)"
                    ));
                }
                if !(semi_major >= 0.0 && semi_minor >= 0.0) {
                    return invalid(format!(
                        "the drift semi-axes are {semi_major} and {semi_minor}"
                    ));
                }
                Ok(())
            }
        }
    }

    /// The three bodies' canvas positions at every recorded step, `track[body][step]`, on the
    /// canvas `[-aspect, aspect] × [-1, 1]` (y up): the module's recipe applied to the raw orbit
    /// `raw[body][step]` recorded with time step `dt`.
    ///
    /// # Errors
    ///
    /// [`EmberError::InvalidView`] for a view that fails [`View::validate`] or a non-positive
    /// `dt` or `aspect`; [`EmberError::DegenerateOrbit`] unless the orbit has exactly three
    /// bodies with the same number (at least 2) of finite recorded positions.
    pub(crate) fn canvas_track(
        &self,
        raw: &[Vec<Vector3<f64>>],
        dt: f64,
        aspect: f64,
    ) -> EmberResult<[Vec<[f64; 2]>; BODIES]> {
        self.validate()?;
        if !(dt.is_finite() && dt > 0.0) {
            return invalid(format!("the recorded time step {dt} must be finite and > 0"));
        }
        if !(aspect.is_finite() && aspect > 0.0) {
            return invalid(format!("the canvas aspect {aspect} must be finite and > 0"));
        }
        let bodies = orbit_shape(raw)?;
        let steps = bodies[0].len();
        let velocity_scale = self.velocity_scale(bodies, dt);
        let frame = self.frame;
        // The frame must scale both axes alike, or the canvas would shear the orbit.
        if ((frame.width / frame.height) / aspect - 1.0).abs() > FRAME_ASPECT_TOLERANCE {
            return invalid(format!(
                "the frame is {} by {}, not the canvas aspect {aspect}",
                frame.width, frame.height
            ));
        }

        let mut track: [Vec<[f64; 2]>; BODIES] = std::array::from_fn(|_| Vec::with_capacity(steps));
        for step in 0..steps {
            let offset = self.drift.offset(step, dt);
            for (body, points) in track.iter_mut().enumerate() {
                let q = rotated_xy(
                    &self.rotation,
                    self.projected(bodies, body, step, dt, velocity_scale),
                );
                let (x, y) = (q[0] + offset[0], q[1] + offset[1]);
                let nx = (x - frame.min_x) / frame.width;
                let ny = (y - frame.min_y) / frame.height;
                let point =
                    [aspect * (2.0 * nx - 1.0) * frame.scale, (1.0 - 2.0 * ny) * frame.scale];
                if !(point[0].is_finite() && point[1].is_finite()) {
                    return degenerate(format!(
                        "body {body} has no finite canvas position at step {step}"
                    ));
                }
                // The frame encloses the whole transformed orbit, so a body off the canvas means
                // the view does not belong to this orbit.
                let reach = 1.0 + CANVAS_TOLERANCE;
                if point[0].abs() > aspect * reach || point[1].abs() > reach {
                    return invalid(format!(
                        "body {body} leaves the canvas at step {step} ({}, {}): the view does not \
                         frame this orbit",
                        point[0], point[1]
                    ));
                }
                points.push(point);
            }
        }
        Ok(track)
    }

    /// `velocity_scale` of the projection space (step 1 of the recipe); 1 for plain positions,
    /// which never use it.
    fn velocity_scale(&self, bodies: [&[Vector3<f64>]; BODIES], dt: f64) -> f64 {
        if self.projection == ViewProjection::Position {
            return 1.0;
        }
        let steps = bodies[0].len();
        // Extents of x and y, of the positions and of the velocities, over finite values.
        let mut low = [f64::INFINITY; 4];
        let mut high = [f64::NEG_INFINITY; 4];
        for body in bodies {
            for step in 0..steps {
                let (p, v) = (body[step], velocity(body, step, dt));
                for (slot, value) in [p.x, p.y, v[0], v[1]].into_iter().enumerate() {
                    if value.is_finite() {
                        low[slot] = min(low[slot], value);
                        high[slot] = max(high[slot], value);
                    }
                }
            }
        }
        let extent = |slot: usize| max(high[slot] - low[slot], MIN_EXTENT);
        let position_extent = max(extent(0), extent(1));
        let velocity_extent = max(extent(2), extent(3));
        if velocity_extent > MIN_EXTENT { position_extent / velocity_extent } else { 1.0 }
    }

    /// The point of `body` at `step` in the seed's projection space (step 1 of the recipe).
    fn projected(
        &self,
        bodies: [&[Vector3<f64>]; BODIES],
        body: usize,
        step: usize,
        dt: f64,
        velocity_scale: f64,
    ) -> [f64; 3] {
        let p = bodies[body][step];
        let scaled =
            || velocity(bodies[body], step, dt).map(|component| component * velocity_scale);
        match self.projection {
            ViewProjection::Position => [p.x, p.y, p.z],
            ViewProjection::PhasePortrait => [p.x, scaled()[0], p.y],
            ViewProjection::CrossBraid => [p.x, bodies[(body + 1) % BODIES][step].y, p.z],
            ViewProjection::Hodograph => {
                let w = scaled();
                [w[0], w[1], p.z]
            }
        }
    }
}

impl ViewDrift {
    /// The `(x, y)` of the offset added to every body at `step` (step 3 of the recipe).
    fn offset(&self, step: usize, dt: f64) -> [f64; 2] {
        match *self {
            Self::None {} => [0.0, 0.0],
            Self::Linear { velocity } => {
                let k = step as f64;
                [(velocity[0] * k) * dt, (velocity[1] * k) * dt]
            }
            Self::Elliptical {
                rotation,
                mean_anomaly,
                mean_motion,
                eccentricity,
                semi_major,
                semi_minor,
            } => {
                let time = step as f64 * dt;
                let anomaly =
                    eccentric_anomaly(wrap_angle(mean_anomaly + mean_motion * time), eccentricity);
                let (sin, cos) = math::sin_cos(anomaly);
                let (x, y) = (semi_major * (cos - eccentricity), semi_minor * sin);
                [rotation[0][0] * x + rotation[0][1] * y, rotation[1][0] * x + rotation[1][1] * y]
            }
        }
    }
}

/// `x` and `y` of `rotation · s`, each row times the vector summed left to right: the order of
/// the main edition's matrix product, so the rotated point has the same bits.
fn rotated_xy(rotation: &[[f64; 3]; 3], s: [f64; 3]) -> [f64; 2] {
    [rotation[0], rotation[1]].map(|row| (row[0] * s[0] + row[1] * s[1]) + row[2] * s[2])
}

/// Velocity of a body at `step` by forward difference over the recorded step `dt`, the last step
/// repeating the one before it (`body` has at least 2 steps).
fn velocity(body: &[Vector3<f64>], step: usize, dt: f64) -> [f64; 3] {
    let next = (step + 1).min(body.len() - 1);
    let (a, b) = (body[next], body[next - 1]);
    [(a.x - b.x) / dt, (a.y - b.y) / dt, (a.z - b.z) / dt]
}

/// `angle` wrapped into `[-π, π]`: the remainder of a division by `2π` (exact), moved by one
/// turn if it lies outside.
fn wrap_angle(angle: f64) -> f64 {
    let remainder = math::fmod(angle, TAU);
    if remainder > PI {
        remainder - TAU
    } else if remainder < -PI {
        remainder + TAU
    } else {
        remainder
    }
}

/// Eccentric anomaly `E` of the mean anomaly `M`: the root of `E - e·sin E = M` by Newton's
/// iteration from `E = M`, at most [`KEPLER_ITERATIONS`] steps, stopping once a step is below
/// [`KEPLER_TOLERANCE`] (the main edition's solver, on the portable `sin` and `cos`).
fn eccentric_anomaly(mean_anomaly: f64, eccentricity: f64) -> f64 {
    if eccentricity.abs() <= f64::EPSILON {
        return mean_anomaly;
    }
    let mut anomaly = mean_anomaly;
    for _ in 0..KEPLER_ITERATIONS {
        let (sin, cos) = math::sin_cos(anomaly);
        let slope = 1.0 - eccentricity * cos;
        if slope.abs() <= f64::EPSILON {
            break;
        }
        let step = (anomaly - eccentricity * sin - mean_anomaly) / slope;
        anomaly -= step;
        if step.abs() < KEPLER_TOLERANCE {
            break;
        }
    }
    anomaly
}

/// The three bodies' recordings, checked: exactly three, equally long, at least 2 steps.
fn orbit_shape(raw: &[Vec<Vector3<f64>>]) -> EmberResult<[&[Vector3<f64>]; BODIES]> {
    let [a, b, c] = raw else {
        return degenerate(format!("expected 3 bodies, got {}", raw.len()));
    };
    if b.len() != a.len() || c.len() != a.len() {
        return degenerate(format!(
            "bodies have different recording lengths ({}, {}, {})",
            a.len(),
            b.len(),
            c.len()
        ));
    }
    if a.len() < 2 {
        return degenerate(format!("{} recorded steps; at least 2 are needed", a.len()));
    }
    Ok([a, b, c])
}

/// `Err(InvalidView)` with `reason`.
fn invalid<T>(reason: String) -> EmberResult<T> {
    Err(EmberError::InvalidView { reason })
}

/// `Err(DegenerateOrbit)` with `reason`.
fn degenerate<T>(reason: String) -> EmberResult<T> {
    Err(EmberError::DegenerateOrbit { reason })
}

/// Serde adapters that write every `f64` of a [`View`] as its exact bit pattern
/// ([`F64Bits`](super::certificate::F64Bits)), so that a certificate read back gives the same
/// view bit for bit.
mod bits {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use crate::ember::certificate::F64Bits;

    /// One `f64`.
    pub(super) mod scalar {
        use super::{Deserialize, Deserializer, F64Bits, Serialize, Serializer};

        pub(in super::super) fn serialize<S: Serializer>(
            value: &f64,
            serializer: S,
        ) -> Result<S::Ok, S::Error> {
            F64Bits::of(*value).serialize(serializer)
        }

        pub(in super::super) fn deserialize<'de, D: Deserializer<'de>>(
            deserializer: D,
        ) -> Result<f64, D::Error> {
            F64Bits::deserialize(deserializer).map(F64Bits::value)
        }
    }

    /// A 3-vector.
    pub(super) mod vector {
        use super::{Deserialize, Deserializer, F64Bits, Serialize, Serializer};

        pub(in super::super) fn serialize<S: Serializer>(
            value: &[f64; 3],
            serializer: S,
        ) -> Result<S::Ok, S::Error> {
            value.map(F64Bits::of).serialize(serializer)
        }

        pub(in super::super) fn deserialize<'de, D: Deserializer<'de>>(
            deserializer: D,
        ) -> Result<[f64; 3], D::Error> {
            <[F64Bits; 3]>::deserialize(deserializer).map(|bits| bits.map(F64Bits::value))
        }
    }

    /// A 3×3 matrix, row-major.
    pub(super) mod matrix {
        use super::{Deserialize, Deserializer, F64Bits, Serialize, Serializer};

        pub(in super::super) fn serialize<S: Serializer>(
            value: &[[f64; 3]; 3],
            serializer: S,
        ) -> Result<S::Ok, S::Error> {
            value.map(|row| row.map(F64Bits::of)).serialize(serializer)
        }

        pub(in super::super) fn deserialize<'de, D: Deserializer<'de>>(
            deserializer: D,
        ) -> Result<[[f64; 3]; 3], D::Error> {
            <[[F64Bits; 3]; 3]>::deserialize(deserializer)
                .map(|rows| rows.map(|row| row.map(F64Bits::value)))
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use nalgebra::Matrix3;
    use sha2::{Digest, Sha256};

    use super::*;

    /// Deterministic pseudo-random numbers in `[0, 1)` (64-bit LCG, top 53 bits).
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        fn range(&mut self, low: f64, high: f64) -> f64 {
            low + (high - low) * self.next()
        }
    }

    const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    /// A view that leaves the orbit as it is and frames the square `[-1, 1]²` (y down).
    pub(crate) fn plain_view() -> View {
        View {
            projection: ViewProjection::Position,
            rotation: IDENTITY,
            drift: ViewDrift::None {},
            frame: ViewFrame { min_x: -1.0, min_y: -1.0, width: 2.0, height: 2.0, scale: 1.0 },
        }
    }

    /// The view of a raw orbit drawn as it is and framed as the main edition frames a
    /// trajectory: the bounding box of `x` and `y`, padded by 5% of its span on each side and
    /// widened on one axis to the canvas `aspect`.
    pub(crate) fn frontal_view(raw: &[Vec<Vector3<f64>>], aspect: f64) -> View {
        let points = || raw.iter().flatten();
        let span = |pick: fn(&Vector3<f64>) -> f64| {
            let low = points().map(pick).fold(f64::INFINITY, f64::min);
            let high = points().map(pick).fold(f64::NEG_INFINITY, f64::max);
            let pad = 0.05 * (high - low);
            (low - pad, high - low + 2.0 * pad)
        };
        let ((mut min_x, mut width), (mut min_y, mut height)) = (span(|p| p.x), span(|p| p.y));
        if width / height < aspect {
            min_x -= 0.5 * (height * aspect - width);
            width = height * aspect;
        } else {
            min_y -= 0.5 * (width / aspect - height);
            height = width / aspect;
        }
        View { frame: ViewFrame { min_x, min_y, width, height, scale: 1.0 }, ..plain_view() }
    }

    /// A generic rotation (orthonormal to rounding).
    fn tilted() -> [[f64; 3]; 3] {
        let (s1, c1) = math::sin_cos(0.7);
        let (s2, c2) = math::sin_cos(-1.1);
        [[c1, -s1 * c2, s1 * s2], [s1, c1 * c2, -c1 * s2], [0.0, s2, c2]]
    }

    /// Three bodies on wobbly loops, `steps` recorded positions each.
    fn wobbly(steps: usize) -> Vec<Vec<Vector3<f64>>> {
        (0..BODIES)
            .map(|body| {
                (0..steps)
                    .map(|k| {
                        let s = k as f64 / (steps - 1) as f64;
                        let (s1, c1) = math::sin_cos(TAU * (1.0 + body as f64) * s + body as f64);
                        let (s2, c2) = math::sin_cos(TAU * 3.0 * s * s);
                        Vector3::new(
                            3.0 * c1 + 0.4 * s2 + body as f64,
                            2.0 * s1 - 0.3 * c2,
                            0.5 * c1 * s2 + 0.1 * s,
                        )
                    })
                    .collect()
            })
            .collect()
    }

    /// [`plain_view`] with a frame of `2·half` by `2·half` about the origin.
    fn square_view(half: f64) -> View {
        let frame = ViewFrame {
            min_x: -half,
            min_y: -half,
            width: 2.0 * half,
            height: 2.0 * half,
            scale: 1.0,
        };
        View { frame, ..plain_view() }
    }

    #[test]
    fn a_plain_view_maps_the_frame_onto_the_canvas_with_y_flipped() {
        let raw: Vec<Vec<Vector3<f64>>> = vec![
            vec![Vector3::new(4.0, 10.0, 9.0), Vector3::new(7.0, 12.0, -9.0)],
            vec![Vector3::new(5.5, 11.0, 0.0), Vector3::new(6.25, 10.5, 0.0)],
            vec![Vector3::new(4.0, 12.0, 0.0), Vector3::new(7.0, 10.0, 0.0)],
        ];
        // A 3 × 2 frame off the origin, on a 3:2 canvas.
        let frame = ViewFrame { min_x: 4.0, min_y: 10.0, width: 3.0, height: 2.0, scale: 1.0 };
        let mut view = View { frame, ..plain_view() };
        let track = view.canvas_track(&raw, 0.001, 1.5).unwrap();
        // The frame's (min_x, min_y) corner is the image's top left: canvas (-aspect, +1).
        assert_eq!(track[0], vec![[-1.5, 1.0], [1.5, -1.0]]);
        assert_eq!(track[1], vec![[0.0, 0.0], [0.75, 0.5]]);
        assert_eq!(track[2], vec![[-1.5, -1.0], [1.5, 1.0]]);
        // The symmetric seeds' scale shrinks the picture about the frame centre.
        view.frame.scale = 0.5;
        assert_eq!(
            view.canvas_track(&raw, 0.001, 1.5).unwrap()[0],
            vec![[-0.75, 0.5], [0.75, -0.5]]
        );
    }

    /// The rotation is the main edition's `Matrix3 * Vector3`, bit for bit.
    #[test]
    fn the_rotation_has_the_bits_of_the_matrix_product() {
        let mut rng = Lcg(5);
        for _ in 0..2000 {
            let r: [[f64; 3]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| rng.range(-1.0, 1.0)));
            let matrix = Matrix3::new(
                r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2], r[2][0], r[2][1], r[2][2],
            );
            let p = Vector3::new(rng.range(-9.0, 9.0), rng.range(-9.0, 9.0), rng.range(-9.0, 9.0));
            let expected = matrix * p;
            let got = rotated_xy(&r, [p.x, p.y, p.z]);
            assert_eq!(got[0].to_bits(), expected.x.to_bits());
            assert_eq!(got[1].to_bits(), expected.y.to_bits());
        }
    }

    /// Each projection space against its definition, with velocities scaled to the positions'
    /// extent.
    #[test]
    fn projection_spaces_follow_their_definitions() {
        let raw = wobbly(40);
        let dt = 0.25;
        let bodies = [raw[0].as_slice(), raw[1].as_slice(), raw[2].as_slice()];
        let fold = |values: Vec<f64>| {
            let low = values.iter().copied().fold(f64::INFINITY, f64::min);
            let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (high - low).max(1e-12)
        };
        let axis = |pick: &dyn Fn(usize, usize) -> f64| {
            fold(
                (0..BODIES)
                    .flat_map(|b| (0..40).map(move |k| (b, k)))
                    .map(|(b, k)| pick(b, k))
                    .collect(),
            )
        };
        let v = |b: usize, k: usize| velocity(bodies[b], k, dt);
        let position_extent = axis(&|b, k| raw[b][k].x).max(axis(&|b, k| raw[b][k].y));
        let velocity_extent = axis(&|b, k| v(b, k)[0]).max(axis(&|b, k| v(b, k)[1]));
        let scale = position_extent / velocity_extent;

        // The last step repeats the one before it.
        assert_eq!(v(1, 39), v(1, 38));
        assert_eq!(v(1, 5)[0], (raw[1][6].x - raw[1][5].x) / dt);

        for projection in [
            ViewProjection::Position,
            ViewProjection::PhasePortrait,
            ViewProjection::CrossBraid,
            ViewProjection::Hodograph,
        ] {
            let view = View { projection, ..plain_view() };
            let velocity_scale = view.velocity_scale(bodies, dt);
            if projection == ViewProjection::Position {
                assert_eq!(velocity_scale, 1.0);
            } else {
                assert_eq!(velocity_scale.to_bits(), scale.to_bits());
            }
            for b in 0..BODIES {
                for k in [0, 7, 38, 39] {
                    let p = raw[b][k];
                    let w = v(b, k).map(|c| c * velocity_scale);
                    let expected = match projection {
                        ViewProjection::Position => [p.x, p.y, p.z],
                        ViewProjection::PhasePortrait => [p.x, w[0], p.y],
                        ViewProjection::CrossBraid => [p.x, raw[(b + 1) % 3][k].y, p.z],
                        ViewProjection::Hodograph => [w[0], w[1], p.z],
                    };
                    assert_eq!(view.projected(bodies, b, k, dt, velocity_scale), expected);
                }
            }
        }
    }

    #[test]
    fn the_eccentric_anomaly_solves_keplers_equation() {
        assert_eq!(eccentric_anomaly(1.234, 0.0), 1.234);
        let mut rng = Lcg(9);
        for _ in 0..2000 {
            let mean = rng.range(-PI, PI);
            let eccentricity = rng.range(0.0, 0.6);
            let anomaly = eccentric_anomaly(mean, eccentricity);
            let residual = anomaly - eccentricity * math::sin(anomaly) - mean;
            assert!(residual.abs() < 1e-12, "M {mean}, e {eccentricity}: {residual:e}");
        }
    }

    #[test]
    fn angles_wrap_into_half_a_turn_either_way() {
        for (angle, wrapped) in [(0.0, 0.0), (PI, PI), (-PI, -PI), (0.5, 0.5), (-0.5, -0.5)] {
            assert_eq!(wrap_angle(angle), wrapped);
        }
        let mut rng = Lcg(3);
        for _ in 0..2000 {
            let angle = rng.range(-40.0, 40.0);
            let wrapped = wrap_angle(angle);
            assert!((-PI..=PI).contains(&wrapped), "{angle} -> {wrapped}");
            let turns = (angle - wrapped) / TAU;
            assert!((turns - turns.round()).abs() < 1e-12, "{angle} -> {wrapped}");
        }
    }

    /// The drift adds the same offset to all three bodies: the linear one grows with the step,
    /// the elliptical one traces the rotated ellipse and is not zero at step 0.
    #[test]
    fn drift_offsets_follow_their_definitions() {
        let dt = 0.001;
        assert_eq!(ViewDrift::None {}.offset(123, dt), [0.0, 0.0]);
        let velocity = [0.3, -0.7, 5.0];
        let linear = ViewDrift::Linear { velocity };
        assert_eq!(linear.offset(0, dt), [0.0, -0.0]);
        assert_eq!(linear.offset(250, dt), [(0.3 * 250.0) * dt, (-0.7 * 250.0) * dt]);

        let (a, b, e) = (2.0, 1.5, 0.45);
        let drift = ViewDrift::Elliptical {
            rotation: tilted(),
            mean_anomaly: -1.3,
            mean_motion: 0.004,
            eccentricity: e,
            semi_major: a,
            semi_minor: b,
        };
        let start = drift.offset(0, dt);
        assert!(start[0].abs() + start[1].abs() > 0.1, "{start:?}");
        for step in [0usize, 1, 5000, 999_999] {
            let anomaly = eccentric_anomaly(wrap_angle(-1.3 + 0.004 * (step as f64 * dt)), e);
            let (x, y) = (a * (math::cos(anomaly) - e), b * math::sin(anomaly));
            let r = tilted();
            assert_eq!(
                drift.offset(step, dt),
                [r[0][0] * x + r[0][1] * y, r[1][0] * x + r[1][1] * y]
            );
        }
        // Applied: every body moves by the same offset (a tenth of it on a canvas that shows
        // `[-10, 10]²`, with y flipped).
        let raw = wobbly(50);
        let plain = square_view(10.0).canvas_track(&raw, dt, 1.0).unwrap();
        let drifted = View { drift, ..square_view(10.0) }.canvas_track(&raw, dt, 1.0).unwrap();
        for step in [0, 17, 49] {
            let o = drift.offset(step, dt);
            for body in 0..BODIES {
                let (p, d) = (plain[body][step], drifted[body][step]);
                assert!((d[0] - p[0] - 0.1 * o[0]).abs() < 1e-12, "{p:?} {d:?} {o:?}");
                assert!((d[1] - p[1] + 0.1 * o[1]).abs() < 1e-12, "{p:?} {d:?} {o:?}");
            }
        }
    }

    /// Golden digest of the canvas track's bits for every projection space under every drift,
    /// through a tilted rotation and a scaled frame: the whole recipe on the portable functions,
    /// so it must not move on any architecture.
    #[test]
    fn the_canvas_track_matches_the_golden_hash() {
        const GOLDEN: &str = "ee5fedc94df75effbe53a2294f9707294be4a3e3480c632dcf59d9801934bf30";
        let raw = wobbly(400);
        let drifts = [
            ViewDrift::None {},
            ViewDrift::Linear { velocity: [1.3, -0.9, 0.4] },
            ViewDrift::Elliptical {
                rotation: tilted(),
                mean_anomaly: 2.9,
                mean_motion: 4.0,
                eccentricity: 0.45,
                semi_major: 2.0,
                semi_minor: 1.5,
            },
        ];
        let mut hasher = Sha256::new();
        for projection in [
            ViewProjection::Position,
            ViewProjection::PhasePortrait,
            ViewProjection::CrossBraid,
            ViewProjection::Hodograph,
        ] {
            for drift in drifts {
                let frame = ViewFrame {
                    min_x: -30.0,
                    min_y: -20.0,
                    width: 60.0,
                    height: 40.0,
                    scale: 0.75,
                };
                let view = View { projection, rotation: tilted(), drift, frame };
                let track = view.canvas_track(&raw, 0.001, 1.5).unwrap();
                for point in track.iter().flatten() {
                    hasher.update(point[0].to_bits().to_le_bytes());
                    hasher.update(point[1].to_bits().to_le_bytes());
                }
            }
        }
        assert_eq!(hex::encode(hasher.finalize()), GOLDEN);
    }

    #[test]
    fn views_round_trip_through_json_bit_for_bit() {
        let odd = f64::from_bits(0x3fd5_5555_5555_5557);
        let views = [
            plain_view(),
            View {
                projection: ViewProjection::CrossBraid,
                rotation: tilted(),
                drift: ViewDrift::Linear { velocity: [odd, -0.0, 1e-300] },
                frame: ViewFrame {
                    min_x: -odd,
                    min_y: 2.5,
                    width: 7.25,
                    height: odd,
                    scale: 0.543,
                },
            },
            View {
                projection: ViewProjection::Hodograph,
                rotation: tilted(),
                drift: ViewDrift::Elliptical {
                    rotation: tilted(),
                    mean_anomaly: -odd,
                    mean_motion: 0.008_168_140_899_333_463,
                    eccentricity: 0.45,
                    semi_major: 3.0 * odd,
                    semi_minor: odd,
                },
                frame: plain_view().frame,
            },
        ];
        for view in views {
            let json = serde_json::to_string_pretty(&view).unwrap();
            let back: View = serde_json::from_str(&json).unwrap();
            assert_eq!(back, view, "{json}");
            assert_eq!(serde_json::to_string_pretty(&back).unwrap(), json);
        }
        let json = serde_json::to_value(plain_view()).unwrap();
        assert_eq!(json["projection"], "position");
        assert_eq!(json["drift"]["mode"], "none");
        assert_eq!(json["rotation"][0][0], "0x3ff0000000000000");
        assert_eq!(json["frame"]["width"], "0x4000000000000000");
        // Unknown keys and decimal numbers are rejected: the view is recorded as bits only.
        let mut extra = json.clone();
        extra["frame"]["fill"] = 0.78.into();
        assert!(serde_json::from_value::<View>(extra).is_err());
        let mut decimal = json.clone();
        decimal["frame"]["width"] = 2.0.into();
        assert!(serde_json::from_value::<View>(decimal).is_err());
        // A drift without fields is as strict as the others: a leftover of another mode next to
        // `"mode": "none"` is an input this view would ignore.
        assert_eq!(json["drift"], serde_json::json!({ "mode": "none" }));
        for (key, value) in
            [("velocity", serde_json::json!(["0x0", "0x0", "0x0"])), ("scale", 1.into())]
        {
            let mut leftover = json.clone();
            leftover["drift"][key] = value;
            assert!(serde_json::from_value::<View>(leftover).is_err(), "{key}");
        }
    }

    #[test]
    fn views_the_edition_cannot_follow_are_rejected() {
        let invalid = |edit: &dyn Fn(&mut View)| {
            let mut view = View {
                drift: ViewDrift::Elliptical {
                    rotation: IDENTITY,
                    mean_anomaly: 0.1,
                    mean_motion: 0.01,
                    eccentricity: 0.4,
                    semi_major: 1.0,
                    semi_minor: 0.9,
                },
                ..plain_view()
            };
            edit(&mut view);
            matches!(view.validate(), Err(EmberError::InvalidView { .. }))
        };
        assert!(!invalid(&|_| {}));
        assert!(invalid(&|v| v.rotation[1][2] = f64::NAN));
        assert!(invalid(&|v| v.frame.width = 0.0));
        assert!(invalid(&|v| v.frame.height = -1.0));
        assert!(invalid(&|v| v.frame.min_x = f64::INFINITY));
        assert!(invalid(&|v| v.frame.scale = 0.0));
        assert!(invalid(&|v| v.frame.scale = 1.5));
        assert!(invalid(&|v| v.drift = ViewDrift::Linear { velocity: [0.0, f64::NAN, 0.0] }));
        for (eccentricity, semi_major) in [(1.0, 1.0), (-0.1, 1.0), (0.4, -1.0), (f64::NAN, 1.0)] {
            assert!(invalid(&|v| {
                if let ViewDrift::Elliptical { eccentricity: e, semi_major: a, .. } = &mut v.drift {
                    (*e, *a) = (eccentricity, semi_major);
                }
            }));
        }

        let raw = wobbly(10);
        let view = frontal_view(&raw, 1.5);
        let track = |raw: &[Vec<Vector3<f64>>], dt, aspect| view.canvas_track(raw, dt, aspect);
        assert!(track(&raw, 0.001, 1.5).is_ok());
        // A time step or canvas that is not one, and a frame of another aspect than the canvas.
        for (dt, aspect) in
            [(0.0, 1.5), (f64::NAN, 1.5), (0.001, 0.0), (0.001, f64::INFINITY), (0.001, 1.6)]
        {
            assert!(matches!(track(&raw, dt, aspect), Err(EmberError::InvalidView { .. })));
        }
        // A view that frames another orbit: this one leaves its canvas.
        let elsewhere: Vec<Vec<Vector3<f64>>> = raw
            .iter()
            .map(|body| body.iter().map(|p| p + Vector3::new(40.0, 0.0, 0.0)).collect())
            .collect();
        let Err(EmberError::InvalidView { reason }) = track(&elsewhere, 0.001, 1.5) else {
            panic!("an orbit outside the frame must be rejected");
        };
        assert!(reason.contains("leaves the canvas"), "{reason}");
        let degenerate = |raw: &[Vec<Vector3<f64>>]| {
            matches!(track(raw, 0.001, 1.5), Err(EmberError::DegenerateOrbit { .. }))
        };
        assert!(degenerate(&raw[..2]));
        let mut unequal = raw.clone();
        unequal[1].pop();
        assert!(degenerate(&unequal));
        assert!(degenerate(&raw.iter().map(|body| body[..1].to_vec()).collect::<Vec<_>>()));
        let mut non_finite = raw.clone();
        non_finite[2][4].y = f64::NAN;
        assert!(degenerate(&non_finite));
    }
}
