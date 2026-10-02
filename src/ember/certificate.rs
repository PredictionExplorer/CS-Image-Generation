//! The determinism certificate `metadata/ember.json`: writing it and reading it back.
//!
//! The certificate records everything the ember frames are a function of — the selected orbit's
//! initial conditions and the main edition's view of it (both as exact bit patterns), the
//! integration settings, the frame schedule, the paper seed and the full [`EmberConfig`] —
//! together with SHA-256 digests of the raw frame streams (the film and the slow film) and of
//! the still's pixels. Rendering the same inputs on any CPU architecture must reproduce every
//! digest exactly. Encoded containers (MP4, WebP, PNG) are derived artefacts: their bytes
//! also depend on encoder versions, so they are not part of the contract. The `build` and
//! `timings_seconds` sections describe the machine that produced the package and are
//! informational only. The layout and the verification procedure are described in
//! `docs/ember-edition.md` and `docs/ember-design.md` §8.
//!
//! # Digests
//!
//! All digests are lowercase hex SHA-256:
//!
//! * `inputs.frames.sha256` — the frame schedule as little-endian `u64` knot indices
//!   ([`schedule_sha256`]);
//! * `inputs.paper_seed_sha256` — the paper-seed bytes ([`paper_seed_sha256`]);
//! * `outputs.frames_rgb48le_sha256`, `outputs.slow_frames_rgb48le_sha256`,
//!   `outputs.still_rgb48le_sha256` — the pixels as `rgb48le` (16-bit little-endian R, G, B per
//!   pixel, row-major from the top-left pixel; the frames of a film concatenated in the order
//!   they are shown).
//!
//! # Reading a certificate back
//!
//! [`EmberCertificate::read_json`] and [`EmberCertificate::from_json`] parse a certificate into
//! the same types it was written from: reading back a written certificate gives a value equal
//! to it, every float bit for bit (the crate enables `serde_json`'s `float_roundtrip`, which
//! parses every decimal exactly). The reader is strict: it rejects other layout versions
//! ([`CERTIFICATE_SCHEMA_VERSION`]) and editions, missing fields — the nullable ones
//! (`outputs.frames_rgb48le_sha256`, `outputs.slow_frames_rgb48le_sha256`,
//! `config.look.floor_tau`) included, which must be present, as `null` or a value — unknown
//! fields at every level (a field this build does not understand could be an input it would
//! silently ignore), and outputs that disagree about the frames
//! ([`CertificateError::InconsistentOutputs`]: frames emitted without a frames digest, which
//! would pass as a still-only certificate, a frames digest over another number of frames than
//! the film has, or a slow film without the normal one).
//!
//! The initial conditions are also written, and read back, as exact IEEE-754 bit patterns
//! ([`F64Bits`]); [`CertificateInputs::bodies`] rebuilds them from those, never from the decimal
//! copies next to them. The view is written as bit patterns only. The decimals are for people; the bit patterns stay authoritative for
//! readers whose JSON parser is not exact.

use std::fmt;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use nalgebra::Vector3;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

use super::config::EmberConfig;
use super::pipeline::{EmberStats, EmberSummary, EmberTimings};
use super::view::View;
use crate::sim::Body;

/// Version of the certificate layout; the reader accepts only this version. Bump it whenever the
/// layout changes (a field added, removed, renamed or retyped, or an optional one made
/// required), so that a reader reports a certificate of another layout as
/// [`CertificateError::UnsupportedSchema`] rather than as a malformed file.
///
/// * 1 — the first layout (test renders only; never published).
/// * 2 — adds `stats.frames_with_cinnabar` and `stats.peak_frame_cinnabar_fraction`, and
///   requires the nullable keys `outputs.frames_rgb48le_sha256`, `config.look.floor_tau` and
///   `config.look.ember_tau` to be present (as `null` or a value). Published with `ember-v1`.
/// * 3 — the sumi edition with tidal bodies (`ember-v2`): no vermilion, so the cinnabar
///   statistics and the look's vermilion settings are gone; adds the `tidal` configuration and
///   `derived.hold_time`, `derived.fade_time` and `derived.tidal_reference`.
/// * 4 — the edition follows the main edition's view and gains the slow film (`ember-v3`): adds
///   `inputs.view`, `inputs.frames.slow_factor`, `derived.slow_first_frame`,
///   `outputs.slow_frames_rgb48le_sha256` and `outputs.slow_frames_emitted`; the principal-plane
///   projection (`config.projection`, `derived.projection`) is gone.
pub const CERTIFICATE_SCHEMA_VERSION: u32 = 4;

/// Version of the rendering algorithm, `ember-v<N>`. Bump `N` whenever a change alters rendered
/// bits; the sync loop (`run.py`) withdraws and re-renders every published edition of an older
/// version (`<generator> --ember-algorithm` prints this value).
///
/// * `ember-v1` — sumi and vermilion on kozo, disc bodies.
/// * `ember-v2` — sumi on kozo, tidally stretched bodies, the fade timed in film time.
/// * `ember-v3` — the bodies follow the main edition's view (projection space, viewing rotation,
///   drift and frame); a slow film; the soak frame is symmetric.
pub const ALGORITHM_VERSION: &str = "ember-v3";

/// The certificate's `edition`.
pub const EDITION: &str = "ember";

/// How the recorded orbit was integrated (the certificate's `inputs.integrator`).
pub const INTEGRATOR: &str = "yoshida4-f64 (warm-up of `steps`, then `steps` recorded knots)";

/// Colour encoding of the digested pixels (the certificate's `outputs.encoding`).
pub const PIXEL_ENCODING: &str =
    "sRGB (IEC 61966-2-1) 16-bit, CAT16-adapted from the gallery LED-V1 illuminant to D65";

/// The statement the digests certify.
const CONTRACT: &str = "outputs.frames_rgb48le_sha256, outputs.slow_frames_rgb48le_sha256 and \
outputs.still_rgb48le_sha256 are a pure function of `inputs` and `config`: rendering them again \
on any IEEE-754 CPU (x86_64, aarch64, any thread count) reproduces every digest bit for bit. \
MP4, WebP and PNG files are encodings of these pixels whose bytes also depend on encoder \
versions.";

/// The full certificate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmberCertificate {
    /// Layout version ([`CERTIFICATE_SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// Always [`EDITION`] (`"ember"`).
    pub edition: String,
    /// Rendering algorithm version ([`ALGORITHM_VERSION`] of the producing build).
    pub algorithm: String,
    /// What is certified.
    pub contract: String,
    /// Inputs the frames are a function of.
    pub inputs: CertificateInputs,
    /// Look and simulation parameters.
    pub config: EmberConfig,
    /// Quantities derived from the inputs (deterministic).
    pub derived: CertificateDerived,
    /// Digests of the rendered pixels.
    pub outputs: CertificateOutputs,
    /// Deterministic render statistics.
    pub stats: EmberStats,
    /// Machine that produced this package (informational).
    pub build: CertificateBuild,
    /// Wall-clock timings in seconds (informational).
    pub timings_seconds: EmberTimings,
}

/// Inputs of the render.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertificateInputs {
    /// Seed of the package (hex, as given on the command line).
    pub seed: String,
    /// Recorded orbit knots (the still shows knot `steps - 1`).
    pub steps: usize,
    /// Integrator time step.
    pub dt: f64,
    /// Gravitational constant.
    pub gravitational_constant: f64,
    /// Integrator used to record the orbit ([`INTEGRATOR`] of the producing build).
    pub integrator: String,
    /// Initial conditions of the selected orbit (before the centre-of-mass shift).
    pub bodies: Vec<BodyRecord>,
    /// The main edition's view of the orbit, which the bodies follow (exact bit patterns).
    pub view: View,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// SHA-256 (hex) of the paper seed bytes ([`paper_seed_sha256`]).
    pub paper_seed_sha256: String,
    /// The frame schedule.
    pub frames: FrameScheduleRecord,
}

impl CertificateInputs {
    /// The recorded initial conditions, bit for bit (rebuilt from the bit patterns, never from
    /// the decimals).
    pub fn bodies(&self) -> Vec<Body> {
        self.bodies.iter().map(BodyRecord::body).collect()
    }
}

/// One body's initial conditions, as decimals (for people) and exact IEEE-754 bit patterns (for
/// machines; authoritative).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BodyRecord {
    /// Mass.
    pub mass: f64,
    /// Position.
    pub position: [f64; 3],
    /// Velocity.
    pub velocity: [f64; 3],
    /// Exact bit patterns of mass, position and velocity.
    pub bits: BodyBits,
}

impl BodyRecord {
    /// The body these initial conditions describe, rebuilt bit for bit from [`Self::bits`].
    pub fn body(&self) -> Body {
        let [px, py, pz] = self.bits.position.map(F64Bits::value);
        let [vx, vy, vz] = self.bits.velocity.map(F64Bits::value);
        Body::new(self.bits.mass.value(), Vector3::new(px, py, pz), Vector3::new(vx, vy, vz))
    }
}

impl From<&Body> for BodyRecord {
    fn from(body: &Body) -> Self {
        let position = [body.position.x, body.position.y, body.position.z];
        let velocity = [body.velocity.x, body.velocity.y, body.velocity.z];
        Self {
            mass: body.mass,
            position,
            velocity,
            bits: BodyBits {
                mass: F64Bits::of(body.mass),
                position: position.map(F64Bits::of),
                velocity: velocity.map(F64Bits::of),
            },
        }
    }
}

/// Exact bit patterns of a [`BodyRecord`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BodyBits {
    /// Mass bits.
    pub mass: F64Bits,
    /// Position bits.
    pub position: [F64Bits; 3],
    /// Velocity bits.
    pub velocity: [F64Bits; 3],
}

/// The exact IEEE-754 bit pattern of an `f64` (`f64::to_bits`), written as `0x` followed by 16
/// lowercase hex digits (`"0x3ff0000000000000"` is 1.0). Reading accepts either case but
/// requires the prefix and all 16 digits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct F64Bits(pub u64);

impl F64Bits {
    /// The bit pattern of `value`.
    pub fn of(value: f64) -> Self {
        Self(value.to_bits())
    }

    /// The `f64` with this bit pattern (`f64::from_bits`; exact, NaN payloads included).
    pub fn value(self) -> f64 {
        f64::from_bits(self.0)
    }
}

impl fmt::Display for F64Bits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#018x}", self.0)
    }
}

impl FromStr for F64Bits {
    type Err = CertificateError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let invalid = || CertificateError::BitPattern { found: text.to_owned() };
        let digits = text.strip_prefix("0x").ok_or_else(invalid)?;
        // `from_str_radix` alone would also accept a sign and fewer digits.
        if digits.len() != 16 || !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(invalid());
        }
        u64::from_str_radix(digits, 16).map(Self).map_err(|_| invalid())
    }
}

impl Serialize for F64Bits {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for F64Bits {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

/// Summary of the frame schedule.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrameScheduleRecord {
    /// Scheduled frames.
    pub count: usize,
    /// First scheduled knot (0 for an empty schedule).
    pub first_step: usize,
    /// Last scheduled knot, the still (0 for an empty schedule).
    pub last_step: usize,
    /// Frames per second of the encoded videos.
    pub frame_rate: u32,
    /// [`schedule_sha256`] of the schedule.
    pub sha256: String,
    /// How many times slower the slow film is: it has this many frames per scheduled interval,
    /// and the snapshot lattice of every render is a multiple of it.
    pub slow_factor: u32,
}

impl FrameScheduleRecord {
    /// The record of `frame_steps` shown at `frame_rate` frames per second, with a slow film
    /// `slow_factor` times slower.
    pub fn new(frame_steps: &[usize], frame_rate: u32, slow_factor: u32) -> Self {
        Self {
            count: frame_steps.len(),
            first_step: frame_steps.first().copied().unwrap_or(0),
            last_step: frame_steps.last().copied().unwrap_or(0),
            frame_rate,
            sha256: schedule_sha256(frame_steps),
            slow_factor,
        }
    }
}

/// Deterministic quantities derived from the inputs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertificateDerived {
    /// Orbit duration in fluid time units.
    pub duration: f64,
    /// Fluid time at which the bodies stopped inking.
    pub valve_time: f64,
    /// Time for which fresh ink stays black, in fluid units.
    pub hold_time: f64,
    /// E-folding time of the fade, in fluid units.
    pub fade_time: f64,
    /// The orbit's reference tidal anisotropy, which scales the bodies' stretch.
    pub tidal_reference: f64,
    /// Fluid grid `[nx, ny]`.
    pub fluid_grid: [usize; 2],
    /// Fluid grid spacing in world units.
    pub fluid_dx: f64,
    /// Ink node grid `[cols, rows]`, margin included.
    pub ink_grid: [usize; 2],
    /// The scheduled frame at which the slow film starts (a little before the ink appears).
    pub slow_first_frame: usize,
}

/// Digests of the rendered pixels.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertificateOutputs {
    /// SHA-256 of the concatenated `rgb48le` frames (`null` for still-only renders). Required
    /// when reading: a certificate without the key is rejected rather than read as still-only.
    #[serde(deserialize_with = "Option::deserialize")]
    pub frames_rgb48le_sha256: Option<String>,
    /// Frames in the stream: 0 for still-only renders, every scheduled frame
    /// (`inputs.frames.count`) for video renders. The reader checks both.
    pub frames_emitted: usize,
    /// SHA-256 of the slow film's concatenated `rgb48le` frames (`null` unless it was rendered).
    /// Required when reading, like the normal film's.
    #[serde(deserialize_with = "Option::deserialize")]
    pub slow_frames_rgb48le_sha256: Option<String>,
    /// Frames in the slow film's stream: 0 unless it was rendered, else every scheduled frame
    /// from `derived.slow_first_frame` on and `inputs.frames.slow_factor - 1` frames between
    /// each two of them. The reader checks both.
    pub slow_frames_emitted: usize,
    /// SHA-256 of the still as `rgb48le`.
    pub still_rgb48le_sha256: String,
    /// Colour encoding of all three ([`PIXEL_ENCODING`]).
    pub encoding: String,
}

impl CertificateOutputs {
    /// Checks that the outputs agree about the frames, with each other and with the frame
    /// `schedule`: a still-only render records neither a frames digest nor frames (`null`, 0),
    /// a video render a digest over every scheduled frame. What a verifier checks follows from
    /// the digest, so a certificate with frames but no digest would pass as still-only.
    ///
    /// The slow film follows the same rule against its own length,
    /// `(count - 1 - slow_first_frame)·slow_factor + 1`, and is only ever rendered together
    /// with the normal film.
    fn check_frames(
        &self,
        schedule: &FrameScheduleRecord,
        slow_first_frame: usize,
    ) -> Result<(), CertificateError> {
        // `None` for numbers no film has (a first frame beyond the schedule, an overflow).
        let slow_frames = slow_first_frame
            .checked_add(1)
            .and_then(|skipped| schedule.count.checked_sub(skipped))
            .and_then(|intervals| intervals.checked_mul(schedule.slow_factor as usize))
            .and_then(|between| between.checked_add(1));
        let reason = match (&self.frames_rgb48le_sha256, self.frames_emitted) {
            (None, 0) => None,
            (None, emitted) => Some(format!(
                "outputs.frames_rgb48le_sha256 is null, but outputs.frames_emitted is {emitted} (a \
                 still-only certificate records 0 frames); the frames could not be verified"
            )),
            (Some(_), emitted) if emitted != schedule.count => Some(format!(
                "outputs.frames_emitted is {emitted}, but a video render emits every one of the \
                 {} scheduled frames (inputs.frames.count)",
                schedule.count
            )),
            (Some(_), _) => None,
        }
        .or_else(|| match (&self.slow_frames_rgb48le_sha256, self.slow_frames_emitted) {
            (None, 0) => None,
            (None, emitted) => Some(format!(
                "outputs.slow_frames_rgb48le_sha256 is null, but outputs.slow_frames_emitted is \
                 {emitted} (a certificate without the slow film records 0 frames)"
            )),
            (Some(_), _) if self.frames_rgb48le_sha256.is_none() => Some(
                "outputs.slow_frames_rgb48le_sha256 is set, but outputs.frames_rgb48le_sha256 is \
                 null (the slow film is rendered together with the normal one)"
                    .to_owned(),
            ),
            (Some(_), emitted) if Some(emitted) != slow_frames => Some(format!(
                "outputs.slow_frames_emitted is {emitted}, but the slow film of {} scheduled \
                 frames from frame {slow_first_frame} at factor {} has {} frames",
                schedule.count,
                schedule.slow_factor,
                slow_frames.map_or("no".to_owned(), |frames| frames.to_string())
            )),
            (Some(_), _) => None,
        });
        reason.map_or(Ok(()), |reason| Err(CertificateError::InconsistentOutputs { reason }))
    }
}

/// The producing machine.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertificateBuild {
    /// Crate version.
    pub crate_version: String,
    /// CPU architecture (`std::env::consts::ARCH`).
    pub target_arch: String,
    /// Operating system (`std::env::consts::OS`).
    pub target_os: String,
    /// Rayon worker threads of the render.
    pub threads: usize,
}

impl CertificateBuild {
    /// This build on this machine, with the current rayon pool's thread count.
    fn current() -> Self {
        Self {
            crate_version: env!("CARGO_PKG_VERSION").to_owned(),
            target_arch: std::env::consts::ARCH.to_owned(),
            target_os: std::env::consts::OS.to_owned(),
            threads: rayon::current_num_threads(),
        }
    }
}

/// Inputs of [`EmberCertificate::new`] that are not in the summary.
#[derive(Clone, Copy, Debug)]
pub struct CertificateContext<'a> {
    /// Seed of the package (hex).
    pub seed: &'a str,
    /// Recorded orbit knots.
    pub steps: usize,
    /// Integrator time step.
    pub dt: f64,
    /// Initial conditions of the selected orbit.
    pub bodies: &'a [Body],
    /// The main edition's view of the orbit.
    pub view: &'a View,
    /// The frame schedule.
    pub frame_steps: &'a [usize],
    /// Frames per second of the encoded videos.
    pub frame_rate: u32,
    /// Paper seed bytes.
    pub paper_seed: &'a [u8],
    /// Configuration used for the render.
    pub config: &'a EmberConfig,
}

/// Why a certificate could not be read.
#[derive(Debug, thiserror::Error)]
pub enum CertificateError {
    /// The file could not be read.
    #[error("cannot read the certificate {}: {source}", path.display())]
    Io {
        /// The certificate's path.
        path: PathBuf,
        /// The I/O error.
        source: io::Error,
    },

    /// Not JSON, or not the certificate layout (a missing, mistyped or unknown field, or a
    /// malformed bit pattern; the message names it and its position).
    #[error("malformed ember certificate: {0}")]
    Json(#[from] serde_json::Error),

    /// A layout version this build cannot read.
    #[error(
        "ember certificate schema version {found} is not supported (this build reads version \
         {supported})",
        supported = CERTIFICATE_SCHEMA_VERSION
    )]
    UnsupportedSchema {
        /// The certificate's `schema_version`.
        found: u32,
    },

    /// The outputs disagree about the frames: frames emitted without a frames digest, or a
    /// frames digest over another number of frames than `inputs.frames.count`.
    #[error("inconsistent ember certificate: {reason}")]
    InconsistentOutputs {
        /// What disagrees, naming the fields and their values.
        reason: String,
    },

    /// Not an ember certificate.
    #[error("the certificate's edition is {found:?}, not {expected:?}", expected = EDITION)]
    WrongEdition {
        /// The certificate's `edition`.
        found: String,
    },

    /// A bit pattern that is not `0x` followed by 16 hex digits.
    #[error("{found:?} is not an f64 bit pattern (`0x` followed by 16 hex digits)")]
    BitPattern {
        /// The offending text.
        found: String,
    },
}

/// SHA-256 (lowercase hex) of a frame schedule: the knot indices as little-endian `u64`s,
/// concatenated in schedule order (the certificate's `inputs.frames.sha256`).
pub fn schedule_sha256(frame_steps: &[usize]) -> String {
    let mut hasher = Sha256::new();
    for &step in frame_steps {
        hasher.update((step as u64).to_le_bytes());
    }
    hex::encode(hasher.finalize())
}

/// SHA-256 (lowercase hex) of the paper-seed bytes (the certificate's
/// `inputs.paper_seed_sha256`).
pub fn paper_seed_sha256(paper_seed: &[u8]) -> String {
    hex::encode(Sha256::digest(paper_seed))
}

impl EmberCertificate {
    /// Assembles the certificate of a finished render.
    pub fn new(context: &CertificateContext<'_>, summary: &EmberSummary) -> Self {
        Self {
            schema_version: CERTIFICATE_SCHEMA_VERSION,
            edition: EDITION.to_owned(),
            algorithm: ALGORITHM_VERSION.to_owned(),
            contract: CONTRACT.to_owned(),
            inputs: CertificateInputs {
                seed: context.seed.to_owned(),
                steps: context.steps,
                dt: context.dt,
                gravitational_constant: crate::sim::G,
                integrator: INTEGRATOR.to_owned(),
                bodies: context.bodies.iter().map(BodyRecord::from).collect(),
                view: *context.view,
                width: summary.width,
                height: summary.height,
                paper_seed_sha256: paper_seed_sha256(context.paper_seed),
                frames: FrameScheduleRecord::new(
                    context.frame_steps,
                    context.frame_rate,
                    summary.slow_factor,
                ),
            },
            config: context.config.clone(),
            derived: CertificateDerived {
                duration: summary.duration,
                valve_time: summary.valve_time,
                hold_time: summary.hold_time,
                fade_time: summary.fade_time,
                tidal_reference: summary.tidal_reference,
                fluid_grid: summary.fluid_grid,
                fluid_dx: summary.fluid_dx,
                ink_grid: summary.ink_grid,
                slow_first_frame: summary.slow_first_frame,
            },
            outputs: CertificateOutputs {
                frames_rgb48le_sha256: summary.frames_sha256.clone(),
                frames_emitted: summary.frames_emitted,
                slow_frames_rgb48le_sha256: summary.slow_frames_sha256.clone(),
                slow_frames_emitted: summary.slow_frames_emitted,
                still_rgb48le_sha256: summary.still_sha256.clone(),
                encoding: PIXEL_ENCODING.to_owned(),
            },
            stats: summary.stats,
            build: CertificateBuild::current(),
            timings_seconds: summary.timings,
        }
    }

    /// Writes the certificate as pretty-printed JSON (with a final newline), then flushes the
    /// buffer and syncs the file explicitly, so that every write error — including one that
    /// surfaces only when the buffer is flushed or the data reaches the disk — is returned
    /// instead of being lost when the writer is dropped. After an error the file may be partial.
    pub fn write_json(&self, path: &Path) -> io::Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer_pretty(&mut writer, self).map_err(io::Error::other)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.into_inner().map_err(io::IntoInnerError::into_error)?.sync_all()
    }

    /// Reads a certificate file (see [`Self::from_json`]).
    pub fn read_json(path: &Path) -> Result<Self, CertificateError> {
        let text = std::fs::read_to_string(path)
            .map_err(|source| CertificateError::Io { path: path.to_owned(), source })?;
        Self::from_json(&text)
    }

    /// Parses a certificate: its layout version must be [`CERTIFICATE_SCHEMA_VERSION`] and its
    /// edition [`EDITION`], every field must be present and known, every bit pattern
    /// well-formed, and the outputs must agree about the frames
    /// ([`CertificateError::InconsistentOutputs`]). Nothing else is checked here: whether this
    /// build can reproduce the certificate (same algorithm, integrator, constants and schedule)
    /// is the verifier's question (`examples/ember_render.rs`).
    pub fn from_json(text: &str) -> Result<Self, CertificateError> {
        /// The fields that identify the layout, read first so that a certificate of another
        /// version is reported as such rather than as a malformed one.
        #[derive(Deserialize)]
        struct Header {
            /// See [`EmberCertificate::schema_version`].
            schema_version: u32,
            /// See [`EmberCertificate::edition`].
            edition: String,
        }
        let header: Header = serde_json::from_str(text)?;
        if header.schema_version != CERTIFICATE_SCHEMA_VERSION {
            return Err(CertificateError::UnsupportedSchema { found: header.schema_version });
        }
        if header.edition != EDITION {
            return Err(CertificateError::WrongEdition { found: header.edition });
        }
        let certificate: Self = serde_json::from_str(text)?;
        certificate
            .outputs
            .check_frames(&certificate.inputs.frames, certificate.derived.slow_first_frame)?;
        Ok(certificate)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;
    use crate::ember::view::{ViewDrift, ViewFrame, ViewProjection};

    /// Initial conditions with awkward decimals: 17 significant digits (one of which `serde_json`
    /// without `float_roundtrip` reads one unit low), signed zero, tiny and large magnitudes.
    fn awkward_bodies() -> Vec<Body> {
        vec![
            Body::new(
                0.1 + 0.2,
                Vector3::new(-0.0, 1.0 / 3.0, 2.0f64.sqrt()),
                Vector3::new(1e-300, -2.5e15 + 0.5, 0.970_004_360_123_456_7),
            ),
            Body::new(
                1.0,
                Vector3::new(0.970_004_36, -0.243_087_53, 0.05),
                Vector3::new(-0.466_203_685, -0.432_365_73, 0.0),
            ),
            Body::new(
                7.0 / 9.0,
                Vector3::new(-1.0 / 7.0, 123_456.789, -9.87e-7),
                Vector3::new(3.0f64.sqrt() / 1e6, 0.1 * 3.0, -1.0 / 3.0),
            ),
        ]
    }

    /// The main video's schedule rule for 3,000 steps every 150: 150, 300, …, 2850, 2999.
    fn schedule() -> Vec<usize> {
        let mut steps: Vec<usize> = (150..3_000).step_by(150).collect();
        steps.push(2_999);
        steps
    }

    /// A render summary whose floats are arbitrary 17-significant-digit values, as real renders
    /// produce (`3.7726531386560076` is a real `stats.max_flow_speed` that `serde_json` without
    /// `float_roundtrip` read back as `3.772653138656008`), with a signed zero and a subnormal.
    fn summary() -> EmberSummary {
        let third = 1.0 / 3.0;
        EmberSummary {
            width: 96,
            height: 64,
            still: vec![0; 96 * 64 * 3],
            still_sha256: "ab".repeat(32),
            frames_emitted: 20,
            frames_sha256: Some("cd".repeat(32)),
            slow_factor: 10,
            slow_first_frame: 2,
            slow_frames_emitted: 171,
            slow_frames_sha256: Some("ef".repeat(32)),
            duration: 8.779_257_088_198_804,
            valve_time: 8.779_257_088_198_804 - 0.2,
            hold_time: 8.779_257_088_198_804 * 0.8 / 30.0,
            fade_time: 8.779_257_088_198_804 * 0.025,
            tidal_reference: third * 1.7e3,
            fluid_grid: [90, 64],
            fluid_dx: 2.0 * 1.35 / 64.0 * third * 3.0,
            ink_grid: [226, 162],
            stats: EmberStats {
                fluid_steps: 1_234,
                min_dt: 0.001 + third * 7e-4,
                max_dt: 0.01 - 1e-18,
                max_flow_speed: 3.772_653_138_656_007_6,
                snapshots: 880,
                contact_events: 4_321,
                still_ink_fraction: 13_579.0 / 24_576.0 * third,
                still_gamut_mapped_pixels: 7,
            },
            timings: EmberTimings {
                fluid_seconds: 1_004.0 + third / 3.0,
                ink_seconds: third * 801.0,
                shade_seconds: 36.0 / 7.0,
                sink_seconds: 97.5 + third,
                total_seconds: 1_405.6 * (1.0 + 1e-16),
            },
        }
    }

    /// A view with awkward numbers in every field: thirds, a signed zero and a subnormal.
    fn view() -> View {
        let third = 1.0 / 3.0;
        View {
            projection: ViewProjection::PhasePortrait,
            rotation: [
                [0.6 + 1e-16, -0.0, third],
                [-third, 0.970_004_360_123_456_7, 5e-324],
                [0.0, 1.0, 2.0f64.sqrt()],
            ],
            drift: ViewDrift::Elliptical {
                rotation: [[third, 0.0, 1.0], [0.25, -third, 0.5], [1.0, 0.0, 0.0]],
                mean_anomaly: -third * 7.0,
                mean_motion: 0.008_168_140_899_333_463,
                eccentricity: 0.4 + third / 10.0,
                semi_major: 3.5 + 1e-15,
                semi_minor: 3.0 * third,
            },
            frame: ViewFrame {
                min_x: -1.547 - third * 1e-4,
                min_y: 0.1 + 0.2,
                width: 7.25 * third,
                height: 1e-300 / 3.0,
                scale: 0.542_863_189_697_265_6,
            },
        }
    }

    /// A non-default configuration (the optional field exercised).
    fn config() -> EmberConfig {
        let mut config = EmberConfig::default();
        config.look.fade_fraction = 0.25 / 7.0;
        config.look.hold_fraction = 0.1;
        config.look.floor_tau = Some(3.0);
        config.tidal.max_aspect = 2.5;
        config
    }

    /// The certificate of a video render of `bodies` with `config` ([`summary`]).
    fn certificate(bodies: &[Body], config: &EmberConfig) -> EmberCertificate {
        certificate_of(bodies, config, &summary())
    }

    /// The certificate of `summary`, a render of `bodies` with `config` on [`schedule`].
    fn certificate_of(
        bodies: &[Body],
        config: &EmberConfig,
        summary: &EmberSummary,
    ) -> EmberCertificate {
        let steps = schedule();
        let context = CertificateContext {
            seed: "0x46205528",
            steps: 3_000,
            dt: crate::render::constants::DEFAULT_DT,
            bodies,
            view: &view(),
            frame_steps: &steps,
            frame_rate: 60,
            paper_seed: b"abc",
            config,
        };
        EmberCertificate::new(&context, summary)
    }

    /// Replaces the decimal copies of the initial conditions by their bit patterns' values.
    fn decimals_from_bits(mut certificate: EmberCertificate) -> EmberCertificate {
        for record in &mut certificate.inputs.bodies {
            record.mass = record.bits.mass.value();
            record.position = record.bits.position.map(F64Bits::value);
            record.velocity = record.bits.velocity.map(F64Bits::value);
        }
        certificate
    }

    fn body_bits(body: &Body) -> [u64; 7] {
        let (p, v) = (body.position, body.velocity);
        [body.mass, p.x, p.y, p.z, v.x, v.y, v.z].map(f64::to_bits)
    }

    #[test]
    fn bit_patterns_round_trip_every_f64_exactly() {
        let values = [
            0.1 + 0.2,
            -0.0,
            5e-324,
            f64::MIN_POSITIVE,
            -f64::MAX,
            f64::INFINITY,
            f64::NAN,
            f64::from_bits(0x7ff8_dead_beef_0001),
        ];
        for value in values {
            let json = serde_json::to_string(&F64Bits::of(value)).expect("serialises");
            let back: F64Bits = serde_json::from_str(&json).expect("parses");
            assert_eq!(back.value().to_bits(), value.to_bits(), "{json}");
        }
        assert_eq!(
            serde_json::to_string(&F64Bits::of(1.0)).expect("serialises"),
            "\"0x3ff0000000000000\""
        );
        assert_eq!("0x3FF0000000000000".parse::<F64Bits>().expect("upper case").value(), 1.0);
    }

    #[test]
    fn malformed_bit_patterns_are_rejected() {
        for text in [
            "3ff0000000000000",
            "0x3ff",
            "0x3ff00000000000000",
            "0x3ff000000000000g",
            "0x+ff0000000000000",
            "0X3ff0000000000000",
            "",
        ] {
            assert!(
                matches!(text.parse::<F64Bits>(), Err(CertificateError::BitPattern { .. })),
                "{text:?} must be rejected"
            );
            assert!(serde_json::from_value::<F64Bits>(Value::from(text)).is_err(), "{text:?}");
        }
        assert!(serde_json::from_str::<F64Bits>("4607182418800017408").is_err(), "not a string");
    }

    #[test]
    fn schedule_digest_hashes_little_endian_u64_knots() {
        // Independent oracle: Python's hashlib over struct.pack('<Q', …) of the same knots.
        assert_eq!(
            schedule_sha256(&[]),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            schedule_sha256(&[0, 1, 255, 256, 65_536, 4_294_967_296]),
            "4d4a1bc0c4fe7f9e215ebbdc6842ec49a1d62239105631c154dd4e11be40b436"
        );
        assert_eq!(
            schedule_sha256(&schedule()),
            "122527417c8f79a93d4697ff39ac74e11cbfffdeeaca9ac8f4d6d3ea3170cb6f"
        );
        let record = FrameScheduleRecord::new(&schedule(), 60, 10);
        assert_eq!((record.count, record.first_step, record.last_step), (20, 150, 2_999));
        assert_eq!((record.frame_rate, record.slow_factor), (60, 10));
        assert_eq!(record.sha256, schedule_sha256(&schedule()));
    }

    #[test]
    fn paper_seed_digest_is_plain_sha256() {
        // FIPS 180-2 test vector.
        assert_eq!(
            paper_seed_sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn certificate_round_trips_through_its_file() {
        let bodies = awkward_bodies();
        let config = config();
        let written = certificate(&bodies, &config);
        let dir = tempfile::tempdir().expect("temporary directory");
        let path = dir.path().join("ember.json");
        written.write_json(&path).expect("writes");
        let text = std::fs::read_to_string(&path).expect("reads");
        assert!(text.ends_with("}\n"), "complete, newline-terminated JSON");
        let read = EmberCertificate::read_json(&path).expect("parses back");

        // Everything comes back exactly: the 17-digit floats of `derived`, `stats` and
        // `timings_seconds` and the decimal copies of the initial conditions included …
        assert_eq!(read, written);
        // … bit for bit (`==` would equate -0.0 with 0.0): writing the certificate read back
        // reproduces the file byte for byte, and the shortest round-trip decimal `serde_json`
        // writes is unique per bit pattern.
        let rewritten = serde_json::to_string_pretty(&read).expect("serialises") + "\n";
        assert_eq!(rewritten, text);
        assert_eq!(read.inputs.view, view());
        assert_eq!(read.inputs.view.rotation[0][1].to_bits(), (-0.0f64).to_bits());
        assert_eq!(read.inputs.view.rotation[1][2].to_bits(), 1);
        assert_eq!(read.stats.max_flow_speed.to_bits(), 3.772_653_138_656_007_6f64.to_bits());
        // The decimal copies equal their bit patterns' values.
        assert_eq!(decimals_from_bits(read.clone()), read);

        // The initial conditions are rebuilt from their bit patterns.
        let rebuilt = read.inputs.bodies();
        assert_eq!(rebuilt.len(), bodies.len());
        for (back, body) in rebuilt.iter().zip(&bodies) {
            assert_eq!(body_bits(back), body_bits(body));
        }
        assert_eq!(read.inputs.frames.sha256, schedule_sha256(&schedule()));
        assert_eq!(read.inputs.paper_seed_sha256, paper_seed_sha256(b"abc"));
        assert_eq!(read.inputs.dt.to_bits(), crate::render::constants::DEFAULT_DT.to_bits());
        assert_eq!(read.inputs.gravitational_constant.to_bits(), crate::sim::G.to_bits());
        assert_eq!(read.config, config);
        assert_eq!(serde_json::to_value(&read.config).ok(), serde_json::to_value(&config).ok());
    }

    #[test]
    fn the_layout_is_the_documented_one() {
        let json = serde_json::to_value(certificate(&awkward_bodies(), &config())).expect("json");
        let keys = |value: &Value| -> Vec<String> {
            let mut keys: Vec<String> =
                value.as_object().expect("an object").keys().cloned().collect();
            keys.sort();
            keys
        };
        assert_eq!(
            keys(&json),
            [
                "algorithm",
                "build",
                "config",
                "contract",
                "derived",
                "edition",
                "inputs",
                "outputs",
                "schema_version",
                "stats",
                "timings_seconds"
            ]
        );
        assert_eq!(keys(&json["timings_seconds"]), ["fluid", "ink", "shade", "sink", "total"]);
        assert_eq!(
            keys(&json["derived"]),
            [
                "duration",
                "fade_time",
                "fluid_dx",
                "fluid_grid",
                "hold_time",
                "ink_grid",
                "slow_first_frame",
                "tidal_reference",
                "valve_time"
            ]
        );
        assert_eq!(keys(&json["inputs"]["view"]), ["drift", "frame", "projection", "rotation"]);
        assert_eq!(
            keys(&json["inputs"]["view"]["frame"]),
            ["height", "min_x", "min_y", "scale", "width"]
        );
        assert_eq!(json["inputs"]["view"]["drift"]["mode"], "elliptical");
        assert_eq!(json["inputs"]["view"]["frame"]["scale"], "0x3fe15f229fbe76c8");
        assert_eq!(
            keys(&json["inputs"]["frames"]),
            ["count", "first_step", "frame_rate", "last_step", "sha256", "slow_factor"]
        );
        assert_eq!(
            keys(&json["stats"]),
            [
                "contact_events",
                "fluid_steps",
                "max_dt",
                "max_flow_speed",
                "min_dt",
                "snapshots",
                "still_gamut_mapped_pixels",
                "still_ink_fraction"
            ]
        );
        assert_eq!(
            keys(&json["outputs"]),
            [
                "encoding",
                "frames_emitted",
                "frames_rgb48le_sha256",
                "slow_frames_emitted",
                "slow_frames_rgb48le_sha256",
                "still_rgb48le_sha256"
            ]
        );
        assert_eq!(json["inputs"]["bodies"][1]["bits"]["mass"], "0x3ff0000000000000");
        assert_eq!(json["edition"], EDITION);
        assert_eq!(json["algorithm"], ALGORITHM_VERSION);
        assert_eq!(json["inputs"]["integrator"], INTEGRATOR);
        assert_eq!(json["outputs"]["encoding"], PIXEL_ENCODING);
    }

    #[test]
    fn the_reader_rejects_other_layouts() {
        let json = serde_json::to_value(certificate(&awkward_bodies(), &config())).expect("json");
        let parse = |edit: &dyn Fn(&mut Value)| {
            let mut value = json.clone();
            edit(&mut value);
            EmberCertificate::from_json(&value.to_string())
        };
        assert!(parse(&|_| {}).is_ok());
        assert_eq!(json["schema_version"], CERTIFICATE_SCHEMA_VERSION);
        // Any other version is reported as such, not as a malformed file …
        for version in [0, 1, 2, 3, CERTIFICATE_SCHEMA_VERSION + 1] {
            let error = parse(&|v| v["schema_version"] = version.into()).expect_err("rejected");
            assert!(
                matches!(error, CertificateError::UnsupportedSchema { found } if found == version),
                "{error}"
            );
        }
        // … a published version-2 certificate of the vermilion look included.
        let version_2 = parse(&|v| {
            v["schema_version"] = 2.into();
            v["algorithm"] = "ember-v1".into();
            let stats = v["stats"].as_object_mut().expect("object");
            stats.insert("frames_with_cinnabar".into(), 13.into());
            stats.insert("peak_frame_cinnabar_fraction".into(), 0.001.into());
            stats.insert("still_cinnabar_fraction".into(), 0.0.into());
        })
        .expect_err("a version-2 certificate is rejected");
        assert_eq!(
            version_2.to_string(),
            format!(
                "ember certificate schema version 2 is not supported (this build reads version \
                 {CERTIFICATE_SCHEMA_VERSION})"
            )
        );
        assert!(matches!(
            parse(&|v| v["edition"] = "main".into()),
            Err(CertificateError::WrongEdition { .. })
        ));
        let unknown = parse(&|v| v["inputs"]["extra"] = 1.into()).expect_err("unknown field");
        assert!(unknown.to_string().contains("unknown field `extra`"), "{unknown}");
        let missing = parse(&|v| {
            v["outputs"].as_object_mut().expect("object").remove("still_rgb48le_sha256");
        })
        .expect_err("missing field");
        assert!(missing.to_string().contains("still_rgb48le_sha256"), "{missing}");
        // Nullable fields are required too: without the key a certificate would read as
        // still-only, or without its fading floor.
        for (section, key) in [
            ("outputs", "frames_rgb48le_sha256"),
            ("outputs", "slow_frames_rgb48le_sha256"),
            ("inputs", "view"),
            ("look", "floor_tau"),
            ("derived", "tidal_reference"),
            ("stats", "still_ink_fraction"),
        ] {
            let error = parse(&|v| {
                let object =
                    if section == "look" { &mut v["config"]["look"] } else { &mut v[section] };
                object.as_object_mut().expect("object").remove(key);
            })
            .expect_err("a missing key is rejected");
            assert!(error.to_string().contains(&format!("missing field `{key}`")), "{error}");
        }
        // … while `null` stays a value: a still-only certificate has no frames digest (and no
        // frames).
        let still_only = parse(&|v| {
            v["outputs"]["frames_rgb48le_sha256"] = Value::Null;
            v["outputs"]["frames_emitted"] = 0.into();
            v["outputs"]["slow_frames_rgb48le_sha256"] = Value::Null;
            v["outputs"]["slow_frames_emitted"] = 0.into();
        })
        .expect("an explicit null is read");
        assert_eq!(still_only.outputs.frames_rgb48le_sha256, None);
        let no_fade = parse(&|v| v["config"]["look"]["floor_tau"] = Value::Null).expect("null");
        assert_eq!(no_fade.config.look.floor_tau, None);
        let bits = parse(&|v| v["inputs"]["bodies"][0]["bits"]["mass"] = "0x12".into())
            .expect_err("bad bit pattern");
        assert!(bits.to_string().contains("\"0x12\" is not an f64 bit pattern"), "{bits}");
        let config =
            parse(&|v| v["config"]["look"]["hold_fraction"] = "3".into()).expect_err("mistyped");
        assert!(matches!(config, CertificateError::Json(_)), "{config}");
    }

    #[test]
    fn the_reader_rejects_outputs_that_disagree_about_the_frames() {
        let video = serde_json::to_value(certificate(&awkward_bodies(), &config())).expect("json");
        assert_eq!(video["outputs"]["frames_emitted"], 20);
        assert_eq!(video["inputs"]["frames"]["count"], 20);
        let parse = |edit: &dyn Fn(&mut Value)| {
            let mut value = video.clone();
            edit(&mut value);
            EmberCertificate::from_json(&value.to_string())
        };
        let inconsistent = |edit: &dyn Fn(&mut Value), expected: &str| {
            let error = parse(edit).expect_err("inconsistent outputs are rejected");
            assert!(matches!(error, CertificateError::InconsistentOutputs { .. }), "{error}");
            assert!(error.to_string().contains(expected), "{error}");
        };

        // A video certificate: a digest over all 20 scheduled frames.
        assert!(parse(&|_| {}).is_ok());
        // A still-only certificate, as a still-only render's summary produces it: no digest, no
        // frames, and the full 20-frame schedule (the still is its last knot).
        let mut still_summary = summary();
        (still_summary.frames_sha256, still_summary.frames_emitted) = (None, 0);
        (still_summary.slow_frames_sha256, still_summary.slow_frames_emitted) = (None, 0);
        let still_only = certificate_of(&awkward_bodies(), &config(), &still_summary);
        assert_eq!(still_only.inputs.frames.count, 20);
        let text = serde_json::to_string(&still_only).expect("serialises");
        assert_eq!(EmberCertificate::from_json(&text).expect("still-only is read"), still_only);

        // Frames without a digest: the certificate would verify the still alone.
        inconsistent(
            &|v| v["outputs"]["frames_rgb48le_sha256"] = Value::Null,
            "outputs.frames_rgb48le_sha256 is null, but outputs.frames_emitted is 20",
        );
        // The slow film follows the same rules against its own length: 17 scheduled intervals
        // after frame 2, ten frames each, and the first frame.
        assert_eq!(video["outputs"]["slow_frames_emitted"], 171);
        assert_eq!(video["derived"]["slow_first_frame"], 2);
        let normal_only = parse(&|v| {
            v["outputs"]["slow_frames_rgb48le_sha256"] = Value::Null;
            v["outputs"]["slow_frames_emitted"] = 0.into();
        })
        .expect("a certificate of the normal film alone is read");
        assert_eq!(normal_only.outputs.slow_frames_rgb48le_sha256, None);
        inconsistent(
            &|v| v["outputs"]["slow_frames_rgb48le_sha256"] = Value::Null,
            "outputs.slow_frames_rgb48le_sha256 is null, but outputs.slow_frames_emitted is 171",
        );
        for emitted in [170, 172, 0, 181] {
            inconsistent(
                &|v| v["outputs"]["slow_frames_emitted"] = emitted.into(),
                &format!(
                    "outputs.slow_frames_emitted is {emitted}, but the slow film of 20 scheduled \
                     frames from frame 2 at factor 10 has 171 frames"
                ),
            );
        }
        inconsistent(&|v| v["derived"]["slow_first_frame"] = 20.into(), "has no frames");
        inconsistent(
            &|v| {
                v["outputs"]["frames_rgb48le_sha256"] = Value::Null;
                v["outputs"]["frames_emitted"] = 0.into();
            },
            "outputs.slow_frames_rgb48le_sha256 is set, but outputs.frames_rgb48le_sha256 is null",
        );
        // A digest over fewer, more or no frames than the schedule has …
        for emitted in [19, 21, 0] {
            inconsistent(
                &|v| v["outputs"]["frames_emitted"] = emitted.into(),
                &format!(
                    "outputs.frames_emitted is {emitted}, but a video render emits every one of \
                     the 20 scheduled frames"
                ),
            );
        }
        // … or than a schedule edited under it.
        inconsistent(
            &|v| v["inputs"]["frames"]["count"] = 21.into(),
            "outputs.frames_emitted is 20, but a video render emits every one of the 21 \
             scheduled frames",
        );
    }

    #[test]
    fn unwritable_and_unreadable_paths_are_errors() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let path = dir.path().join("missing").join("ember.json");
        let written = certificate(&awkward_bodies(), &config());
        assert!(written.write_json(&path).is_err());
        assert!(matches!(EmberCertificate::read_json(&path), Err(CertificateError::Io { .. })));
    }
}
