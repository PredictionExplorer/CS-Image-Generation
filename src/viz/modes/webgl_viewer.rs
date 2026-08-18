//! V53 `webgl-viewer` -- Hold Your Orbit.
//!
//! A single-file offline HTML viewer: the decimated trajectory as glowing
//! additive 3D lines with an orbit/zoom camera, time scrub, exposure, and
//! body toggles. Point data is 16-bit quantized and base64-inlined into
//! the frozen WebGL2 template (`assets/viewer/viewer.html`, no external
//! dependencies); `orbit.json` duplicates the data for the website team.

use crate::error::Result;
use crate::oklab::oklab_to_xyz;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use tracing::{info, warn};

/// Target decimated points per body.
const POINTS_PER_BODY: usize = 30_000;
/// The frozen viewer template.
const TEMPLATE: &str = include_str!("../../../assets/viewer/viewer.html");

/// Minimal standard-alphabet base64 encoder (no padding surprises).
pub(crate) fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = u32::from(chunk[0]);
        let b1 = u32::from(chunk.get(1).copied().unwrap_or(0));
        let b2 = u32::from(chunk.get(2).copied().unwrap_or(0));
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[(triple >> 18) as usize & 63] as char);
        out.push(ALPHABET[(triple >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { ALPHABET[(triple >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { ALPHABET[triple as usize & 63] as char } else { '=' });
    }
    out
}

/// `OkLab` -> sRGB bytes (gamut-clamped).
/// Exported for V57 `instrument` (same viewer color path).
pub(crate) fn oklab_to_srgb_bytes(l: f64, a: f64, b: f64) -> [u8; 3] {
    let (x, y, z) = oklab_to_xyz(l, a, b);
    let lr = 3.240_969_941_904_521 * x - 1.537_383_177_570_093 * y - 0.498_610_760_293_003 * z;
    let lg = -0.969_243_636_280_87 * x + 1.875_967_501_507_72 * y + 0.041_555_057_407_175 * z;
    let lb = 0.055_630_079_696_993 * x - 0.203_976_958_888_976 * y + 1.056_971_514_242_878 * z;
    let encode = |channel: f64| -> u8 {
        let clamped = channel.clamp(0.0, 1.0);
        let gamma = if clamped <= 0.003_130_8 {
            12.92 * clamped
        } else {
            1.055 * clamped.powf(1.0 / 2.4) - 0.055
        };
        (gamma * 255.0).round() as u8
    };
    [encode(lr), encode(lg), encode(lb)]
}

/// A packed, decimated trajectory ready for template embedding: the V53
/// record schema shared verbatim with V57 `instrument`.
pub(crate) struct PackedOrbit {
    /// 14-byte little-endian records (`u16 x,y,z,t,speed` + `u8 r,g,b,pad`).
    pub(crate) bytes: Vec<u8>,
    /// Points kept per body.
    pub(crate) counts: [usize; 3],
    /// Quantization bounding box (lo per axis).
    pub(crate) lo: [f64; 3],
    /// Quantization bounding box (hi per axis).
    pub(crate) hi: [f64; 3],
    /// Step stride of the decimation.
    pub(crate) stride: usize,
    /// JSON duplicate of the per-body points (website schema).
    pub(crate) json_bodies: Vec<serde_json::Value>,
}

/// Decimate and pack the trajectory into the shared viewer record schema.
pub(crate) fn pack_orbit(ctx: &VizContext<'_>) -> PackedOrbit {
    let steps = ctx.step_count();
    let per_body = POINTS_PER_BODY.min(steps);
    let stride = (steps / per_body).max(1);
    let speed_window = ctx.kinematics().speed_window();

    // Global bbox over the decimated points for 16-bit quantization.
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for body in 0..3 {
        for step in (0..steps).step_by(stride) {
            let p = ctx.positions[body][step];
            for (axis, value) in [p.x, p.y, p.z].into_iter().enumerate() {
                if value.is_finite() {
                    lo[axis] = lo[axis].min(value);
                    hi[axis] = hi[axis].max(value);
                }
            }
        }
    }
    let extent: Vec<f64> = (0..3).map(|axis| (hi[axis] - lo[axis]).max(1e-12)).collect();

    // Pack 14-byte records; collect the JSON duplicate alongside.
    let mut packed: Vec<u8> = Vec::new();
    let mut counts = [0usize; 3];
    let mut json_bodies: Vec<serde_json::Value> = Vec::new();
    #[allow(clippy::needless_range_loop)]
    for body in 0..3 {
        let mut json_points: Vec<serde_json::Value> = Vec::new();
        for step in (0..steps).step_by(stride) {
            let p = ctx.positions[body][step];
            let q = |axis: usize, value: f64| -> u16 {
                (((value - lo[axis]) / extent[axis]).clamp(0.0, 1.0) * 65535.0) as u16
            };
            let t = ((step as f64 / (steps - 1) as f64) * 65535.0) as u16;
            let speed = ctx.kinematics().speeds[body][step];
            let speed_q = (ctx.kinematics().normalized_speed(speed_window, speed) * 65535.0) as u16;
            let (cl, ca, cb) = ctx.colors[body][step.min(ctx.colors[body].len() - 1)];
            let rgb = oklab_to_srgb_bytes(cl, ca, cb);
            for value in [q(0, p.x), q(1, p.y), q(2, p.z), t, speed_q] {
                packed.extend_from_slice(&value.to_le_bytes());
            }
            packed.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 0]);
            counts[body] += 1;
            if json_points.len() < POINTS_PER_BODY {
                json_points.push(serde_json::json!([
                    (p.x * 1e4).round() / 1e4,
                    (p.y * 1e4).round() / 1e4,
                    (p.z * 1e4).round() / 1e4,
                ]));
            }
        }
        json_bodies.push(serde_json::json!({
            "points": json_points,
            "color": oklab_to_srgb_bytes(
                ctx.mean_color(body).0,
                ctx.mean_color(body).1,
                ctx.mean_color(body).2,
            ),
        }));
    }
    PackedOrbit { bytes: packed, counts, lo, hi, stride, json_bodies }
}

/// The webgl-viewer mode.
pub struct WebglViewer;

impl VizMode for WebglViewer {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("webgl-viewer").expect("webgl-viewer is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 2 {
            warn!("webgl-viewer skipped: trajectory too short");
            return Ok(());
        }
        let PackedOrbit { bytes: packed, counts, lo, hi, stride, json_bodies } = pack_orbit(ctx);
        info!(
            "   webgl-viewer: {} points packed ({:.1} MB inline)",
            counts.iter().sum::<usize>(),
            packed.len() as f64 * 4.0 / 3.0 / 1_048_576.0
        );

        // Inject into the frozen template.
        let accent = {
            let (l, a, b) = ctx.mean_color(0);
            let rgb = oklab_to_srgb_bytes(0.82, a * 1.2, b * 1.2);
            let _ = l;
            format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2])
        };
        let meta = serde_json::json!({ "bodies": counts });
        let html = TEMPLATE
            .replace("__TITLE__", &format!("0x{}", ctx.seed_hex.to_uppercase()))
            .replace("__ACCENT__", &accent)
            .replace("__META__", &meta.to_string())
            .replace("__DATA_B64__", &base64_encode(&packed));
        sink.write_text("viewer.html", &html, "data")?;

        let orbit = serde_json::json!({
            "seed": format!("0x{}", ctx.seed_hex),
            "steps": steps,
            "stride": stride,
            "bbox": { "lo": lo, "hi": hi },
            "bodies": json_bodies,
        });
        let json = serde_json::to_string(&orbit).map_err(std::io::Error::other)?;
        sink.write_text("orbit.json", &json, "data")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_known_vectors() {
        assert_eq!(base64_encode(b"Man"), "TWFu");
        assert_eq!(base64_encode(b"Ma"), "TWE=");
        assert_eq!(base64_encode(b"M"), "TQ==");
        assert_eq!(base64_encode(b"light work."), "bGlnaHQgd29yay4=");
    }

    #[test]
    fn template_carries_every_placeholder() {
        for placeholder in ["__TITLE__", "__ACCENT__", "__META__", "__DATA_B64__"] {
            assert!(TEMPLATE.contains(placeholder), "viewer template is missing {placeholder}");
        }
        assert!(TEMPLATE.contains("webgl2"), "template must request WebGL2");
    }
}
