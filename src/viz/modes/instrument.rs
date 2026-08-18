//! V57 `instrument` -- Play Your Orbit.
//!
//! The V53 viewer extended into an instrument: a single offline HTML file
//! where scrubbing time *bows* the orbit. Three `WebAudio` voices synthesize
//! V16's just-intonation score (the same quantized segments, the same
//! partial stack -- one truth), scrub velocity maps to bow pressure
//! (lowpass cutoff + gain), consonance to the reverb send. A scope mode
//! renders the beam with V54's dwell-intensity law in the fragment shader;
//! keys 1/2/3 solo bodies; a record button exports a WAV via an inline PCM
//! re-encode. `instrument_data.json` duplicates trajectory + intervals in
//! the V53 schema for the website team.

use crate::error::Result;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::context::VizContext;
use crate::viz::modes::chord_progression::{JI_LATTICE, SCORE_HOPS, quantized_score};
use crate::viz::modes::webgl_viewer::{
    PackedOrbit, base64_encode, oklab_to_srgb_bytes, pack_orbit,
};
use crate::viz::sink::ArtifactSink;
use tracing::{info, warn};

/// Embedded-data budget from the spec (bytes of the final HTML).
const DATA_BUDGET_BYTES: usize = 4 * 1024 * 1024;
/// The frozen instrument template.
const TEMPLATE: &str = include_str!("../../../assets/viewer/instrument.html");

/// Delta-encode a quantized u8 series: first byte absolute, then wrapping
/// two's-complement deltas (the template's decoder mirrors this exactly).
pub(crate) fn delta_encode_u8(series: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(series.len());
    let mut previous = 0u8;
    for (index, &value) in series.iter().enumerate() {
        if index == 0 {
            out.push(value);
        } else {
            out.push(value.wrapping_sub(previous));
        }
        previous = value;
    }
    out
}

/// Reference decoder for the round-trip unit test (matches the JS).
#[cfg(test)]
pub(crate) fn delta_decode_u8(packed: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(packed.len());
    let mut acc = 0u8;
    for (index, &byte) in packed.iter().enumerate() {
        acc = if index == 0 { byte } else { acc.wrapping_add(byte) };
        out.push(acc);
    }
    out
}

/// The instrument mode.
pub struct Instrument;

impl VizMode for Instrument {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("instrument").expect("instrument is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 100 {
            warn!("instrument skipped: trajectory too short");
            return Ok(());
        }

        // --- Trajectory: the V53 record schema, shared by construction.
        let PackedOrbit { bytes: packed, counts, lo, hi, stride, json_bodies } = pack_orbit(ctx);

        // --- Score: V16's exact quantized voices + consonance curve.
        let score = quantized_score(ctx);
        let voices_json: Vec<serde_json::Value> = score
            .voices
            .iter()
            .map(|notes| {
                let segments: Vec<serde_json::Value> = notes
                    .iter()
                    .map(|note| {
                        let (num, den) = JI_LATTICE[note.ratio];
                        let start_hop = (note.start * SCORE_HOPS as f64).round() as u32;
                        let end_hop = (note.end * SCORE_HOPS as f64).round() as u32;
                        serde_json::json!([start_hop, end_hop, num, den, note.octave])
                    })
                    .collect();
                serde_json::Value::Array(segments)
            })
            .collect();
        let consonance_u8: Vec<u8> =
            score.consonance.iter().map(|&value| (value.clamp(0.0, 1.0) * 255.0) as u8).collect();
        let consonance_b64 = base64_encode(&delta_encode_u8(&consonance_u8));
        let score_json = serde_json::json!({
            "root_hz": score.root_hz,
            "hops": SCORE_HOPS,
            "voices": voices_json,
            "consonance_b64": consonance_b64,
        });

        // --- Template fill.
        let accent = {
            let (_, a, b) = ctx.mean_color(0);
            let rgb = oklab_to_srgb_bytes(0.82, a * 1.2, b * 1.2);
            format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2])
        };
        let meta = serde_json::json!({
            "bodies": counts,
            "seed": format!("0x{}", ctx.seed_hex),
        });
        let html = TEMPLATE
            .replace("__TITLE__", &format!("0x{}", ctx.seed_hex.to_uppercase()))
            .replace("__ACCENT__", &accent)
            .replace("__META__", &meta.to_string())
            .replace("__SCORE__", &score_json.to_string())
            .replace("__DATA_B64__", &base64_encode(&packed));
        if html.len() > DATA_BUDGET_BYTES {
            warn!(
                "instrument: {} exceeds the 4 MB embed budget ({:.2} MB)",
                "instrument.html",
                html.len() as f64 / 1_048_576.0
            );
        }
        info!(
            "   instrument: {} points, {} note segments, {:.2} MB html",
            counts.iter().sum::<usize>(),
            score.voices.iter().map(Vec::len).sum::<usize>(),
            html.len() as f64 / 1_048_576.0
        );
        sink.write_text("instrument.html", &html, "data")?;

        // --- Data sidecar: trajectory (V53 schema) + intervals.
        let data = serde_json::json!({
            "seed": format!("0x{}", ctx.seed_hex),
            "steps": steps,
            "stride": stride,
            "bbox": { "lo": lo, "hi": hi },
            "bodies": json_bodies,
            "score": score_json,
        });
        let json = serde_json::to_string(&data).map_err(std::io::Error::other)?;
        sink.write_text("instrument_data.json", &json, "data")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_carries_every_placeholder() {
        for placeholder in ["__TITLE__", "__ACCENT__", "__META__", "__SCORE__", "__DATA_B64__"] {
            assert!(TEMPLATE.contains(placeholder), "instrument template is missing {placeholder}");
        }
        assert!(TEMPLATE.contains("webgl2"), "template must request WebGL2");
        assert!(TEMPLATE.contains("AudioContext"), "template must build a WebAudio graph");
        assert!(TEMPLATE.contains("MediaRecorder"), "template must record performances");
        assert!(TEMPLATE.contains("latencyHint"), "template must request interactive latency");
        assert!(
            TEMPLATE.contains("createPeriodicWave") && TEMPLATE.contains("0.25, 0.12"),
            "voices must use V16's exact partial stack"
        );
    }

    #[test]
    fn delta_encoding_round_trips_any_series() {
        let series: Vec<u8> = (0..1200).map(|index| ((index * 37) % 256) as u8).collect();
        let packed = delta_encode_u8(&series);
        assert_eq!(packed.len(), series.len());
        assert_eq!(delta_decode_u8(&packed), series);
    }

    #[test]
    fn delta_encoding_of_smooth_series_is_small_valued() {
        // A smooth consonance ramp produces near-zero deltas (compressible).
        let series: Vec<u8> = (0..255).map(|index| index as u8).collect();
        let packed = delta_encode_u8(&series);
        assert!(packed.iter().skip(1).all(|&delta| delta == 1));
    }
}
