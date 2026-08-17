//! V19 `slit-scan` -- The Whole Film in One Image.
//!
//! Collapses the entire main trajectory video into two chronograms using the
//! frame tap: a linear slit (the trajectory-centroid column of every frame
//! laid side by side) and a radial slit (a fixed ring unrolled into rows).
//! Pure post-processing on frames the encoder already received.

use crate::error::Result;
use crate::render::{ImageBuffer, Rgb};
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use tracing::warn;

/// The slit-scan chronogram mode.
pub struct SlitScan;

impl VizMode for SlitScan {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("slit-scan").expect("slit-scan is in the catalog")
    }

    fn needs_frame_tap(&self) -> bool {
        true
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let Some(tap) = ctx.frame_tap.as_ref() else {
            warn!("slit-scan skipped: no frame tap (video render required, not --image-only)");
            return Ok(());
        };
        if tap.frames == 0 {
            warn!("slit-scan skipped: frame tap observed no frames");
            return Ok(());
        }

        // Linear chronogram: x = frame index, y = source row.
        let frames = tap.frames as u32;
        let height = tap.height;
        let mut linear = vec![0u16; tap.frames * height as usize * 3];
        let rows = height as usize;
        for frame in 0..tap.frames {
            for row in 0..rows {
                let source = (frame * rows + row) * 3;
                let dest = (row * tap.frames + frame) * 3;
                linear[dest..dest + 3].copy_from_slice(&tap.linear[source..source + 3]);
            }
        }
        let linear_image: ImageBuffer<Rgb<u16>, Vec<u16>> =
            ImageBuffer::from_raw(frames, height, linear).expect("linear chronogram buffer size");
        sink.save_png16(&linear_image, "slitscan_linear.png")?;

        // Radial chronogram: x = ring angle, y = frame index.
        let ring_width = tap.ring_samples as u32;
        let radial_image: ImageBuffer<Rgb<u16>, Vec<u16>> =
            ImageBuffer::from_raw(ring_width, frames, tap.ring.clone())
                .expect("radial chronogram buffer size");
        sink.save_png16(&radial_image, "slitscan_radial.png")?;

        let meta = serde_json::json!({
            "frames": tap.frames,
            "column_x": tap.column_x,
            "ring_samples": tap.ring_samples,
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("slitscan.json", &json, "data")?;
        Ok(())
    }
}
