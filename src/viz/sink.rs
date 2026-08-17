//! Artifact sink: owns one mode's output directory under
//! `output/<name>/viz/<flag>/`, provides typed writers, and records every
//! artifact for the viz manifest.

use crate::error::Result;
use crate::render::{ImageBuffer, Rgb, save_image_as_png_16bit};
use serde::Serialize;
use std::fs;

/// One recorded artifact for `viz/manifest.json`.
#[derive(Clone, Debug, Serialize)]
pub struct ArtifactRecord {
    /// Mode flag that produced the artifact.
    pub mode: String,
    /// Path relative to the seed package root.
    pub path: String,
    /// Artifact kind (`image`, `video`, `audio`, `vector`, `data`).
    pub kind: String,
    /// File size in bytes (if the file exists when recorded).
    pub bytes: Option<u64>,
}

/// Sink for one mode's artifacts.
pub struct ArtifactSink {
    seed_dir: String,
    mode_flag: &'static str,
    records: Vec<ArtifactRecord>,
}

impl ArtifactSink {
    /// Create the sink and its `viz/<flag>/` directory.
    pub fn new(seed_dir: &str, mode_flag: &'static str) -> Result<Self> {
        let dir = format!("{seed_dir}/viz/{mode_flag}");
        fs::create_dir_all(&dir)?;
        Ok(Self { seed_dir: seed_dir.to_string(), mode_flag, records: Vec::new() })
    }

    /// Absolute path for an artifact file inside the mode directory.
    #[must_use]
    pub fn path(&self, file_name: &str) -> String {
        format!("{}/viz/{}/{}", self.seed_dir, self.mode_flag, file_name)
    }

    /// Record an artifact that was written at [`Self::path`]`(file_name)`.
    pub fn record(&mut self, file_name: &str, kind: &str) {
        let path = self.path(file_name);
        let bytes = fs::metadata(&path).ok().map(|meta| meta.len());
        self.records.push(ArtifactRecord {
            mode: self.mode_flag.to_string(),
            path: format!("viz/{}/{}", self.mode_flag, file_name),
            kind: kind.to_string(),
            bytes,
        });
    }

    /// Save a 16-bit Display P3 PNG and record it.
    pub fn save_png16(
        &mut self,
        image: &ImageBuffer<Rgb<u16>, Vec<u16>>,
        file_name: &str,
    ) -> Result<()> {
        save_image_as_png_16bit(image, &self.path(file_name))?;
        self.record(file_name, "image");
        Ok(())
    }

    /// Write a UTF-8 text or JSON artifact and record it.
    pub fn write_text(&mut self, file_name: &str, contents: &str, kind: &str) -> Result<()> {
        fs::write(self.path(file_name), contents)?;
        self.record(file_name, kind);
        Ok(())
    }

    /// Consume the sink, returning its artifact records.
    #[must_use]
    pub fn into_records(self) -> Vec<ArtifactRecord> {
        self.records
    }
}

/// Write the aggregated viz manifest for all executed modes.
pub fn write_viz_manifest(seed_dir: &str, records: &[ArtifactRecord]) -> Result<()> {
    #[derive(Serialize)]
    struct VizManifest<'a> {
        schema_version: u32,
        artifacts: &'a [ArtifactRecord],
    }
    let dir = format!("{seed_dir}/viz");
    fs::create_dir_all(&dir)?;
    let manifest = VizManifest { schema_version: 1, artifacts: records };
    let json = serde_json::to_string_pretty(&manifest).map_err(std::io::Error::other)?;
    fs::write(format!("{dir}/manifest.json"), json)?;
    Ok(())
}
