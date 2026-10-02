//! Application orchestration and workflow management
//!
//! This module breaks down the main application flow into clean, focused functions,
//! each with a single responsibility. This improves testability, readability, and
//! maintainability.

use crate::drift::{AppliedDrift, parse_drift_mode};
use crate::drift_config::{ResolvedDriftConfig, resolve_drift_config};
use crate::ember::certificate::{CertificateContext, EmberCertificate};
use crate::ember::{
    self, EmberConfig, EmberError, EmberFrame, EmberMode, EmberRequest, EmberResult, EmberSummary,
    View, ViewDrift, ViewFrame, ViewProjection,
};
use crate::error::{AppError, ConfigError, Result};
use crate::generation_log::{
    DriftConfig, GenerationLogger, GenerationRecord, LoggedRenderConfig, OrbitInfo,
    SimulationConfig,
};
use crate::render::{
    self, ChannelLevels, RenderConfig, SpectralRenderSettings, SpectralScene, ToneMappingControls,
    VideoEncodingOptions, VideoOutputSpec, constants, create_video_groups_from_frames,
    create_videos_from_frames_singlepass, generate_body_color_sequences,
    pass_1_build_histogram_spectral, pass_2_write_frames_spectral, save_image_as_png_16bit,
    save_image_as_srgb_png_16bit,
};
use crate::sim::{self, Body, Sha3RandomByteStream, TrajectoryResult};
use chrono::Local;
use image::{ImageBuffer, Rgb};
use nalgebra::{Matrix3, Vector3};
use serde::Serialize;
use std::fs::{self, File};
use std::io::Write as _;
use std::process::Command;
use std::time::Instant;
use tracing::{info, warn};

/// RNG fork domain for the seeded viewing orientation.
const VIEW_RNG_DOMAIN: &[u8] = b"cosmic-view/v1";
/// Maximum width of the lightweight WebP preview image.
pub const WEB_PREVIEW_MAX_WIDTH: u32 = 640;

/// Package-relative path of the ember edition's 16-bit sRGB still.
pub const EMBER_STILL_PATH: &str = "images/source/ember.png";
/// Package-relative path of the ember still's full-resolution WebP.
pub const EMBER_FULL_WEBP_PATH: &str = "images/web/ember_full.webp";
/// Package-relative path of the ember still's preview WebP.
pub const EMBER_PREVIEW_WEBP_PATH: &str = "images/web/ember_preview.webp";
/// Package-relative path of the ember edition's browser-compatible H.264 video.
pub const EMBER_WEB_VIDEO_PATH: &str = "videos/web/ember.mp4";
/// Package-relative path of the ember edition's slow film (browser-compatible H.264): the same
/// film [`EMBER_SLOW_FACTOR`] times slower.
pub const EMBER_SLOW_WEB_VIDEO_PATH: &str = "videos/web/ember_slow.mp4";
/// Package-relative path of the ember edition's archival HEVC video.
pub const EMBER_HQ_VIDEO_PATH: &str = "videos/hq/ember.mp4";
/// Package-relative path of the ember edition's determinism certificate.
pub const EMBER_CERTIFICATE_PATH: &str = "metadata/ember.json";
/// Package-relative paths of every file the ember edition writes, in [`EmberOutputPaths`] field
/// order: still, full WebP, preview WebP, web video, slow web video, HQ video, certificate.
/// `run.py` requires each of them in a complete package (`EMBER_PACKAGE_FILES`).
pub const EMBER_OUTPUT_PATHS: [&str; 7] = [
    EMBER_STILL_PATH,
    EMBER_FULL_WEBP_PATH,
    EMBER_PREVIEW_WEBP_PATH,
    EMBER_WEB_VIDEO_PATH,
    EMBER_SLOW_WEB_VIDEO_PATH,
    EMBER_HQ_VIDEO_PATH,
    EMBER_CERTIFICATE_PATH,
];

/// How many times slower the ember edition's slow film is: it shows this many frames for each
/// frame of the normal film, every one of them simulated (`ember::EmberRequest::slow_factor`).
pub const EMBER_SLOW_FACTOR: u32 = 10;

/// Appended to the package seed bytes to seed the ember edition's kozo sheet: a separate,
/// versioned domain, so the paper texture never correlates with any other seeded choice and
/// changes only if this string does.
const EMBER_PAPER_SEED_DOMAIN: &[u8] = b"\0cosmic-ember/kozo-sheet/v1";

/// Core `CosmicSignature` enhancement flags.
#[derive(Clone, Debug)]
pub struct Enhancements {
    /// Enable chroma boosting for richer color saturation.
    pub chroma_boost: bool,
    /// Enable perceptual saturation boost.
    pub sat_boost: bool,
    /// Enable ACES-inspired tone-mapping tweak.
    pub aces_tweak: bool,
    /// Enable per-body alpha variation for visual depth.
    pub alpha_variation: bool,
    /// Enable aspect-ratio correction for non-square outputs.
    pub aspect_correction: bool,
    /// Enable spectral dispersion boost.
    pub dispersion_boost: bool,
}

impl Default for Enhancements {
    fn default() -> Self {
        Self {
            chroma_boost: true,
            sat_boost: true,
            aces_tweak: true,
            alpha_variation: true,
            aspect_correction: true,
            dispersion_boost: false,
        }
    }
}

/// Configuration recorded for generation logging.
pub struct GenerationLogConfig {
    /// Number of simulation time-steps.
    pub num_steps_sim: usize,
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// Black-point clipping percentile.
    pub clip_black: f64,
    /// White-point clipping percentile.
    pub clip_white: f64,
    /// Alpha denominator controlling base trail opacity.
    pub alpha_denom: usize,
    /// Alpha compression factor.
    pub alpha_compress: f64,
    /// Escape threshold for orbit rejection.
    pub escape_threshold: f64,
    /// Drift mode identifier (e.g. `"elliptical"`, `"none"`).
    pub drift_mode: String,
    /// Visual profile identifier.
    pub visual_profile: String,
    /// Whether any finish effect was enabled.
    pub post_effects_enabled: bool,
    /// Bloom post-processing mode, kept for compatibility with older logs.
    pub bloom_mode: String,
    /// HDR tone-mapping mode.
    pub hdr_mode: String,
    /// HDR intensity scale factor.
    pub hdr_scale: f64,
    /// Active radial spectral dispersion strength.
    pub dispersion_strength: f64,
    /// Active spectral dispersion mode.
    pub dispersion_mode: String,
    /// Continuous palette genome fingerprint.
    pub palette_fingerprint: String,
    /// Palette beauty-gate descriptor (attempts / repair flag).
    pub palette_gate: String,
    /// Minimum body mass for simulation.
    pub min_mass: f64,
    /// Maximum body mass for simulation.
    pub max_mass: f64,
    /// Initial location spread parameter.
    pub location: f64,
    /// Initial velocity spread parameter.
    pub velocity: f64,
    /// Weight for chaos metric in Borda scoring.
    pub chaos_weight: f64,
    /// Weight for equilibrium metric in Borda scoring.
    pub equil_weight: f64,
    /// Whether Borda weights were randomized.
    pub weights_randomized: bool,
}

/// Paths for the generated still image and its website derivatives.
#[derive(Clone, Copy, Debug)]
pub struct ImageOutputPaths<'a> {
    /// Maximum-quality 16-bit PNG source image.
    pub master_png: &'a str,
    /// Full-resolution WebP with the same pixel dimensions as `master_png`.
    pub full_webp: &'a str,
    /// Smaller same-aspect-ratio WebP for cards, previews, and video posters.
    pub preview_webp: &'a str,
}

/// Paths for a pair of video variants generated from one animation.
#[derive(Clone, Copy, Debug)]
pub struct VideoOutputPaths<'a> {
    /// Website-compatible H.264 MP4.
    pub web: &'a str,
    /// High-quality HEVC MP4.
    pub high_quality: &'a str,
}

/// Initialize per-seed output directory structure:
///   output/{seed}/images/source/
///   output/{seed}/images/web/
///   output/{seed}/videos/web/
///   output/{seed}/videos/hq/
///   output/{seed}/spectral/
///   output/{seed}/metadata/
///
/// Rejects output names containing path separators or `..` to prevent directory traversal.
pub fn setup_seed_directory(seed: &str) -> Result<String> {
    if seed.contains("..") || seed.contains('/') || seed.contains('\\') {
        return Err(ConfigError::InvalidResolution {
            reason: format!("Output name '{seed}' must not contain path separators or '..'"),
        }
        .into());
    }

    let seed_dir = format!("output/{seed}");
    fs::create_dir_all(&seed_dir).map_err(|e| ConfigError::FileSystem {
        operation: "create directory".to_string(),
        path: seed_dir.clone(),
        error: e,
    })?;

    for subdir in ["images/source", "images/web", "videos/web", "videos/hq", "spectral", "metadata"]
    {
        let path = format!("{seed_dir}/{subdir}");
        fs::create_dir_all(&path).map_err(|e| ConfigError::FileSystem {
            operation: "create directory".to_string(),
            path,
            error: e,
        })?;
    }

    Ok(seed_dir)
}

/// Parse and validate hex seed
pub fn parse_seed(seed: &str) -> Result<Vec<u8>> {
    let hex_seed = seed.strip_prefix("0x").unwrap_or(seed);

    hex::decode(hex_seed)
        .map_err(|e| ConfigError::InvalidSeed { seed: seed.to_string(), error: e }.into())
}

#[derive(Serialize)]
struct AssetManifest {
    schema_version: u32,
    generated_at: String,
    assets: Vec<AssetEntry>,
}

#[derive(Serialize)]
struct AssetEntry {
    path: String,
    kind: String,
    role: String,
    format: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frame_rate: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    codec: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pixel_format: Option<String>,
    /// Colour encoding, where it is not the main renderer's (the ember entries are `srgb`).
    #[serde(skip_serializing_if = "Option::is_none")]
    color_space: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    file_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sha256: Option<String>,
}

/// Codec facts of a video entry.
struct VideoFacts<'a> {
    /// Playback duration.
    duration_seconds: f64,
    /// Frames per second.
    frame_rate: u32,
    /// Codec name as the manifest spells it (`h264`, `hevc`).
    codec: &'a str,
    /// `FFmpeg` pixel format of the encoded stream.
    pixel_format: &'a str,
}

impl AssetEntry {
    /// One file of the package with its size and SHA-256 (both absent when the file is missing,
    /// e.g. a video of an image-only run).
    fn file(seed_dir: &str, path: &str, kind: &str, role: &str, format: &str) -> Self {
        Self {
            path: path.to_string(),
            kind: kind.to_string(),
            role: role.to_string(),
            format: format.to_string(),
            width: None,
            height: None,
            duration_seconds: None,
            frame_rate: None,
            codec: None,
            pixel_format: None,
            color_space: None,
            file_count: None,
            bytes: file_size(seed_dir, path),
            sha256: file_sha256(seed_dir, path),
        }
    }

    /// An image file of `size` (`width`, `height`) pixels.
    fn image(seed_dir: &str, path: &str, role: &str, format: &str, size: (u32, u32)) -> Self {
        Self {
            width: Some(size.0),
            height: Some(size.1),
            ..Self::file(seed_dir, path, "image", role, format)
        }
    }

    /// An MP4 video of `size` pixels.
    fn video(
        seed_dir: &str,
        path: &str,
        role: &str,
        size: (u32, u32),
        facts: &VideoFacts<'_>,
    ) -> Self {
        Self {
            width: Some(size.0),
            height: Some(size.1),
            duration_seconds: Some(facts.duration_seconds),
            frame_rate: Some(facts.frame_rate),
            codec: Some(facts.codec.to_string()),
            pixel_format: Some(facts.pixel_format.to_string()),
            ..Self::file(seed_dir, path, "video", role, "mp4")
        }
    }

    fn with_pixel_format(self, pixel_format: &str) -> Self {
        Self { pixel_format: Some(pixel_format.to_string()), ..self }
    }

    fn with_color_space(self, color_space: &str) -> Self {
        Self { color_space: Some(color_space.to_string()), ..self }
    }
}

/// What the ember stage produced, as recorded in `metadata/assets.json`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmberManifest {
    /// Frames in each ember video of normal speed (`0` when no video was encoded).
    pub frames_emitted: usize,
    /// Frames in the slow film (`0` when no video was encoded).
    pub slow_frames_emitted: usize,
    /// Frames per second of the ember videos.
    pub frame_rate: u32,
    /// Whether the ember videos were encoded (not under `--image-only`).
    pub has_video: bool,
    /// Whether the HQ slot holds the software fast encode (`--fast-encode`) instead of HEVC.
    pub fast_encode: bool,
}

impl EmberManifest {
    /// Manifest facts of a finished ember render encoded at the product frame rate.
    pub fn from_summary(summary: &EmberSummary, fast_encode: bool) -> Self {
        Self {
            frames_emitted: summary.frames_emitted,
            slow_frames_emitted: summary.slow_frames_emitted,
            frame_rate: constants::DEFAULT_VIDEO_FPS,
            has_video: summary.frames_emitted > 0,
            fast_encode,
        }
    }
}

fn file_size(seed_dir: &str, relative_path: &str) -> Option<u64> {
    fs::metadata(format!("{seed_dir}/{relative_path}")).ok().map(|meta| meta.len())
}

/// Streaming SHA-256 of an output file, hex-encoded (lowercase). Returns
/// `None` when the file does not exist (e.g. video entries in image-only
/// runs), mirroring `file_size`.
fn file_sha256(seed_dir: &str, relative_path: &str) -> Option<String> {
    use sha2::{Digest as _, Sha256};
    use std::io::Read as _;

    let mut file = File::open(format!("{seed_dir}/{relative_path}")).ok()?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer).ok()?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Some(hex::encode(hasher.finalize()))
}

fn preview_dimensions(width: u32, height: u32) -> (u32, u32) {
    if width <= WEB_PREVIEW_MAX_WIDTH {
        return (width, height);
    }
    let preview_height =
        (f64::from(height) * f64::from(WEB_PREVIEW_MAX_WIDTH) / f64::from(width)).round() as u32;
    (WEB_PREVIEW_MAX_WIDTH, preview_height.max(1))
}

fn run_webp_encode(input_path: &str, output_path: &str, scale_filter: Option<&str>) -> Result<()> {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-y", "-i", input_path, "-map_metadata", "-1"]);
    if let Some(filter) = scale_filter {
        cmd.args(["-vf", filter]);
    }
    cmd.args([
        "-frames:v",
        "1",
        "-c:v",
        "libwebp",
        "-preset",
        "picture",
        "-quality",
        "82",
        "-compression_level",
        "6",
        output_path,
    ]);

    let status = cmd.status().map_err(render::error::RenderError::VideoEncoding)?;
    if !status.success() {
        return Err(render::error::RenderError::ImageEncoding {
            reason: format!("ffmpeg WebP encode failed for {output_path}"),
        }
        .into());
    }

    info!("   Saved WebP => {output_path}");
    Ok(())
}

/// Generate full-size and preview WebP derivatives from the master PNG.
pub fn generate_webp_images(paths: ImageOutputPaths<'_>) -> Result<()> {
    run_webp_encode(paths.master_png, paths.full_webp, None)?;
    run_webp_encode(paths.master_png, paths.preview_webp, Some("scale=w=min(640\\,iw):h=-1"))
}

/// Write a website-oriented asset manifest for the generated package.
///
/// `steps` is the number of recorded orbit steps (`--steps`): `main.mp4` has one frame per
/// entry of `render::main_video_checkpoints(steps)`, so its duration is that count over the frame
/// rate.
///
/// `ember` describes the ember edition's outputs (`None` when it was skipped with `--no-ember`
/// or failed); its entries are appended after the main renderer's with their own roles
/// (`ember_source_master`, `ember_web_full`, `ember_web_preview`, `ember_web`, `ember_slow_web`,
/// `ember_hq`), so the manifest stays at `schema_version` 2 for existing readers.
pub fn write_asset_manifest(
    seed_dir: &str,
    width: u32,
    height: u32,
    steps: usize,
    image_only: bool,
    ember: Option<&EmberManifest>,
) -> Result<()> {
    let size = (width, height);
    let mut assets = main_image_entries(seed_dir, size);
    if !image_only {
        assets.extend(main_video_entries(seed_dir, size, steps));
    }
    if let Some(ember) = ember {
        assets.extend(ember_entries(seed_dir, size, ember));
    }

    let manifest =
        AssetManifest { schema_version: 2, generated_at: Local::now().to_rfc3339(), assets };
    let path = format!("{seed_dir}/metadata/assets.json");
    crate::utils::write_json_pretty(&path, &manifest)?;
    info!("   Saved asset metadata => {path}");
    Ok(())
}

/// The main renderer's master still and its WebP derivatives.
fn main_image_entries(seed_dir: &str, size: (u32, u32)) -> Vec<AssetEntry> {
    let preview = preview_dimensions(size.0, size.1);
    vec![
        AssetEntry::image(seed_dir, "images/source/master.png", "source_master", "png", size)
            .with_pixel_format("rgb48"),
        AssetEntry::image(seed_dir, "images/web/full.webp", "web_full", "webp", size),
        AssetEntry::image(seed_dir, "images/web/preview.webp", "web_preview", "webp", preview),
    ]
}

/// The main renderer's videos (of an orbit of `steps` recorded steps) and the spectral gallery.
fn main_video_entries(seed_dir: &str, size: (u32, u32), steps: usize) -> Vec<AssetEntry> {
    let fps = constants::DEFAULT_VIDEO_FPS;
    let seconds = |frames: usize| frames as f64 / f64::from(fps);
    // The frames `render_video` actually encodes (1,802 at the default 1,000,000 steps), not the
    // nominal target: the ember videos, frame-locked to `main.mp4`, get the same duration.
    let main_duration = seconds(render::main_video_checkpoints(steps).len());
    let sweep_duration = seconds(constants::CYCLE_TOTAL_FRAMES as usize);
    let facts = |duration_seconds, codec, pixel_format| VideoFacts {
        duration_seconds,
        frame_rate: fps,
        codec,
        pixel_format,
    };
    let (web, hq) = (("h264", "yuv420p"), ("hevc", "yuv422p10le"));
    vec![
        AssetEntry::video(
            seed_dir,
            "videos/web/main.mp4",
            "main_web",
            size,
            &facts(main_duration, web.0, web.1),
        ),
        AssetEntry::video(
            seed_dir,
            "videos/hq/main.mp4",
            "main_hq",
            size,
            &facts(main_duration, hq.0, hq.1),
        ),
        AssetEntry::video(
            seed_dir,
            "videos/web/spectral_sweep.mp4",
            "spectral_sweep_web",
            size,
            &facts(sweep_duration, web.0, web.1),
        ),
        AssetEntry::video(
            seed_dir,
            "videos/hq/spectral_sweep.mp4",
            "spectral_sweep_hq",
            size,
            &facts(sweep_duration, hq.0, hq.1),
        ),
        AssetEntry {
            width: Some(size.0),
            height: Some(size.1),
            pixel_format: Some("rgb48".to_string()),
            file_count: Some(crate::spectrum::NUM_BINS),
            // A directory: no single size or digest.
            bytes: None,
            sha256: None,
            ..AssetEntry::file(seed_dir, "spectral/", "image_set", "spectral_bins", "png")
        },
    ]
}

/// The ember edition's still, its WebP derivatives and (when encoded) its videos, all sRGB.
fn ember_entries(seed_dir: &str, size: (u32, u32), ember: &EmberManifest) -> Vec<AssetEntry> {
    let preview = preview_dimensions(size.0, size.1);
    let mut entries = vec![
        AssetEntry::image(seed_dir, EMBER_STILL_PATH, "ember_source_master", "png", size)
            .with_pixel_format("rgb48"),
        AssetEntry::image(seed_dir, EMBER_FULL_WEBP_PATH, "ember_web_full", "webp", size),
        AssetEntry::image(seed_dir, EMBER_PREVIEW_WEBP_PATH, "ember_web_preview", "webp", preview),
    ];
    if ember.has_video {
        let [web, hq] = ember_video_options(ember.fast_encode);
        for (path, role, options, frames) in [
            (EMBER_WEB_VIDEO_PATH, "ember_web", &web, ember.frames_emitted),
            (EMBER_SLOW_WEB_VIDEO_PATH, "ember_slow_web", &web, ember.slow_frames_emitted),
            (EMBER_HQ_VIDEO_PATH, "ember_hq", &hq, ember.frames_emitted),
        ] {
            let facts = VideoFacts {
                duration_seconds: frames as f64 / f64::from(ember.frame_rate),
                frame_rate: ember.frame_rate,
                codec: manifest_codec(&options.codec),
                pixel_format: &options.pixel_format,
            };
            entries.push(AssetEntry::video(seed_dir, path, role, size, &facts));
        }
    }
    entries.into_iter().map(|entry| entry.with_color_space("srgb")).collect()
}

/// Codec name the manifest uses for an `FFmpeg` encoder.
fn manifest_codec(encoder: &str) -> &'static str {
    if encoder.contains("265") || encoder.contains("hevc") { "hevc" } else { "h264" }
}

fn write_generation_record(path: &str, record: &GenerationRecord) -> Result<()> {
    crate::utils::write_json_pretty(path, record)?;
    info!("   Saved generation metadata => {path}");
    Ok(())
}

/// Winning orbit of the combined Borda + aesthetic selection.
pub struct AestheticSelection {
    /// Initial body states of the winning candidate.
    pub bodies: Vec<Body>,
    /// Borda/physics metrics of the winning candidate.
    pub result: TrajectoryResult,
    /// Full trajectory of the winning candidate, already transformed into the
    /// seed's projection space (no re-simulation needed).
    pub positions: Vec<Vec<Vector3<f64>>>,
    /// Proxy-render aesthetic score of the winning candidate.
    pub aesthetic: render::aesthetic_score::AestheticScore,
    /// Aesthetic score plus the small prior bonus used for tie-breaking.
    pub selection_score: f64,
    /// Layer stack chosen by adaptive orbit × stack scoring.
    pub stack: render::LayerStack,
    /// Layer stack originally rolled by the seed.
    pub preferred_stack: render::LayerStack,
    /// Number of bounded retry searches used before accepting this candidate.
    pub retry_count: usize,
}

/// Number of top Borda candidates proxy-rendered for aesthetic selection.
pub const AESTHETIC_SHORTLIST_LEN: usize = 24;
/// Minimum proxy aesthetic score accepted before bounded retry searches.
pub const AESTHETIC_QUALITY_FLOOR: f64 = 0.82;
/// Maximum deterministic retry searches when the best proxy score is below the floor.
pub const AESTHETIC_MAX_RETRIES: usize = 3;
/// RNG fork domain prefix for bounded aesthetic retry searches.
const AESTHETIC_RETRY_RNG_DOMAIN_PREFIX: &str = "cosmic-retry/v1/";

/// Transform a trajectory into the seed's projection space.
///
/// Phase-space projections plot mixed position/velocity coordinates, producing
/// Lissajous-like curve families impossible in plain position space. Velocity
/// axes are rescaled to the position extent so mixed-axis projections stay
/// well-proportioned; the render context refits the bounding box afterwards.
#[must_use]
pub fn apply_projection(
    positions: &[Vec<Vector3<f64>>],
    projection: render::ProjectionMode,
) -> Vec<Vec<Vector3<f64>>> {
    use render::ProjectionMode;

    if projection == ProjectionMode::Position {
        return positions.to_vec();
    }
    let steps = positions.first().map_or(0, Vec::len);
    if positions.len() < 3 || steps < 2 {
        return positions.to_vec();
    }

    // Per-body velocities by forward difference (last step repeats).
    let dt = constants::DEFAULT_DT;
    let velocities: Vec<Vec<Vector3<f64>>> = positions
        .iter()
        .map(|body| {
            (0..steps)
                .map(|step| {
                    let next = (step + 1).min(steps - 1);
                    let prev = next.saturating_sub(1);
                    (body[next] - body[prev]) / dt
                })
                .collect()
        })
        .collect();

    let extent = |data: &[Vec<Vector3<f64>>], axis: usize| -> f64 {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for body in data {
            for p in body {
                if p[axis].is_finite() {
                    min = min.min(p[axis]);
                    max = max.max(p[axis]);
                }
            }
        }
        (max - min).max(1e-12)
    };
    let pos_extent = extent(positions, 0).max(extent(positions, 1));
    let vel_extent = extent(&velocities, 0).max(extent(&velocities, 1));
    let velocity_scale = if vel_extent > 1e-12 { pos_extent / vel_extent } else { 1.0 };

    (0..positions.len())
        .map(|body| {
            (0..steps)
                .map(|step| {
                    let p = positions[body][step];
                    let v = velocities[body][step] * velocity_scale;
                    match projection {
                        ProjectionMode::Position => p,
                        ProjectionMode::PhasePortrait => Vector3::new(p.x, v.x, p.y),
                        ProjectionMode::CrossBraid => {
                            let partner = positions[(body + 1) % 3][step];
                            Vector3::new(p.x, partner.y, p.z)
                        }
                        ProjectionMode::Hodograph => Vector3::new(v.x, v.y, p.z),
                    }
                })
                .collect()
        })
        .collect()
}

/// Curated fallback stacks evaluated alongside the seed's own stack.
///
/// Keeps adaptive selection bounded while ensuring that if the seed's stack
/// genuinely does not suit the orbit, a reliable alternative exists.
fn adaptive_structure_stacks(preferred: render::LayerStack) -> Vec<render::LayerStack> {
    let mut stacks = vec![preferred];
    for fallback in [
        render::LayerStack::solo(preferred.primary),
        render::LayerStack {
            primary: render::StructureMode::OrbitRibbons,
            underlay: Some(render::StackLayer {
                vocabulary: render::StructureMode::HarmonicWeave,
                alpha: 0.29,
            }),
            accent: None,
        },
        render::LayerStack {
            primary: render::StructureMode::TimeChords,
            underlay: Some(render::StackLayer {
                vocabulary: render::StructureMode::HarmonicWeave,
                alpha: 0.24,
            }),
            accent: None,
        },
        render::LayerStack::solo(render::StructureMode::OrbitRibbons),
        render::LayerStack::solo(render::StructureMode::TimeChords),
    ] {
        if !stacks.contains(&fallback) {
            stacks.push(fallback);
        }
    }
    stacks
}

fn has_underlay(stack: render::LayerStack, vocabulary: render::StructureMode) -> bool {
    stack.underlay.is_some_and(|layer| layer.vocabulary == vocabulary)
}

fn retry_borda_weights(chaos_weight: f64, rng: &mut Sha3RandomByteStream) -> (f64, f64) {
    let descriptor = render::parameter_descriptors::EQUIL_CHAOS_RATIO;
    let log_min = descriptor.min.ln();
    let log_max = descriptor.max.ln();
    let ratio = (log_min + rng.next_f64() * (log_max - log_min)).exp();
    (chaos_weight, chaos_weight * ratio)
}

/// Prior bonus protecting seed identity during adaptive stack selection.
///
/// The seed's own stack gets the strongest bonus (larger for the newer
/// vocabularies so adaptive scoring cannot systematically homogenize the
/// population back to webs); fallbacks get small nudges. The strongest fallback
/// nudge goes to ribbon/chord + harmonic-weave stacks because the proxy score
/// otherwise underrates their lush coverage relative to sparse solo ribbons.
fn mode_selection_prior(stack: render::LayerStack, preferred: render::LayerStack) -> f64 {
    if stack == preferred {
        return match preferred.primary {
            render::StructureMode::OrbitRibbons
            | render::StructureMode::TimeChords
            | render::StructureMode::NebulaVeil
            | render::StructureMode::HarmonicWeave
            | render::StructureMode::StippleConstellation
            | render::StructureMode::TangentCaustics => 0.22,
            render::StructureMode::TriangleWeb
            | render::StructureMode::Duet { .. }
            | render::StructureMode::Spokes => 0.12,
        };
    }
    if has_underlay(stack, render::StructureMode::HarmonicWeave)
        && matches!(
            stack.primary,
            render::StructureMode::OrbitRibbons | render::StructureMode::TimeChords
        )
    {
        return 0.075;
    }
    if stack.primary == preferred.primary {
        // The seed's primary without its secondary layers.
        return 0.04;
    }
    match stack.primary {
        // Chord sails carry the family's calligraphic signature, so the
        // fallback gets a slightly stronger nudge when scores are comparable.
        render::StructureMode::TimeChords => 0.035,
        render::StructureMode::OrbitRibbons => {
            if stack.underlay.is_some() {
                -0.07
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

fn best_aesthetic_selection_from_shortlist(
    shortlist: Vec<sim::ShortlistedTrajectory>,
    num_steps_sim: usize,
    preferred_stack: render::LayerStack,
    projection: render::ProjectionMode,
    retry_count: usize,
) -> AestheticSelection {
    let stacks = adaptive_structure_stacks(preferred_stack);
    let mut best: Option<AestheticSelection> = None;

    for (rank, candidate) in shortlist.into_iter().enumerate() {
        let raw_positions = sim::get_positions(candidate.bodies.clone(), num_steps_sim).positions;
        // Score candidates in the projection space they will be rendered in,
        // so the aesthetic gate curates phase-space seeds too.
        let positions = apply_projection(&raw_positions, projection);
        for &stack in &stacks {
            let proxy_params = render::aesthetic_score::ProxyRenderParams::for_candidates(stack);
            let score = render::aesthetic_score::score_trajectory(&positions, proxy_params);
            let prior_bonus = mode_selection_prior(stack, preferred_stack);
            // No upper clamp: clamping at 1.0 used to collapse every strong
            // preferred-stack candidate to the same score, so the first tie in
            // iteration order won instead of the genuinely best orbit.
            let selection_score = (score.total + prior_bonus).max(0.0);
            info!(
                "   candidate {rank}: orbit idx {} stack={} borda {:.1} aesthetic {:.4} select {:.4} \
                 (coverage {:.3}, balance {:.3}, contrast {:.3}, mix {:.3}, mush {:.3}, veil {:.3}, crisp {:.3}, void {:.3}, full {:.3})",
                candidate.result.selected_index,
                stack.label(),
                candidate.result.total_score_weighted,
                score.total,
                selection_score,
                score.coverage,
                score.balance,
                score.contrast,
                score.body_mix,
                score.mush_fraction,
                score.veil_fraction,
                score.crispness,
                score.negative_space,
                score.fullness,
            );

            let improves = best.as_ref().is_none_or(|cur| selection_score > cur.selection_score);
            if improves {
                best = Some(AestheticSelection {
                    bodies: candidate.bodies.clone(),
                    result: candidate.result.clone(),
                    positions: positions.clone(),
                    aesthetic: score,
                    selection_score,
                    stack,
                    preferred_stack,
                    retry_count,
                });
            }
        }
    }

    best.expect("shortlist is non-empty by construction")
}

/// Run Borda selection, then pick the most paintable candidate and layer stack by proxy render.
///
/// The Borda search ranks orbits on physics proxies only; this pass re-simulates
/// the top [`AESTHETIC_SHORTLIST_LEN`] candidates (a negligible cost next to the
/// search itself), scores each one in image space under the seed's preferred
/// stack plus a curated set of high-yield alternatives, and returns the
/// highest-scoring `(orbit, stack)` pair together with its full trajectory in
/// the seed's projection space.
pub fn run_borda_selection_with_aesthetics(
    rng: &mut Sha3RandomByteStream,
    num_sims: usize,
    num_steps_sim: usize,
    chaos_weight: f64,
    equil_weight: f64,
    escape_threshold: f64,
    preferred_stack: render::LayerStack,
    projection: render::ProjectionMode,
) -> Result<AestheticSelection> {
    let stacks = adaptive_structure_stacks(preferred_stack);
    let mut global_best: Option<AestheticSelection> = None;

    for attempt in 0..=AESTHETIC_MAX_RETRIES {
        let (attempt_cw, attempt_ew, shortlist) = if attempt == 0 {
            info!("STAGE 1/7: Borda search over {} random orbits...", num_sims);
            (
                chaos_weight,
                equil_weight,
                sim::select_best_trajectory_shortlist(
                    rng,
                    num_sims,
                    num_steps_sim,
                    chaos_weight,
                    equil_weight,
                    escape_threshold,
                    AESTHETIC_SHORTLIST_LEN,
                )?,
            )
        } else {
            let domain = format!("{AESTHETIC_RETRY_RNG_DOMAIN_PREFIX}{attempt}");
            let mut retry_rng = rng.fork(domain.as_bytes());
            let (retry_cw, retry_ew) = retry_borda_weights(chaos_weight, &mut retry_rng);
            info!(
                "STAGE 1/7: Retry {attempt}/{AESTHETIC_MAX_RETRIES} over {num_sims} random orbits \
                 (chaos={retry_cw:.3}, equil={retry_ew:.3})..."
            );
            let shortlist = match sim::select_best_trajectory_shortlist(
                &mut retry_rng,
                num_sims,
                num_steps_sim,
                retry_cw,
                retry_ew,
                escape_threshold,
                AESTHETIC_SHORTLIST_LEN,
            ) {
                Ok(shortlist) => shortlist,
                Err(e) if global_best.is_some() => {
                    warn!("Aesthetic retry {attempt} produced no valid orbits: {e}");
                    continue;
                }
                Err(e) => return Err(e),
            };
            (retry_cw, retry_ew, shortlist)
        };

        info!(
            "STAGE 1.5/7: Aesthetic scoring of {} shortlisted candidate(s) across {} stack(s) [projection={}]: {}",
            shortlist.len(),
            stacks.len(),
            projection.label(),
            stacks.iter().map(render::LayerStack::label).collect::<Vec<_>>().join(", ")
        );
        let attempt_best = best_aesthetic_selection_from_shortlist(
            shortlist,
            num_steps_sim,
            preferred_stack,
            projection,
            attempt,
        );
        info!(
            "   => Attempt {attempt} winner: orbit idx {} stack={} score {:.4} (select {:.4}, chaos={attempt_cw:.3}, equil={attempt_ew:.3})",
            attempt_best.result.selected_index,
            attempt_best.stack.label(),
            attempt_best.aesthetic.total,
            attempt_best.selection_score,
        );

        let improves = global_best
            .as_ref()
            .is_none_or(|cur| attempt_best.selection_score > cur.selection_score);
        if improves {
            global_best = Some(attempt_best);
        }
        if global_best.as_ref().is_some_and(|best| best.aesthetic.total >= AESTHETIC_QUALITY_FLOOR)
        {
            break;
        }
    }

    let selection = global_best.expect("at least one Borda attempt must return candidates");
    info!(
        "   => Aesthetic winner: orbit idx {} stack={} preferred={} score {:.4} retries={}",
        selection.result.selected_index,
        selection.stack.label(),
        selection.preferred_stack.label(),
        selection.aesthetic.total,
        selection.retry_count,
    );
    Ok(selection)
}

/// Re-run the best orbit to get full trajectory
pub fn simulate_best_orbit(best_bodies: Vec<Body>, num_steps_sim: usize) -> Vec<Vec<Vector3<f64>>> {
    info!("STAGE 2/7: Re-running best orbit for {} steps...", num_steps_sim);
    let sim_result = sim::get_positions(best_bodies, num_steps_sim);
    info!("   => Done.");
    sim_result.positions
}

/// Number of seeded candidate viewing orientations evaluated per seed.
pub const VIEW_CANDIDATE_COUNT: usize = 4;

/// Build the rotation matrix for one Shoemake-uniform quaternion triple.
fn shoemake_rotation(u1: f64, u2: f64, u3: f64) -> Matrix3<f64> {
    let two_pi = crate::render::constants::TWO_PI;
    let (qx, qy) =
        ((1.0 - u1).sqrt() * (two_pi * u2).sin(), (1.0 - u1).sqrt() * (two_pi * u2).cos());
    let (qz, qw) = (u1.sqrt() * (two_pi * u3).sin(), u1.sqrt() * (two_pi * u3).cos());
    quaternion_to_matrix(qw, qx, qy, qz)
}

/// Rotate a strided sample of the trajectory (cheap copy for view scoring).
fn rotated_sample(
    positions: &[Vec<Vector3<f64>>],
    rotation: &Matrix3<f64>,
    samples_per_body: usize,
) -> Vec<Vec<Vector3<f64>>> {
    positions
        .iter()
        .map(|body| {
            let stride = (body.len() / samples_per_body.max(1)).max(1);
            body.iter().step_by(stride).map(|position| rotation * *position).collect()
        })
        .collect()
}

/// Rotate the whole trajectory by the best of several seeded 3D orientations.
///
/// The simulation produces fully three-dimensional structures, but the
/// renderer projects onto the fixed x/y plane. Without this step every seed
/// is photographed from the same axis. Instead of committing to a single
/// random angle, [`VIEW_CANDIDATE_COUNT`] Shoemake-uniform rotations are drawn
/// from the forked view RNG and each is scored on its projected 2D occupancy
/// (coverage / balance via [`render::aesthetic_score`]); the best-composed
/// viewpoint wins. Deterministic per seed.
///
/// Positions are already expressed in the centre-of-mass frame, so rotating
/// about the origin is rotation about the COM. Uses a forked RNG domain so it
/// does not perturb the main seed stream consumed by drift and colors.
///
/// Returns the winning rotation, which the ember edition re-applies to follow the same view
/// ([`ember_view`]).
pub fn apply_view_orientation(
    positions: &mut [Vec<Vector3<f64>>],
    rng: &Sha3RandomByteStream,
    stack: render::LayerStack,
) -> Matrix3<f64> {
    info!(
        "STAGE 2.25/7: Selecting best of {} seeded viewing orientations...",
        VIEW_CANDIDATE_COUNT
    );
    let mut view_rng = rng.fork(VIEW_RNG_DOMAIN);
    let proxy_params = render::aesthetic_score::ProxyRenderParams::for_view_selection(stack);

    let mut best_triple = (0.0, 0.0, 0.0);
    let mut best_score = f64::NEG_INFINITY;
    for candidate in 0..VIEW_CANDIDATE_COUNT {
        let triple = (view_rng.next_f64(), view_rng.next_f64(), view_rng.next_f64());
        let rotation = shoemake_rotation(triple.0, triple.1, triple.2);
        let sample = rotated_sample(positions, &rotation, proxy_params.samples_per_body);
        let score = render::aesthetic_score::score_trajectory(&sample, proxy_params).total;
        info!(
            "   view candidate {candidate}: ({:.3}, {:.3}, {:.3}) occupancy score {score:.4}",
            triple.0, triple.1, triple.2
        );
        if score > best_score {
            best_score = score;
            best_triple = triple;
        }
    }

    let rotation = shoemake_rotation(best_triple.0, best_triple.1, best_triple.2);
    for body_positions in positions.iter_mut() {
        for position in body_positions.iter_mut() {
            *position = rotation * *position;
        }
    }

    info!(
        "   => View quaternion components: ({:.3}, {:.3}, {:.3}) score {best_score:.4}",
        best_triple.0, best_triple.1, best_triple.2
    );
    rotation
}

fn quaternion_to_matrix(w: f64, x: f64, y: f64, z: f64) -> Matrix3<f64> {
    let (xx, yy, zz) = (x * x, y * y, z * z);
    let (xy, xz, yz) = (x * y, x * z, y * z);
    let (wx, wy, wz) = (w * x, w * y, w * z);

    Matrix3::new(
        1.0 - 2.0 * (yy + zz),
        2.0 * (xy - wz),
        2.0 * (xz + wy),
        2.0 * (xy + wz),
        1.0 - 2.0 * (xx + zz),
        2.0 * (yz - wx),
        2.0 * (xz - wy),
        2.0 * (yz + wx),
        1.0 - 2.0 * (xx + yy),
    )
}

/// The drift a run resolved and what it added to the positions.
#[derive(Clone, Debug)]
pub struct DriftOutcome {
    /// The resolved drift configuration (recorded in the generation log and the traits).
    pub config: ResolvedDriftConfig,
    /// What was added to every body at every step (re-applied by the ember edition).
    pub applied: AppliedDrift,
}

/// Apply drift transformation to positions
pub fn apply_drift_transformation(
    positions: &mut [Vec<Vector3<f64>>],
    drift_mode: &str,
    drift_scale: Option<f64>,
    drift_arc_fraction: Option<f64>,
    drift_orbit_eccentricity: Option<f64>,
    rng: &mut Sha3RandomByteStream,
) -> Result<DriftOutcome> {
    info!("STAGE 2.5/7: Resolving drift configuration...");

    let resolved =
        resolve_drift_config(drift_scale, drift_arc_fraction, drift_orbit_eccentricity, rng)?;

    info!("Applying {} drift...", drift_mode);
    let num_steps = positions[0].len();
    let drift_params = resolved.to_drift_parameters();

    if crate::utils::is_zero(drift_params.arc_fraction)
        && drift_mode.to_lowercase().starts_with("ell")
    {
        warn!("Elliptical drift requested with zero arc fraction; skipping motion");
    }

    let mut drift_transform = parse_drift_mode(drift_mode, rng, drift_params, num_steps)?;
    let applied = drift_transform.apply(positions, constants::DEFAULT_DT);

    info!("   => Drift applied successfully");
    Ok(DriftOutcome { config: resolved, applied })
}

/// Generate color sequences and alpha values for bodies
pub fn generate_colors(
    rng: &mut Sha3RandomByteStream,
    num_steps_sim: usize,
    alpha_denom: usize,
    enhancements: &Enhancements,
    palette_phase: f64,
) -> (Vec<Vec<render::OklabColor>>, Vec<f64>) {
    info!("STAGE 3/7: Generating color sequences + alpha...");
    generate_body_color_sequences(
        rng,
        num_steps_sim,
        alpha_denom,
        enhancements.chroma_boost,
        enhancements.alpha_variation,
        palette_phase,
    )
}

/// Build histogram and determine color levels.
///
/// `profile` supplies the seed's scene traits (the histogram must sample the
/// same structure mode that pass 2 renders) and the display exposure key:
/// values below 1 produce darker, ember-like seeds and values above 1
/// brighter, airier seeds.
pub fn build_histogram_and_levels(
    positions: &[Vec<Vector3<f64>>],
    colors: &[Vec<render::OklabColor>],
    body_alphas: &[f64],
    resolved_config: &render::randomizable_config::ResolvedEffectConfig,
    render_config: &RenderConfig,
    aspect_correction: bool,
    profile: &render::visual_profile::CosmicSignatureParameters,
) -> Result<ChannelLevels> {
    info!("STAGE 5/7: PASS 1 => building global histogram...");

    let target_frames = constants::DEFAULT_HISTOGRAM_SAMPLE_FRAMES;
    let frame_interval = (positions[0].len() / target_frames as usize).max(1);

    let histogram = pass_1_build_histogram_spectral(
        SpectralScene::new(positions, colors, body_alphas),
        frame_interval,
        SpectralRenderSettings::new(resolved_config, render_config, aspect_correction)
            .with_traits(profile.scene_traits()),
    );

    info!("STAGE 6/7: Determine global black/white/gamma...");
    let analysis = render::histogram::analyze_tonemapping(
        histogram.data(),
        resolved_config.clip_black,
        resolved_config.clip_white,
    );

    let exposure_key = profile.exposure_key;
    let keyed_exposure = analysis.exposure_scale * exposure_key.clamp(0.5, 1.5);
    info!(
        "   => R:[{:.3e},{:.3e}] G:[{:.3e},{:.3e}] B:[{:.3e},{:.3e}] exposure={:.3} key={:.3} near_clip={:.3}%",
        analysis.black_r,
        analysis.white_r,
        analysis.black_g,
        analysis.white_g,
        analysis.black_b,
        analysis.white_b,
        keyed_exposure,
        exposure_key,
        analysis.near_clip_ratio * constants::PERCENT_FACTOR
    );

    Ok(ChannelLevels::with_tone_mapping(
        analysis.black_r,
        analysis.white_r,
        analysis.black_g,
        analysis.white_g,
        analysis.black_b,
        analysis.white_b,
        ToneMappingControls {
            exposure_scale: keyed_exposure,
            paper_white: constants::DEFAULT_TONEMAP_PAPER_WHITE,
            highlight_rolloff: constants::DEFAULT_TONEMAP_HIGHLIGHT_ROLLOFF,
        },
    ))
}

/// Render full video, returning the fully accumulated SPD buffer for spectral outputs.
pub fn render_video(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    output_videos: VideoOutputPaths<'_>,
    output_images: ImageOutputPaths<'_>,
    fast_encode: bool,
) -> Result<Vec<[f64; crate::spectrum::NUM_BINS]>> {
    if fast_encode {
        info!("STAGE 7/7: PASS 2 => final frames => video (FAST ENCODE MODE)...");
    } else {
        info!("STAGE 7/7: PASS 2 => final frames => video (HIGH QUALITY MODE)...");
    }

    let frame_rate = constants::DEFAULT_VIDEO_FPS;
    // The ember edition renders the same checkpoints (`render::main_video_checkpoints`).
    let frame_interval = render::main_video_frame_interval(scene.step_count());

    let mut last_frame_png: Option<ImageBuffer<Rgb<u16>, Vec<u16>>> = None;
    let video_options = if fast_encode {
        VideoEncodingOptions::fast_encode()
    } else {
        VideoEncodingOptions::high_quality()
    };
    let video_outputs = [
        VideoOutputSpec {
            output_file: output_videos.web.to_string(),
            options: VideoEncodingOptions::web_compatible(),
        },
        VideoOutputSpec {
            output_file: output_videos.high_quality.to_string(),
            options: video_options,
        },
    ];

    let mut accum_spd = Vec::new();

    create_videos_from_frames_singlepass(
        settings.resolved_config.width,
        settings.resolved_config.height,
        frame_rate,
        |out| {
            pass_2_write_frames_spectral(
                render::Pass2Params {
                    scene,
                    frame_interval,
                    levels,
                    settings,
                    last_frame_out: &mut last_frame_png,
                    accum_spd: &mut accum_spd,
                },
                |buf_8bit| {
                    out.write_all(buf_8bit).map_err(render::error::RenderError::VideoEncoding)?;
                    Ok(())
                },
            )?;
            Ok(())
        },
        &video_outputs,
    )?;

    if let Some(frame) = last_frame_png {
        info!("Saving still image from final video frame: {}", output_images.master_png);
        save_image_as_png_16bit(&frame, output_images.master_png)?;
        generate_webp_images(output_images)?;
    } else {
        warn!("Warning: No final frame was generated to save as PNG.");
    }

    Ok(accum_spd)
}

/// Render only the fully accumulated still image, skipping all video outputs.
///
/// This is the fast-iteration path for parameter-tuning batches: the still is
/// identical in composition to the final video frame but avoids encoding the
/// trajectory video and the spectral sweep.
pub fn render_still_image(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    output_images: ImageOutputPaths<'_>,
) -> Result<()> {
    info!("STAGE 7/7: PASS 2 => final still only (IMAGE-ONLY MODE)...");
    let frame = render::render_final_frame_spectral(scene, levels, settings)?;
    info!("Saving still image: {}", output_images.master_png);
    save_image_as_png_16bit(&frame, output_images.master_png)?;
    generate_webp_images(output_images)
}

/// Generate the spectral gallery: 64 per-bin 16-bit PNGs in `spectral_dir`.
pub fn generate_spectral_gallery(
    accum_spd: &[[f64; crate::spectrum::NUM_BINS]],
    width: u32,
    height: u32,
    spectral_dir: &str,
) -> Result<()> {
    Ok(render::spectral_output::generate_spectral_gallery(accum_spd, width, height, spectral_dir)?)
}

/// Generate the spectral sweep video (violet-to-red-to-violet cycle) at `output_path`.
pub fn generate_spectral_sweep_video(
    accum_spd: &[[f64; crate::spectrum::NUM_BINS]],
    width: u32,
    height: u32,
    output_videos: VideoOutputPaths<'_>,
    fast_encode: bool,
) -> Result<()> {
    Ok(render::spectral_output::generate_spectral_sweep_video(
        accum_spd,
        width,
        height,
        output_videos.web,
        output_videos.high_quality,
        fast_encode,
    )?)
}

/// Output files of the ember edition (see the `EMBER_*_PATH` constants for their package paths).
#[derive(Clone, Copy, Debug)]
pub struct EmberOutputPaths<'a> {
    /// 16-bit sRGB PNG of the still (the final frame, orbit step `steps - 1`).
    pub still_png: &'a str,
    /// Full-resolution WebP of the still.
    pub full_webp: &'a str,
    /// Preview WebP of the still (at most [`WEB_PREVIEW_MAX_WIDTH`] wide).
    pub preview_webp: &'a str,
    /// Browser-compatible H.264 video (not written for image-only renders).
    pub web_video: &'a str,
    /// The slow film as browser-compatible H.264 (not written for image-only renders).
    pub slow_web_video: &'a str,
    /// Archival HEVC video, or the software fast encode (not written for image-only renders).
    pub hq_video: &'a str,
    /// Determinism certificate (`metadata/ember.json`).
    pub certificate: &'a str,
}

/// Inputs of [`render_ember_edition`].
#[derive(Clone, Copy, Debug)]
pub struct EmberEditionRequest<'a> {
    /// Package seed as hex (recorded in the certificate).
    pub seed_hex: &'a str,
    /// Package seed bytes (the kozo sheet's seed is derived from them).
    pub seed_bytes: &'a [u8],
    /// Initial conditions of the selected orbit, before the centre-of-mass shift.
    pub bodies: &'a [Body],
    /// The main edition's view of that orbit ([`ember_view`]), which the bodies follow.
    pub view: &'a View,
    /// Recorded orbit steps (the main renderer's `--steps`).
    pub steps: usize,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Render only the still (with its WebP derivatives and certificate); no videos.
    pub image_only: bool,
    /// Encode the HQ slot with the software fast encoder instead of archival HEVC.
    pub fast_encode: bool,
    /// Look and simulation parameters (the product renders [`EmberConfig::default`]); recorded
    /// in the certificate.
    pub config: &'a EmberConfig,
    /// Where to write the outputs.
    pub paths: EmberOutputPaths<'a>,
}

/// Seed of the ember edition's kozo sheet: the package seed bytes followed by
/// `"\0cosmic-ember/kozo-sheet/v1"`. Public so that a package can be re-rendered and verified
/// from its certificate (`examples/ember_render.rs`).
pub fn ember_paper_seed(seed_bytes: &[u8]) -> Vec<u8> {
    [seed_bytes, EMBER_PAPER_SEED_DOMAIN].concat()
}

/// The ember video's frame schedule: exactly `main.mp4`'s checkpoints
/// (`render::main_video_checkpoints`, the schedule `render_video` encodes), so frame `i` of
/// both videos shows the same orbit step (and the last frame shows step `steps - 1`).
pub fn ember_frame_schedule(steps: usize) -> Vec<usize> {
    render::main_video_checkpoints(steps)
}

/// What [`preflight_ember_edition`] established about the selected orbit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmberPreflight {
    /// Orbit duration `T` in fluid time units (the median body speed is the reference speed).
    pub duration: f64,
    /// Fluid time at which the bodies stop inking (`T - valve_lead`).
    pub valve_time: f64,
    /// Frames of the ember video; the last one is the still.
    pub frames: usize,
    /// Frames of the slow film.
    pub slow_frames: usize,
    /// Fluid steps the orbit needs at least (the render's cost grows with them).
    pub estimated_fluid_steps: f64,
    /// Fastest body speed on the canvas, in units of the median speed.
    pub peak_speed: f64,
    /// Smallest distance of a body's centre from the canvas edge (the canvas is 2 high).
    pub edge_clearance: f64,
    /// Fraction of the orbit during which two bodies overlap on the canvas.
    pub overlap_fraction: f64,
}

/// The three bodies' initial masses, which set the tidal field that stretches them in the ember
/// edition ([`ember::EmberRequest::masses`]).
///
/// # Errors
///
/// [`ember::EmberError::DegenerateOrbit`] unless there are exactly three bodies.
pub fn ember_masses(bodies: &[Body]) -> Result<[f64; 3]> {
    match bodies {
        [a, b, c] => Ok([a.mass, b.mass, c.mass]),
        _ => Err(ember::EmberError::DegenerateOrbit {
            reason: format!("the ember edition needs 3 bodies, got {}", bodies.len()),
        }
        .into()),
    }
}

/// The main edition's view of the selected orbit, for the ember edition to follow: the seed's
/// projection space, the viewing rotation [`apply_view_orientation`] chose, the drift
/// [`apply_drift_transformation`] added, and the frame the renderer fits to the transformed
/// orbit (`bounds`, with the scale its symmetry applies to the primary copy of every stroke).
///
/// # Errors
///
/// [`EmberError::InvalidView`] for a Brownian drift: its path is a seeded random walk that is
/// not recorded, so the ember edition cannot follow it.
pub fn ember_view(
    projection: render::ProjectionMode,
    rotation: &Matrix3<f64>,
    drift: AppliedDrift,
    bounds: &render::context::BoundingBox,
    symmetry: render::SymmetryOp,
    (width, height): (u32, u32),
) -> Result<View> {
    let rows = |m: &Matrix3<f64>| std::array::from_fn(|i| std::array::from_fn(|j| m[(i, j)]));
    let drift = match drift {
        AppliedDrift::None => ViewDrift::None {},
        AppliedDrift::Linear { velocity } => {
            ViewDrift::Linear { velocity: [velocity.x, velocity.y, velocity.z] }
        }
        AppliedDrift::Elliptical {
            rotation,
            initial_mean_anomaly,
            mean_motion,
            eccentricity,
            semi_major,
            semi_minor,
        } => ViewDrift::Elliptical {
            rotation: rows(&rotation),
            mean_anomaly: initial_mean_anomaly,
            mean_motion,
            eccentricity,
            semi_major,
            semi_minor,
        },
        AppliedDrift::Brownian => {
            return Err(EmberError::InvalidView {
                reason: "a brownian drift is a random walk whose path is not recorded; the ember \
                         edition follows no, linear or elliptical drift"
                    .into(),
            }
            .into());
        }
    };
    Ok(View {
        projection: match projection {
            render::ProjectionMode::Position => ViewProjection::Position,
            render::ProjectionMode::PhasePortrait => ViewProjection::PhasePortrait,
            render::ProjectionMode::CrossBraid => ViewProjection::CrossBraid,
            render::ProjectionMode::Hodograph => ViewProjection::Hodograph,
        },
        rotation: rows(rotation),
        drift,
        frame: ViewFrame {
            min_x: bounds.min_x,
            min_y: bounds.min_y,
            width: bounds.width,
            height: bounds.height,
            scale: f64::from(render::batch_drawing::primary_symmetry_scale(
                symmetry, width, height,
            )),
        },
    })
}

/// The view of an orbit that is drawn as it is: plain positions seen along the z axis, no
/// drift, framed as the main edition frames a trajectory. For renders outside the generator
/// (`examples/ember_render.rs`), which have no main edition to follow.
pub fn ember_frontal_view(positions: &[Vec<Vector3<f64>>], width: u32, height: u32) -> View {
    let context = render::context::RenderContext::new(width, height, positions, true);
    ember_view(
        render::ProjectionMode::Position,
        &Matrix3::identity(),
        AppliedDrift::None,
        context.bounds(),
        render::SymmetryOp::None,
        (width, height),
    )
    .expect("a view without drift is always representable")
}

/// Checks, in a fraction of a second, that the ember edition can render the selected orbit at
/// this output size with `config`, so that an orbit it would reject is known before the main
/// render rather than in the ember stage, which runs last.
///
/// Re-simulates the raw orbit exactly as [`render_ember_edition`] does and plans the render with
/// [`ember::plan_ember`] — the checks [`ember::render_ember`] itself makes before its fluid
/// starts: the configuration, the output size, the frame schedule (at least 2 recorded steps),
/// the view and the canvas track it gives the orbit, the orbit's duration against the pre-roll
/// and the valve, and the fluid and ink grids. Failures that only show while simulating (e.g. a
/// non-finite flow) cannot be predicted. The plan's figures are logged: they say what the render
/// will cost and how the orbit sits on the sheet.
pub fn preflight_ember_edition(
    bodies: &[Body],
    view: &View,
    steps: usize,
    width: u32,
    height: u32,
    config: &EmberConfig,
) -> Result<EmberPreflight> {
    let positions = sim::get_positions(bodies.to_vec(), steps).positions;
    let frame_steps = ember_frame_schedule(steps);
    let preflight = check_ember_request(&EmberRequest {
        positions: &positions,
        dt: constants::DEFAULT_DT,
        masses: ember_masses(bodies)?,
        view,
        frame_steps: &frame_steps,
        slow_factor: EMBER_SLOW_FACTOR,
        width,
        height,
        paper_seed: &[],
        config,
        mode: EmberMode::StillOnly,
    })?;
    info!(
        "   => Ember preflight: orbit lasts {:.3} fluid time units (inking {:.3}..{:.3}), {} \
         frames, {} in the slow film",
        preflight.duration,
        config.contact.pre_roll,
        preflight.valve_time,
        preflight.frames,
        preflight.slow_frames
    );
    info!(
        "   => Ember plan: at least {:.0} fluid steps; peak body speed {:.1}x the median; \
         bodies come within {:.3} of the sheet's edge and overlap for {:.1}% of the orbit",
        preflight.estimated_fluid_steps,
        preflight.peak_speed,
        preflight.edge_clearance,
        100.0 * preflight.overlap_fraction
    );
    Ok(preflight)
}

/// Plans an ember request ([`ember::plan_ember`]): the checks [`ember::render_ember`] makes
/// before its fluid starts. Cost: one pass of the view over the orbit and the track's tables (a
/// fraction of a second for a million steps), no allocation proportional to the output size.
fn check_ember_request(request: &EmberRequest<'_>) -> Result<EmberPreflight> {
    let plan = ember::plan_ember(request)?;
    Ok(EmberPreflight {
        duration: plan.duration(),
        valve_time: plan.valve_time(),
        frames: plan.frames(),
        slow_frames: plan.slow_frames(),
        estimated_fluid_steps: plan.estimated_fluid_steps(),
        peak_speed: plan.peak_speed(),
        edge_clearance: plan.edge_clearance(),
        overlap_fraction: plan.overlap_fraction(),
    })
}

/// Constant rate factor of the ember edition's web H.264.
///
/// Unlike `main.mp4`'s mostly black field, every ember frame is textured paper under a slowly
/// drifting ink wash, which is expensive to encode. Measured on a production package (3456 ×
/// 2234, 1802 frames): CRF 18 gave 141 MB (≈ 38 Mbit/s), CRF 22 84 MB and CRF 26 52 MB. CRF 22
/// is indistinguishable from CRF 18 in 1:1 crops, whereas CRF 26 starts to soften the kozo grain
/// and a 12 Mbit/s cap bands the wash. The archival HEVC copy keeps its CRF.
pub const EMBER_WEB_CRF: u32 = 22;

/// The ember edition's `[web, hq]` encodes: sRGB H.264 for the web (at [`EMBER_WEB_CRF`]) and
/// sRGB archival HEVC, or the software fast encode under `--fast-encode` (never a hardware
/// encoder: the ember edition is CPU-only).
fn ember_video_options(fast_encode: bool) -> [VideoEncodingOptions; 2] {
    let hq = if fast_encode {
        VideoEncodingOptions::software_fast_srgb()
    } else {
        VideoEncodingOptions::high_quality_srgb()
    };
    let web =
        VideoEncodingOptions { crf: EMBER_WEB_CRF, ..VideoEncodingOptions::web_compatible_srgb() };
    [web, hq]
}

/// Renders the ember edition of the selected orbit and writes its package files.
///
/// The orbit is re-simulated raw with [`sim::get_positions`] and the main edition's view is
/// re-applied to it with portable arithmetic (`ember::View`), so that the bodies move as they
/// do in `main.mp4` and every CPU reproduces the result bit for bit. Frames follow
/// `main.mp4`'s schedule at [`constants::DEFAULT_VIDEO_FPS`], with the slow film's in-between
/// frames ([`EMBER_SLOW_FACTOR`]); they are streamed to the encoders as they are shaded (unless
/// `image_only`), the final frame is saved as a 16-bit sRGB PNG with two WebP derivatives, and
/// `metadata/ember.json` certifies the SHA-256 digests of the raw frames of both films and of
/// the still.
///
/// An error can leave some outputs behind (partial videos, the PNG of a still whose WebP
/// derivatives failed, a truncated certificate); [`remove_ember_outputs`] deletes them.
pub fn render_ember_edition(request: &EmberEditionRequest<'_>) -> Result<EmberSummary> {
    let started = Instant::now();
    let mode = if request.image_only { EmberMode::StillOnly } else { EmberMode::VideoAndSlow };
    info!(
        "STAGE EMBER: ember edition ({}) — re-simulating the raw orbit ({} steps)...",
        if request.image_only { "still only" } else { "still + videos" },
        request.steps
    );
    let positions = sim::get_positions(request.bodies.to_vec(), request.steps).positions;
    let frame_steps = ember_frame_schedule(request.steps);
    let paper_seed = ember_paper_seed(request.seed_bytes);
    let ember_request = EmberRequest {
        positions: &positions,
        dt: constants::DEFAULT_DT,
        masses: ember_masses(request.bodies)?,
        view: request.view,
        frame_steps: &frame_steps,
        slow_factor: EMBER_SLOW_FACTOR,
        width: request.width,
        height: request.height,
        paper_seed: &paper_seed,
        config: request.config,
        mode,
    };
    // Cheap, and it must pass before the encoders are spawned (`main` has already run it).
    check_ember_request(&ember_request)?;

    let summary = if request.image_only {
        ember::render_ember(&ember_request, &mut |_| Ok(()))?
    } else {
        encode_ember_videos(
            &ember_request,
            request.paths,
            ember_video_options(request.fast_encode),
        )?
    };
    drop(positions);

    let paths = request.paths;
    save_image_as_srgb_png_16bit(&summary.still_image(), paths.still_png)?;
    generate_webp_images(ImageOutputPaths {
        master_png: paths.still_png,
        full_webp: paths.full_webp,
        preview_webp: paths.preview_webp,
    })?;

    let context = CertificateContext {
        seed: request.seed_hex,
        steps: request.steps,
        dt: constants::DEFAULT_DT,
        bodies: request.bodies,
        view: request.view,
        frame_steps: &frame_steps,
        frame_rate: constants::DEFAULT_VIDEO_FPS,
        paper_seed: &paper_seed,
        config: request.config,
    };
    EmberCertificate::new(&context, &summary)
        .write_json(std::path::Path::new(paths.certificate))?;
    info!("   Saved ember certificate => {}", paths.certificate);

    log_ember_summary(&summary, started.elapsed().as_secs_f64());
    Ok(summary)
}

/// Renders every frame straight into the encoders of the two films (`rgb48le` streams, no
/// temporary files) and returns the render's summary: the normal film goes to its web and HQ
/// encoders, the slow film to its web encoder.
///
/// Errors:
/// - a render failure (e.g. [`EmberError::NonFinite`]) is returned as such; the encoders are
///   killed and no video is finalised;
/// - an encoder of either film that dies mid-stream breaks its frame pipe: the video module's
///   error is returned, naming the encoder and its exit status followed by the pipe error, and
///   every other encoder is killed without finalising its file;
/// - an encoder that fails after the last frame is reported by the video module with its exit
///   status.
fn encode_ember_videos(
    request: &EmberRequest<'_>,
    paths: EmberOutputPaths<'_>,
    [web, hq]: [VideoEncodingOptions; 2],
) -> Result<EmberSummary> {
    let slow =
        [VideoOutputSpec { output_file: paths.slow_web_video.to_string(), options: web.clone() }];
    let normal = [
        VideoOutputSpec { output_file: paths.web_video.to_string(), options: web },
        VideoOutputSpec { output_file: paths.hq_video.to_string(), options: hq },
    ];
    let mut outcome: Option<EmberResult<EmberSummary>> = None;
    let encoded = create_video_groups_from_frames(
        request.width,
        request.height,
        constants::DEFAULT_VIDEO_FPS,
        &[&normal, &slow],
        |films| {
            let pipe = |e: std::io::Error| EmberError::Sink(format!("video encoder pipe: {e}"));
            let mut sink = |frame: &EmberFrame<'_>| -> EmberResult<()> {
                if frame.index.is_some() {
                    films[0].write_all(frame.rgb48le).map_err(pipe)?;
                }
                if frame.slow_index.is_some() {
                    films[1].write_all(frame.rgb48le).map_err(pipe)?;
                }
                Ok(())
            };
            let result = ember::render_ember(request, &mut sink);
            let stream = match &result {
                Ok(_) => Ok(()),
                Err(e) => Err(format!("ember render: {e}").into()),
            };
            outcome = Some(result);
            stream
        },
    );
    ember_encode_outcome(outcome, encoded)
}

/// Deletes every ember edition file ([`EMBER_OUTPUT_PATHS`]) from the package at `seed_dir`:
/// whatever a failed ember stage left behind (a partial PNG, WebP or video, a truncated
/// certificate) and any stale ember file of an earlier run into the same directory, so that the
/// package holds no ember file its asset manifest does not list. Files that do not exist are
/// skipped.
///
/// Every file is attempted; the first deletion error (with its path) is returned after the rest
/// have been tried.
pub fn remove_ember_outputs(seed_dir: &str) -> Result<()> {
    let mut first_error = None;
    for relative_path in EMBER_OUTPUT_PATHS {
        let path = format!("{seed_dir}/{relative_path}");
        match fs::remove_file(&path) {
            Ok(()) => info!("   Removed ember output => {path}"),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                warn!("   Could not remove ember output {path}: {error}");
                if first_error.is_none() {
                    first_error =
                        Some(ConfigError::FileSystem { operation: "remove".into(), path, error });
                }
            }
        }
    }
    first_error.map_or(Ok(()), |error| Err(error.into()))
}

/// Combines the ember render's result with the encoders' (see [`encode_ember_videos`]).
fn ember_encode_outcome(
    rendered: Option<EmberResult<EmberSummary>>,
    encoded: std::result::Result<(), render::error::RenderError>,
) -> Result<EmberSummary> {
    match (rendered, encoded) {
        (Some(Ok(summary)), Ok(())) => Ok(summary),
        // The frame pipe broke: the video module's error names the encoder that quit and its
        // exit status, and ends with the pipe error.
        (Some(Err(EmberError::Sink(_))), Err(encode_error)) => Err(encode_error.into()),
        (Some(Err(render_error)), _) => Err(render_error.into()),
        (_, Err(encode_error)) => Err(encode_error.into()),
        (None, Ok(())) => Err(AppError::Ember(EmberError::Sink(
            "the video encoders finished without requesting any frames".into(),
        ))),
    }
}

/// Logs what the ember stage produced and where the time went.
fn log_ember_summary(summary: &EmberSummary, stage_seconds: f64) {
    let stats = &summary.stats;
    let timings = &summary.timings;
    let seconds = |frames: usize| frames as f64 / f64::from(constants::DEFAULT_VIDEO_FPS);
    info!(
        "   => Ember edition: orbit {:.3} fluid time units (valve at {:.3}), fluid {}x{}, ink \
         nodes {}x{}, {} frames ({:.2}s of video), {} in the slow film ({:.2}s)",
        summary.duration,
        summary.valve_time,
        summary.fluid_grid[0],
        summary.fluid_grid[1],
        summary.ink_grid[0],
        summary.ink_grid[1],
        summary.frames_emitted,
        seconds(summary.frames_emitted),
        summary.slow_frames_emitted,
        seconds(summary.slow_frames_emitted),
    );
    info!(
        "   => Ember look: ink black for {:.3} and fading with tau {:.3} fluid time units; tidal \
         reference anisotropy {:.4e}",
        summary.hold_time, summary.fade_time, summary.tidal_reference,
    );
    info!(
        "   => Ember still: sha256 {} — ink on {:.1}% of nodes, {} gamut-mapped pixels",
        summary.still_sha256,
        100.0 * stats.still_ink_fraction,
        stats.still_gamut_mapped_pixels,
    );
    if let Some(frames_sha256) = &summary.frames_sha256 {
        info!("   => Ember frames: sha256 {frames_sha256} (rgb48le stream)");
    }
    if let Some(slow_sha256) = &summary.slow_frames_sha256 {
        info!("   => Ember slow film: sha256 {slow_sha256} (rgb48le stream)");
    }
    info!(
        "   => Ember work: {} fluid steps (dt {:.2e}..{:.2e}, max flow speed {:.2}), {} \
         snapshots, {} contact events",
        stats.fluid_steps,
        stats.min_dt,
        stats.max_dt,
        stats.max_flow_speed,
        stats.snapshots,
        stats.contact_events,
    );
    info!(
        "   => Ember timings: fluid {:.1}s, ink {:.1}s, shading {:.1}s, sink {:.1}s, render \
         {:.1}s; stage total {stage_seconds:.1}s (with orbit, PNG, WebP and certificate)",
        timings.fluid_seconds,
        timings.ink_seconds,
        timings.shade_seconds,
        timings.sink_seconds,
        timings.total_seconds,
    );
}

/// Log generation parameters for reproducibility.
///
/// Returns `Ok(())` on success, or an `Err` describing the I/O or serialisation failure.
pub fn log_generation(
    config: &GenerationLogConfig,
    file_name: &str,
    seed: &str,
    drift_config: &Option<ResolvedDriftConfig>,
    num_sims: usize,
    selection: &AestheticSelection,
    randomization_log: Option<&render::effect_randomizer::RandomizationLog>,
    package_record_path: Option<&str>,
) -> Result<()> {
    let logger = GenerationLogger::new();

    let mut record = GenerationRecord::new(file_name.to_string(), format!("0x{seed}"));

    record.render_config = LoggedRenderConfig {
        width: config.width,
        height: config.height,
        clip_black: config.clip_black,
        clip_white: config.clip_white,
        alpha_denom: config.alpha_denom,
        alpha_compress: config.alpha_compress,
        visual_profile: config.visual_profile.clone(),
        post_effects_enabled: config.post_effects_enabled,
        bloom_mode: config.bloom_mode.clone(),
        hdr_mode: config.hdr_mode.clone(),
        hdr_scale: config.hdr_scale,
        dispersion_strength: config.dispersion_strength,
        dispersion_mode: config.dispersion_mode.clone(),
        palette_fingerprint: config.palette_fingerprint.clone(),
        palette_gate: config.palette_gate.clone(),
    };

    record.drift_config = if let Some(drift) = drift_config {
        DriftConfig {
            enabled: true,
            mode: config.drift_mode.clone(),
            scale: drift.scale,
            arc_fraction: drift.arc_fraction,
            orbit_eccentricity: drift.orbit_eccentricity,
            randomized: drift.was_randomized,
        }
    } else {
        DriftConfig {
            enabled: false,
            mode: "none".to_string(),
            scale: 0.0,
            arc_fraction: 0.0,
            orbit_eccentricity: 0.0,
            randomized: false,
        }
    };

    record.simulation_config = SimulationConfig {
        num_sims,
        num_steps_sim: config.num_steps_sim,
        location: config.location,
        velocity: config.velocity,
        min_mass: config.min_mass,
        max_mass: config.max_mass,
        chaos_weight: config.chaos_weight,
        equil_weight: config.equil_weight,
        escape_threshold: config.escape_threshold,
        weights_randomized: config.weights_randomized,
    };

    let best_info = &selection.result;
    let score = selection.aesthetic;
    record.orbit_info = OrbitInfo {
        selected_index: best_info.selected_index,
        weighted_score: best_info.total_score_weighted,
        total_candidates: num_sims * (selection.retry_count + 1),
        discarded_count: best_info.discarded_count,
        preferred_structure: selection.preferred_stack.label(),
        chosen_structure: selection.stack.label(),
        retry_count: selection.retry_count,
        aesthetic_score: score.total,
        selection_score: selection.selection_score,
        coverage: score.coverage,
        balance: score.balance,
        contrast: score.contrast,
        body_mix: score.body_mix,
        mush_fraction: score.mush_fraction,
        veil_fraction: score.veil_fraction,
        crispness: score.crispness,
        negative_space: score.negative_space,
        fullness: score.fullness,
    };

    // Include randomization log if provided
    record.randomization_log = randomization_log.cloned();

    if let Some(path) = package_record_path {
        write_generation_record(path, &record)?;
    }

    logger.log_generation(record)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_seed_valid() {
        let result = parse_seed("0x100033");
        assert!(result.is_ok());

        let bytes = result.expect("hex bytes should parse");
        assert_eq!(bytes, vec![0x10, 0x00, 0x33]);
    }

    #[test]
    fn test_parse_seed_no_prefix() {
        let result = parse_seed("100033");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_seed_invalid() {
        let result = parse_seed("0xZZZ");
        assert!(result.is_err());
    }

    #[test]
    fn test_enhancements_default_quality_profile() {
        let e = Enhancements::default();
        assert!(e.chroma_boost);
        assert!(e.sat_boost);
        assert!(e.aces_tweak);
        assert!(e.alpha_variation);
        assert!(e.aspect_correction);
        assert!(!e.dispersion_boost);
    }

    #[test]
    fn test_enhancements_selective_disable() {
        let e = Enhancements {
            chroma_boost: false,
            sat_boost: true,
            aces_tweak: false,
            alpha_variation: true,
            aspect_correction: false,
            dispersion_boost: true,
        };
        assert!(!e.chroma_boost);
        assert!(e.sat_boost);
        assert!(!e.aces_tweak);
        assert!(e.alpha_variation);
        assert!(!e.aspect_correction);
        assert!(e.dispersion_boost);
    }

    #[test]
    fn test_generate_colors_with_enhancements() {
        use crate::sim::Sha3RandomByteStream;
        let mut rng = Sha3RandomByteStream::new(&[1, 2, 3, 4], 100.0, 300.0, 300.0, 1.0);
        let enhancements = Enhancements::default();
        let (colors, alphas) = generate_colors(&mut rng, 100, 15_000_000, &enhancements, 0.5);

        assert_eq!(colors.len(), 3);
        assert_eq!(alphas.len(), 3);
        for body_colors in &colors {
            assert_eq!(body_colors.len(), 100);
        }
        let unique: std::collections::HashSet<u64> = alphas.iter().map(|a| a.to_bits()).collect();
        assert!(unique.len() > 1, "default enhancements should enable alpha variation");
    }

    #[test]
    fn test_generate_colors_no_enhancements() {
        use crate::sim::Sha3RandomByteStream;
        let mut rng = Sha3RandomByteStream::new(&[1, 2, 3, 4], 100.0, 300.0, 300.0, 1.0);
        let enhancements =
            Enhancements { alpha_variation: false, chroma_boost: false, ..Enhancements::default() };
        let (colors, alphas) = generate_colors(&mut rng, 100, 15_000_000, &enhancements, 0.5);

        assert_eq!(colors.len(), 3);
        assert_eq!(alphas[0], alphas[1]);
        assert_eq!(alphas[1], alphas[2]);
    }

    /// Run the full seed-to-pixels pipeline at minimal scale and return the
    /// raw 16-bit pixel buffer.  Two calls with the same seed MUST return
    /// bitwise-identical buffers on the same architecture.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum PipelineRenderMode {
        DefaultParallel,
        SerialReference,
    }

    fn assert_pixel_buffers_eq(actual: &[u16], expected: &[u16], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: pixel buffer lengths differ");

        if actual != expected {
            let diff_count = actual.iter().zip(expected).filter(|(a, b)| a != b).count();
            let first = actual
                .iter()
                .zip(expected)
                .position(|(a, b)| a != b)
                .expect("expected differing pixel position");
            panic!(
                "{label}: pixel buffers differ: {diff_count} of {} values, \
                 first at index {first} ({} vs {})",
                actual.len(),
                actual[first],
                expected[first],
            );
        }
    }

    fn run_full_pipeline(seed: &[u8], mode: PipelineRenderMode) -> Vec<u16> {
        use crate::sim::Sha3RandomByteStream;

        let width = 64u32;
        let height = 36u32;
        let num_sims = 20;
        let num_steps = 5_000;

        let mut rng = Sha3RandomByteStream::new(seed, 100.0, 300.0, 300.0, 1.0);

        let resolved = render::randomizable_config::ResolvedEffectConfig {
            width,
            height,
            hdr_scale: 0.12,
            clip_black: 0.01,
            clip_white: 0.99,
            ..Default::default()
        };

        let (best_bodies, _) =
            crate::sim::select_best_trajectory(&mut rng, num_sims, num_steps, 0.75, 11.0, -0.3)
                .expect("Borda search should find at least one valid orbit");

        let mut positions = simulate_best_orbit(best_bodies, num_steps);

        apply_drift_transformation(&mut positions, "elliptical", None, None, None, &mut rng)
            .expect("drift config resolution should succeed with all-None args");

        let enhancements = Enhancements {
            chroma_boost: false,
            sat_boost: false,
            aces_tweak: false,
            alpha_variation: false,
            aspect_correction: false,
            dispersion_boost: false,
        };
        let (colors, body_alphas) =
            generate_colors(&mut rng, num_steps, 15_000_000, &enhancements, 0.5);

        let render_config =
            render::RenderConfig { hdr_scale: resolved.hdr_scale, ..Default::default() };
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let settings = SpectralRenderSettings::new(&resolved, &render_config, false);
        let frame_interval = (scene.step_count()
            / render::constants::DEFAULT_HISTOGRAM_SAMPLE_FRAMES as usize)
            .max(1);
        let histogram = match mode {
            PipelineRenderMode::DefaultParallel => {
                render::pass_1_build_histogram_spectral(scene, frame_interval, settings)
            }
            PipelineRenderMode::SerialReference => {
                render::pass_1_build_histogram_spectral_serial_reference(
                    scene,
                    frame_interval,
                    settings,
                )
            }
        };
        let analysis = render::histogram::analyze_tonemapping(
            histogram.data(),
            resolved.clip_black,
            resolved.clip_white,
        );
        let levels = render::ChannelLevels::with_tone_mapping(
            analysis.black_r,
            analysis.white_r,
            analysis.black_g,
            analysis.white_g,
            analysis.black_b,
            analysis.white_b,
            render::ToneMappingControls {
                exposure_scale: analysis.exposure_scale,
                paper_white: render::constants::DEFAULT_TONEMAP_PAPER_WHITE,
                highlight_rolloff: render::constants::DEFAULT_TONEMAP_HIGHLIGHT_ROLLOFF,
            },
        );

        let image = match mode {
            PipelineRenderMode::DefaultParallel => {
                render::render_single_frame_spectral(scene, &levels, settings)
            }
            PipelineRenderMode::SerialReference => {
                render::render_single_frame_spectral_serial_reference(scene, &levels, settings)
            }
        }
        .expect("render should succeed");

        image.into_raw()
    }

    #[test]
    fn test_end_to_end_pipeline_determinism() {
        for seed in [[0xCA, 0xFE], [0xBE, 0xEF], [0x12, 0x34]] {
            let pixels_a = run_full_pipeline(&seed, PipelineRenderMode::DefaultParallel);
            let pixels_b = run_full_pipeline(&seed, PipelineRenderMode::DefaultParallel);
            assert_pixel_buffers_eq(&pixels_a, &pixels_b, &format!("default_parallel/{seed:02X?}"));
        }
    }

    #[test]
    fn test_setup_seed_directory_returns_correct_path() {
        let result = setup_seed_directory("test_seed_42");
        assert!(result.is_ok());
        let seed_dir = result.expect("seed directory setup should succeed");
        assert_eq!(seed_dir, "output/test_seed_42");
        assert!(std::path::Path::new("output/test_seed_42").is_dir());
        assert!(std::path::Path::new("output/test_seed_42/images/source").is_dir());
        assert!(std::path::Path::new("output/test_seed_42/images/web").is_dir());
        assert!(std::path::Path::new("output/test_seed_42/videos/web").is_dir());
        assert!(std::path::Path::new("output/test_seed_42/videos/hq").is_dir());
        assert!(std::path::Path::new("output/test_seed_42/spectral").is_dir());
        assert!(std::path::Path::new("output/test_seed_42/metadata").is_dir());
        let _ = fs::remove_dir_all("output/test_seed_42");
    }

    #[test]
    fn test_setup_seed_directory_idempotent() {
        let r1 = setup_seed_directory("seed_idem");
        let r2 = setup_seed_directory("seed_idem");
        assert!(r1.is_ok());
        assert!(r2.is_ok());
        let _ = fs::remove_dir_all("output/seed_idem");
    }

    proptest::proptest! {
        #[test]
        fn proptest_parse_seed_never_panics(input in "\\PC*") {
            let _ = parse_seed(&input);
        }
    }

    fn projection_fixture() -> Vec<Vec<Vector3<f64>>> {
        (0..3)
            .map(|body| {
                let phase = f64::from(body as u32) * 2.1;
                (0..256)
                    .map(|step| {
                        let t = f64::from(step) * 0.05;
                        Vector3::new(
                            (t + phase).cos() * (1.0 + 0.3 * f64::from(body as u32)),
                            (t * 1.3 + phase).sin(),
                            0.2 * (t * 0.7).sin(),
                        )
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_position_projection_is_identity() {
        let positions = projection_fixture();
        let projected = apply_projection(&positions, render::ProjectionMode::Position);
        assert_eq!(projected, positions);
    }

    #[test]
    fn test_phase_projections_are_finite_deterministic_and_distinct() {
        let positions = projection_fixture();
        for projection in [
            render::ProjectionMode::PhasePortrait,
            render::ProjectionMode::CrossBraid,
            render::ProjectionMode::Hodograph,
        ] {
            let a = apply_projection(&positions, projection);
            let b = apply_projection(&positions, projection);
            assert_eq!(a, b, "{projection:?} must be deterministic");

            assert_eq!(a.len(), positions.len());
            let mut differs = false;
            for (body_idx, body) in a.iter().enumerate() {
                assert_eq!(body.len(), positions[body_idx].len());
                for (step, p) in body.iter().enumerate() {
                    assert!(
                        p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
                        "{projection:?} body {body_idx} step {step} not finite: {p:?}"
                    );
                    if (p - positions[body_idx][step]).norm() > 1e-9 {
                        differs = true;
                    }
                }
            }
            assert!(differs, "{projection:?} must differ from position space");
        }
    }

    #[test]
    fn test_phase_projection_velocity_axes_match_position_scale() {
        // The velocity axis of the phase portrait must be rescaled into the
        // same magnitude band as the position axes, or the composition would
        // collapse onto a line.
        let positions = projection_fixture();
        let projected = apply_projection(&positions, render::ProjectionMode::PhasePortrait);

        let extent = |data: &[Vec<Vector3<f64>>], axis: usize| {
            let mut min = f64::INFINITY;
            let mut max = f64::NEG_INFINITY;
            for body in data {
                for p in body {
                    min = min.min(p[axis]);
                    max = max.max(p[axis]);
                }
            }
            max - min
        };
        let x_extent = extent(&projected, 0);
        let v_extent = extent(&projected, 1);
        assert!(
            v_extent > x_extent * 0.05 && v_extent < x_extent * 20.0,
            "velocity axis badly scaled: x={x_extent} v={v_extent}"
        );
    }

    #[test]
    fn test_projection_handles_degenerate_inputs() {
        let empty: Vec<Vec<Vector3<f64>>> = vec![Vec::new(), Vec::new(), Vec::new()];
        let projected = apply_projection(&empty, render::ProjectionMode::Hodograph);
        assert_eq!(projected.len(), 3);

        let single = vec![
            vec![Vector3::new(1.0, 2.0, 3.0)],
            vec![Vector3::new(-1.0, 0.0, 1.0)],
            vec![Vector3::new(0.0, 1.0, -1.0)],
        ];
        let projected = apply_projection(&single, render::ProjectionMode::PhasePortrait);
        assert_eq!(projected, single, "single-step trajectories pass through unchanged");
    }

    #[test]
    fn test_adaptive_stacks_include_seed_stack_and_fallbacks() {
        let preferred = render::LayerStack::with_underlay(
            render::StructureMode::NebulaVeil,
            render::StructureMode::StippleConstellation,
            0.25,
        );
        let stacks = adaptive_structure_stacks(preferred);

        assert_eq!(stacks[0], preferred, "seed stack must be evaluated first");
        assert!(
            stacks.contains(&render::LayerStack::solo(render::StructureMode::NebulaVeil)),
            "primary-only variant must be a fallback"
        );
        assert!(
            stacks.contains(&render::LayerStack::solo(render::StructureMode::OrbitRibbons)),
            "ribbons must be a reliable fallback"
        );
        assert!(
            stacks.contains(&render::LayerStack::with_underlay(
                render::StructureMode::OrbitRibbons,
                render::StructureMode::HarmonicWeave,
                0.29,
            )),
            "target-like ribbon+weave stack must be evaluated"
        );
        assert!(stacks.len() <= 6, "adaptive evaluation must stay bounded: {}", stacks.len());

        // No duplicates.
        for (i, stack) in stacks.iter().enumerate() {
            for other in &stacks[i + 1..] {
                assert_ne!(stack, other, "duplicate stack in adaptive list");
            }
        }
    }

    #[test]
    fn test_mode_selection_prior_protects_new_vocabularies() {
        let veil = render::LayerStack::solo(render::StructureMode::NebulaVeil);
        let web = render::LayerStack::solo(render::StructureMode::TriangleWeb);
        let ribbons = render::LayerStack::solo(render::StructureMode::OrbitRibbons);

        assert!(
            mode_selection_prior(veil, veil) > mode_selection_prior(web, web),
            "novel vocabularies need a stronger identity prior than the classic web"
        );
        assert!(
            mode_selection_prior(veil, veil) > mode_selection_prior(ribbons, veil),
            "the seed's own stack must outrank fallbacks at equal aesthetic score"
        );
    }

    #[test]
    fn test_view_orientation_is_deterministic_and_rigid() {
        let make_positions = || -> Vec<Vec<Vector3<f64>>> {
            vec![
                vec![Vector3::new(1.0, 2.0, 3.0), Vector3::new(4.0, 5.0, 6.0)],
                vec![Vector3::new(-1.0, 0.5, 2.0), Vector3::new(0.0, -3.0, 1.0)],
                vec![Vector3::new(7.0, -2.0, 0.0), Vector3::new(-4.0, 1.0, -1.0)],
            ]
        };

        let rng = Sha3RandomByteStream::new(&[0xAB, 0xCD], 100.0, 300.0, 300.0, 1.0);
        let mut a = make_positions();
        let mut b = make_positions();
        let original = make_positions();

        let stack = render::LayerStack::solo(render::StructureMode::TriangleWeb);
        apply_view_orientation(&mut a, &rng, stack);
        apply_view_orientation(&mut b, &rng, stack);

        for body in 0..3 {
            for step in 0..2 {
                // Deterministic: identical results across calls with the same seed.
                assert_eq!(a[body][step], b[body][step], "body {body} step {step} diverged");
                // Rigid: rotation preserves distance from the origin (COM).
                let norm_before = original[body][step].norm();
                let norm_after = a[body][step].norm();
                assert!(
                    (norm_before - norm_after).abs() < 1e-9,
                    "rotation must preserve norms: {norm_before} vs {norm_after}"
                );
            }
        }

        // The orientation should actually rotate (not be the identity).
        let moved = (0..3).any(|body| (a[body][0] - original[body][0]).norm() > 1e-6);
        assert!(moved, "seeded view orientation should differ from identity");

        // A different seed must produce a different orientation.
        let rng2 = Sha3RandomByteStream::new(&[0x11, 0x22], 100.0, 300.0, 300.0, 1.0);
        let mut c = make_positions();
        apply_view_orientation(&mut c, &rng2, stack);
        let differs = (0..3).any(|body| (a[body][0] - c[body][0]).norm() > 1e-6);
        assert!(differs, "different seeds should view from different angles");
    }

    /// A raw orbit without any symmetry: three bodies on different lopsided loops away from the
    /// origin, moving in all three dimensions.
    fn lopsided_orbit(steps: usize) -> Vec<Vec<Vector3<f64>>> {
        (0..3u32)
            .map(|body| {
                let b = f64::from(body);
                (0..steps)
                    .map(|step| {
                        let t = step as f64 / steps as f64;
                        let a = std::f64::consts::TAU * (1.0 + 0.5 * b) * t + 0.9 * b;
                        Vector3::new(
                            (2.0 + 0.4 * b) * a.cos() + 0.7 * (3.0 * a).sin() + 1.5 - b,
                            (1.2 - 0.2 * b) * a.sin() + 0.3 * (2.0 * a).cos() + 2.0 * t,
                            0.8 * (1.7 * a).sin() - 0.5 * t + 0.3 * b,
                        )
                    })
                    .collect()
            })
            .collect()
    }

    /// The ember edition's bodies move as the main edition draws them: for every projection
    /// space, drift and symmetry, on a wide and on a tall sheet, the canvas track of the captured
    /// view lands on the pixel where the main edition draws the head of each trail (the primary
    /// symmetry copy), to within a billionth of a pixel at every step.
    #[test]
    fn test_ember_bodies_follow_the_main_edition_view() {
        use render::{ProjectionMode, SymmetryOp};

        let raw = lopsided_orbit(600);
        let stack = render::LayerStack::solo(render::StructureMode::TriangleWeb);
        let projections = [
            ProjectionMode::Position,
            ProjectionMode::PhasePortrait,
            ProjectionMode::CrossBraid,
            ProjectionMode::Hodograph,
        ];
        let symmetries = [
            SymmetryOp::None,
            SymmetryOp::MirrorX,
            SymmetryOp::Rotational { k: 3 },
            SymmetryOp::Dihedral { k: 4 },
        ];
        let mut worst = 0.0_f64;
        for (case, projection) in projections.into_iter().enumerate() {
            // The seeded defaults of each mode, an elliptical drift that sweeps a turn and a half
            // on an eccentric ellipse (its mean anomaly wraps), and one with no arc, which adds
            // nothing: the view records what was added, not what was configured.
            let drifts: [(&str, Option<[f64; 3]>); 5] = [
                ("none", None),
                ("linear", None),
                ("elliptical", None),
                ("elliptical", Some([1.3, crate::drift::MAX_ARC_FRACTION, 0.9])),
                ("elliptical", Some([1.0, 0.0, 0.3])),
            ];
            for (index, (drift_mode, explicit)) in drifts.into_iter().enumerate() {
                // The main pipeline, as `main` runs it.
                let seed = [case as u8, index as u8];
                let mut rng = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
                let mut positions = apply_projection(&raw, projection);
                let rotation = apply_view_orientation(&mut positions, &rng, stack);
                let [scale, arc, eccentricity] = match explicit {
                    Some(values) => values.map(Some),
                    None => [None; 3],
                };
                let drift = apply_drift_transformation(
                    &mut positions,
                    drift_mode,
                    scale,
                    arc,
                    eccentricity,
                    &mut rng,
                )
                .expect("the drift resolves")
                .applied;
                let adds_nothing = drift_mode == "none" || arc == Some(0.0);
                assert_eq!(drift == AppliedDrift::None, adds_nothing, "{drift_mode} {explicit:?}");

                for symmetry in symmetries {
                    for (width, height) in [(96_u32, 64_u32), (64, 96)] {
                        let label = format!(
                            "{projection:?}, {drift_mode} {explicit:?}, {symmetry:?}, {width}x{height}"
                        );
                        let context =
                            render::context::RenderContext::new(width, height, &positions, true);
                        let bounds = context.bounds();
                        let view = ember_view(
                            projection,
                            &rotation,
                            drift,
                            bounds,
                            symmetry,
                            (width, height),
                        )
                        .expect("the view is representable");
                        let (w, h) = (f64::from(width), f64::from(height));
                        let track = view
                            .canvas_track(&raw, constants::DEFAULT_DT, w / h)
                            .unwrap_or_else(|e| panic!("{label}: {e}"));
                        let scale = f64::from(render::batch_drawing::primary_symmetry_scale(
                            symmetry, width, height,
                        ));
                        for (body, points) in track.iter().enumerate() {
                            for (step, &[x, y]) in points.iter().enumerate() {
                                // Main: the frame's pixel, then the primary symmetry copy's
                                // scale about the frame centre.
                                let p = positions[body][step];
                                let (nx, ny) = bounds.normalize(p.x, p.y);
                                let main = [
                                    0.5 * w + scale * (nx * w - 0.5 * w),
                                    0.5 * h + scale * (ny * h - 0.5 * h),
                                ];
                                // Ember: the canvas [-aspect, aspect] × [-1, 1], y up, on the
                                // same sheet.
                                let ember = [(x / (w / h) + 1.0) * 0.5 * w, (1.0 - y) * 0.5 * h];
                                let error =
                                    (main[0] - ember[0]).abs().max((main[1] - ember[1]).abs());
                                assert!(
                                    error < 1e-9,
                                    "{label}: body {body} step {step}: main {main:?}, ember {ember:?}"
                                );
                                worst = worst.max(error);
                            }
                        }
                    }
                }
            }
        }
        println!("largest difference between main and ember: {worst:e} px");
    }

    /// Left stays left and up stays up: in a frontal view, the body that the main edition draws
    /// nearest the top left corner of the image is nearest the canvas's top left for ember.
    #[test]
    fn test_ember_frontal_view_keeps_the_image_orientation() {
        // Body 0 goes right and (in image rows) down, body 1 the other way, body 2 stays put.
        let raw = vec![
            vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(3.0, 1.0, 0.0)],
            vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(-3.0, -1.0, 0.0)],
            vec![Vector3::new(0.0, 0.0, 5.0), Vector3::new(0.0, 0.0, -5.0)],
        ];
        let (width, height) = (300, 200);
        let view = ember_frontal_view(&raw, width, height);
        let track = view.canvas_track(&raw, constants::DEFAULT_DT, 1.5).expect("a track");
        let context = render::context::RenderContext::new(width, height, &raw, true);
        // The main edition: body 0 ends right of and below the centre (image y points down).
        let (px, py) = context.to_pixel(3.0, 1.0);
        assert!(px > 150.0 && py > 100.0, "({px}, {py})");
        // Ember: right of and below the centre too (canvas y points up).
        assert!(track[0][1][0] > 0.0 && track[0][1][1] < 0.0, "{:?}", track[0][1]);
        assert!(track[1][1][0] < 0.0 && track[1][1][1] > 0.0, "{:?}", track[1][1]);
        assert_eq!(track[2], vec![[0.0, 0.0], [0.0, 0.0]]);
        // The 5% margin of the main frame on the long axis: the bodies stop short of the edge.
        assert!((track[0][1][0] - 1.5 / 1.1).abs() < 1e-12, "{:?}", track[0][1]);
    }

    #[test]
    fn test_end_to_end_pipeline_parallel_matches_serial_reference() {
        for seed in [[0xCA, 0xFE], [0xBE, 0xEF], [0x12, 0x34]] {
            let parallel_pixels = run_full_pipeline(&seed, PipelineRenderMode::DefaultParallel);
            let serial_pixels = run_full_pipeline(&seed, PipelineRenderMode::SerialReference);
            assert_pixel_buffers_eq(
                &parallel_pixels,
                &serial_pixels,
                &format!("parallel_vs_serial_reference/{seed:02X?}"),
            );
        }
    }

    // ---------------------------------------------------------------------------------------
    // Ember edition plumbing and the asset manifest
    // ---------------------------------------------------------------------------------------

    #[test]
    fn test_ember_paper_seed_is_the_package_seed_in_its_own_domain() {
        let seed = [0x10, 0x00, 0x33];
        let paper = ember_paper_seed(&seed);
        assert_eq!(paper, b"\x10\x00\x33\0cosmic-ember/kozo-sheet/v1");
        assert_ne!(ember_paper_seed(&[0x10, 0x00, 0x34]), paper, "seeds must not share a sheet");
        assert_eq!(ember_paper_seed(&seed), paper, "deterministic");
    }

    #[test]
    fn test_ember_schedule_is_the_main_video_schedule() {
        let production = ember_frame_schedule(1_000_000);
        assert_eq!(production.len(), 1_802);
        assert_eq!(production.first(), Some(&555));
        assert_eq!(production.last(), Some(&999_999), "the last frame is the still");
        for steps in [2, 17, 1_800, 1_801, 3_600, 123_457] {
            // `render_video` encodes `main.mp4` at `render::main_video_frame_interval`, whose
            // pass-2 frames are exactly `main_video_checkpoints` (tested in `render`).
            let schedule = ember_frame_schedule(steps);
            assert_eq!(schedule, render::main_video_checkpoints(steps), "{steps} steps");
            assert_eq!(schedule.last(), Some(&(steps - 1)), "{steps} steps");
            assert!(schedule.windows(2).all(|pair| pair[0] < pair[1]), "{steps} steps");
        }
    }

    /// The frontal view of the orbit `bodies` recorded for `steps` steps: what a render outside
    /// the generator follows ([`ember_frontal_view`]).
    fn frontal_view(bodies: &[Body], steps: usize, width: u32, height: u32) -> View {
        ember_frontal_view(&sim::get_positions(bodies.to_vec(), steps).positions, width, height)
    }

    /// Three bodies of an ordinary (non-degenerate) configuration.
    fn preflight_bodies() -> Vec<Body> {
        vec![
            Body::new(150.0, Vector3::new(120.0, -40.0, 30.0), Vector3::new(0.1, 0.8, -0.2)),
            Body::new(220.0, Vector3::new(-90.0, 60.0, -20.0), Vector3::new(-0.5, -0.2, 0.3)),
            Body::new(180.0, Vector3::new(-30.0, -110.0, 50.0), Vector3::new(0.4, -0.4, 0.1)),
        ]
    }

    #[test]
    fn test_ember_preflight_rejects_what_the_renderer_rejects_before_projecting() {
        // These fail before the orbit is projected (so before any real work).
        let bodies = preflight_bodies();
        let config = EmberConfig::default();
        for (steps, width, height) in [(1, 64, 40), (0, 64, 40), (100, 0, 40), (100, 16_385, 40)] {
            let result = preflight_ember_edition(
                &bodies,
                &frontal_view(&bodies, steps, width, height),
                steps,
                width,
                height,
                &config,
            );
            assert!(result.is_err(), "{steps} steps at {width}x{height} must be rejected");
        }
        assert!(matches!(
            preflight_ember_edition(&bodies, &frontal_view(&bodies, 1, 64, 40), 1, 64, 40, &config),
            Err(AppError::Ember(EmberError::InvalidSchedule { .. }))
        ));
        // The configuration under test is the one checked.
        let mut invalid = EmberConfig::default();
        invalid.fluid.cfl = 0.0;
        assert!(matches!(
            preflight_ember_edition(
                &figure_eight_bodies(),
                &frontal_view(&figure_eight_bodies(), 20_000, 64, 40),
                20_000,
                64,
                40,
                &invalid
            ),
            Err(AppError::Ember(EmberError::InvalidConfig { .. }))
        ));
    }

    /// The figure-eight choreography (Chenciner–Montgomery), unit masses, rescaled from `G = 1`
    /// to the simulator's `G` (velocities × √G, so the period is 6.3259/√G ≈ 2.02 time units).
    fn figure_eight_bodies() -> Vec<Body> {
        let speed = sim::G.sqrt();
        let (x, y) = (0.970_004_36, -0.243_087_53);
        let (vx, vy) = (-0.932_407_37 * speed, -0.864_731_46 * speed);
        vec![
            Body::new(1.0, Vector3::new(x, y, 0.0), Vector3::new(-vx / 2.0, -vy / 2.0, 0.0)),
            Body::new(1.0, Vector3::new(-x, -y, 0.0), Vector3::new(-vx / 2.0, -vy / 2.0, 0.0)),
            Body::new(1.0, Vector3::zeros(), Vector3::new(vx, vy, 0.0)),
        ]
    }

    #[test]
    fn test_ember_preflight_accepts_a_long_enough_orbit_and_rejects_a_short_one() {
        let config = EmberConfig::default();
        let contact = &config.contact;
        // About 10 periods: the bodies travel far across the canvas.
        let steps = 20_000;
        let preflight = preflight_ember_edition(
            &figure_eight_bodies(),
            &frontal_view(&figure_eight_bodies(), steps, 64, 40),
            steps,
            64,
            40,
            &config,
        )
        .expect("ok");
        assert!(preflight.duration > contact.pre_roll + contact.valve_lead, "{preflight:?}");
        assert_eq!(preflight.valve_time, preflight.duration - contact.valve_lead);
        assert_eq!(preflight.frames, render::main_video_checkpoints(steps).len());
        // A sliver of a slow orbit lasts far too little fluid time.
        assert!(matches!(
            preflight_ember_edition(
                &preflight_bodies(),
                &frontal_view(&preflight_bodies(), 100, 64, 40),
                100,
                64,
                40,
                &config
            ),
            Err(AppError::Ember(EmberError::OrbitTooShort { .. }))
        ));
    }

    fn encode_failure(message: &str) -> render::error::RenderError {
        render::error::RenderError::VideoEncoding(std::io::Error::other(message.to_string()))
    }

    #[test]
    fn test_ember_encode_outcome_reports_the_cause() {
        let quit = "FFmpeg for web/ember.mp4 exited early with exit status: 1; the frame stream \
                    then failed: ember render: frame sink failed: video encoder pipe: Broken pipe";
        let pipe = || Some(Err(EmberError::Sink("video encoder pipe: Broken pipe".into())));

        // An encoder that died mid-stream: the video module's error, which names it.
        match ember_encode_outcome(pipe(), Err(encode_failure(quit))) {
            Err(AppError::RenderInternal(error)) => {
                let source = std::error::Error::source(&error).expect("I/O source").to_string();
                assert_eq!(source, quit);
            }
            other => panic!("expected the encoder's error, got {other:?}"),
        }
        // A render failure is reported as such, even though it also stopped the encoders.
        let non_finite = Some(Err(EmberError::NonFinite { stage: "fluid", time: 1.5 }));
        assert!(matches!(
            ember_encode_outcome(non_finite, Err(encode_failure("ember render: non-finite"))),
            Err(AppError::Ember(EmberError::NonFinite { .. }))
        ));
        // An encoder failing after the last frame.
        assert!(matches!(
            ember_encode_outcome(Some(Ok(summary_with_frames(3))), Err(encode_failure("late"))),
            Err(AppError::RenderInternal(_))
        ));
        assert_eq!(
            ember_encode_outcome(Some(Ok(summary_with_frames(3))), Ok(()))
                .expect("both succeeded")
                .frames_emitted,
            3
        );
        assert!(matches!(
            ember_encode_outcome(None, Ok(())),
            Err(AppError::Ember(EmberError::Sink(_)))
        ));
        assert!(matches!(
            ember_encode_outcome(None, Err(encode_failure("spawn"))),
            Err(AppError::RenderInternal(_))
        ));
    }

    #[test]
    fn test_ember_video_options_are_srgb_and_software_only() {
        for fast_encode in [false, true] {
            let [web, hq] = ember_video_options(fast_encode);
            let expected_web = VideoEncodingOptions::web_compatible_srgb();
            let expected_hq = if fast_encode {
                VideoEncodingOptions::software_fast_srgb()
            } else {
                VideoEncodingOptions::high_quality_srgb()
            };
            for (options, expected) in [(&web, &expected_web), (&hq, &expected_hq)] {
                assert_eq!(options.codec, expected.codec);
                assert_eq!(options.pixel_format, expected.pixel_format);
                assert_eq!(options.extra_args, expected.extra_args);
                assert!(options.codec.starts_with("lib"), "software encoder: {}", options.codec);
                assert!(
                    options.extra_args.iter().any(|arg| arg.contains("out_color_matrix=bt709")),
                    "explicit BT.709 conversion"
                );
            }
            assert_eq!(web.crf, EMBER_WEB_CRF, "the ember web encode has its own rate factor");
            assert_eq!(hq.crf, expected_hq.crf, "the HQ encode keeps its rate factor");
        }
    }

    fn summary_with_frames(frames_emitted: usize) -> EmberSummary {
        EmberSummary {
            width: 2,
            height: 1,
            still: vec![0; 6],
            still_sha256: String::new(),
            frames_emitted,
            frames_sha256: None,
            slow_factor: EMBER_SLOW_FACTOR,
            slow_first_frame: 0,
            slow_frames_emitted: frames_emitted.saturating_sub(1) * EMBER_SLOW_FACTOR as usize
                + usize::from(frames_emitted > 0),
            slow_frames_sha256: None,
            duration: 10.0,
            valve_time: 9.75,
            hold_time: 0.25,
            fade_time: 0.25,
            tidal_reference: 1.0,
            fluid_grid: [16, 16],
            fluid_dx: 0.1,
            ink_grid: [8, 4],
            stats: ember::pipeline::EmberStats::default(),
            timings: ember::pipeline::EmberTimings::default(),
        }
    }

    #[test]
    fn test_ember_manifest_from_summary() {
        let video = EmberManifest::from_summary(&summary_with_frames(1_802), true);
        assert_eq!(
            video,
            EmberManifest {
                frames_emitted: 1_802,
                slow_frames_emitted: 18_011,
                frame_rate: constants::DEFAULT_VIDEO_FPS,
                has_video: true,
                fast_encode: true,
            }
        );
        let still = EmberManifest::from_summary(&summary_with_frames(0), false);
        assert!(!still.has_video);
        assert_eq!(still.frames_emitted, 0);
    }

    /// A package directory holding `files` (each containing its own path) and `metadata/`.
    fn package_fixture(files: &[&str]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("temp dir");
        fs::create_dir_all(dir.path().join("metadata")).expect("metadata dir");
        for file in files {
            let path = dir.path().join(file);
            fs::create_dir_all(path.parent().expect("parent dir")).expect("package dir");
            fs::write(&path, file.as_bytes()).expect("fixture file");
        }
        dir
    }

    /// Recorded steps of the manifest tests' orbit: the production default, for which `main.mp4`
    /// has 1,802 frames.
    const MANIFEST_STEPS: usize = 1_000_000;

    fn written_manifest(
        dir: &tempfile::TempDir,
        image_only: bool,
        ember: Option<&EmberManifest>,
    ) -> serde_json::Value {
        written_manifest_of(dir, MANIFEST_STEPS, image_only, ember)
    }

    fn written_manifest_of(
        dir: &tempfile::TempDir,
        steps: usize,
        image_only: bool,
        ember: Option<&EmberManifest>,
    ) -> serde_json::Value {
        let seed_dir = dir.path().to_str().expect("UTF-8 temp path");
        write_asset_manifest(seed_dir, 3456, 2234, steps, image_only, ember).expect("manifest");
        let bytes = fs::read(dir.path().join("metadata/assets.json")).expect("assets.json");
        serde_json::from_slice(&bytes).expect("valid JSON")
    }

    fn roles(manifest: &serde_json::Value) -> Vec<&str> {
        manifest["assets"]
            .as_array()
            .expect("assets array")
            .iter()
            .map(|entry| entry["role"].as_str().expect("role"))
            .collect()
    }

    fn entry<'a>(manifest: &'a serde_json::Value, role: &str) -> &'a serde_json::Value {
        manifest["assets"]
            .as_array()
            .expect("assets array")
            .iter()
            .find(|entry| entry["role"] == role)
            .unwrap_or_else(|| panic!("no {role} entry"))
    }

    /// Hex SHA-256 of a fixture file (whose content is its own path).
    fn fixture_sha256(path: &str) -> String {
        use sha2::{Digest as _, Sha256};
        hex::encode(Sha256::digest(path.as_bytes()))
    }

    const MAIN_STILL_ROLES: [&str; 3] = ["source_master", "web_full", "web_preview"];
    const MAIN_VIDEO_ROLES: [&str; 5] =
        ["main_web", "main_hq", "spectral_sweep_web", "spectral_sweep_hq", "spectral_bins"];
    const EMBER_STILL_ROLES: [&str; 3] =
        ["ember_source_master", "ember_web_full", "ember_web_preview"];
    const EMBER_VIDEO_ROLES: [&str; 3] = ["ember_web", "ember_slow_web", "ember_hq"];

    #[test]
    fn test_manifest_without_ember_keeps_the_legacy_entries() {
        let master = "images/source/master.png";
        let main_web = "videos/web/main.mp4";
        let dir = package_fixture(&[master, main_web]);
        let manifest = written_manifest(&dir, false, None);

        assert_eq!(manifest["schema_version"], 2);
        assert_eq!(roles(&manifest), [&MAIN_STILL_ROLES[..], &MAIN_VIDEO_ROLES[..]].concat());
        assert_eq!(
            *entry(&manifest, "source_master"),
            serde_json::json!({
                "path": master, "kind": "image", "role": "source_master", "format": "png",
                "width": 3456, "height": 2234, "pixel_format": "rgb48",
                "bytes": master.len(), "sha256": fixture_sha256(master),
            })
        );
        assert_eq!(
            *entry(&manifest, "web_preview"),
            serde_json::json!({
                "path": "images/web/preview.webp", "kind": "image", "role": "web_preview",
                "format": "webp", "width": 640, "height": 414,
            }),
            "missing files carry neither size nor digest"
        );
        assert_eq!(
            *entry(&manifest, "main_web"),
            serde_json::json!({
                "path": main_web, "kind": "video", "role": "main_web", "format": "mp4",
                "width": 3456, "height": 2234, "duration_seconds": 1_802.0 / 60.0,
                "frame_rate": 60,
                "codec": "h264", "pixel_format": "yuv420p",
                "bytes": main_web.len(), "sha256": fixture_sha256(main_web),
            })
        );
        assert_eq!(entry(&manifest, "main_hq")["codec"], "hevc");
        assert_eq!(entry(&manifest, "main_hq")["pixel_format"], "yuv422p10le");
        assert_eq!(entry(&manifest, "main_hq")["duration_seconds"], 1_802.0 / 60.0);
        assert_eq!(entry(&manifest, "spectral_sweep_hq")["duration_seconds"], 10.0);
        assert_eq!(
            *entry(&manifest, "spectral_bins"),
            serde_json::json!({
                "path": "spectral/", "kind": "image_set", "role": "spectral_bins",
                "format": "png", "width": 3456, "height": 2234, "pixel_format": "rgb48",
                "file_count": 64,
            })
        );
        for asset in manifest["assets"].as_array().expect("assets") {
            assert!(asset.get("color_space").is_none(), "legacy entries gain no fields");
        }
    }

    #[test]
    fn test_manifest_appends_the_ember_edition() {
        let dir = package_fixture(&[
            EMBER_STILL_PATH,
            EMBER_WEB_VIDEO_PATH,
            EMBER_SLOW_WEB_VIDEO_PATH,
            EMBER_HQ_VIDEO_PATH,
        ]);
        let ember = EmberManifest {
            frames_emitted: 1_802,
            slow_frames_emitted: 17_941,
            frame_rate: 60,
            has_video: true,
            fast_encode: false,
        };
        let manifest = written_manifest(&dir, false, Some(&ember));

        assert_eq!(manifest["schema_version"], 2, "ember entries are additive");
        let expected: Vec<&str> =
            [&MAIN_STILL_ROLES[..], &MAIN_VIDEO_ROLES[..], &EMBER_STILL_ROLES, &EMBER_VIDEO_ROLES]
                .concat();
        assert_eq!(roles(&manifest), expected);
        assert_eq!(
            *entry(&manifest, "ember_source_master"),
            serde_json::json!({
                "path": "images/source/ember.png", "kind": "image", "role": "ember_source_master",
                "format": "png", "width": 3456, "height": 2234, "pixel_format": "rgb48",
                "color_space": "srgb",
                "bytes": EMBER_STILL_PATH.len(), "sha256": fixture_sha256(EMBER_STILL_PATH),
            })
        );
        assert_eq!(
            *entry(&manifest, "ember_web_full"),
            serde_json::json!({
                "path": "images/web/ember_full.webp", "kind": "image", "role": "ember_web_full",
                "format": "webp", "width": 3456, "height": 2234, "color_space": "srgb",
            })
        );
        let preview = entry(&manifest, "ember_web_preview");
        assert_eq!(preview["path"], "images/web/ember_preview.webp");
        assert_eq!((&preview["width"], &preview["height"]), (&640.into(), &414.into()));
        assert_eq!(
            *entry(&manifest, "ember_web"),
            serde_json::json!({
                "path": "videos/web/ember.mp4", "kind": "video", "role": "ember_web",
                "format": "mp4", "width": 3456, "height": 2234,
                "duration_seconds": 1_802.0 / 60.0, "frame_rate": 60,
                "codec": "h264", "pixel_format": "yuv420p", "color_space": "srgb",
                "bytes": EMBER_WEB_VIDEO_PATH.len(),
                "sha256": fixture_sha256(EMBER_WEB_VIDEO_PATH),
            })
        );
        // The slow film: the web encode's settings, and its own, longer duration.
        assert_eq!(
            *entry(&manifest, "ember_slow_web"),
            serde_json::json!({
                "path": "videos/web/ember_slow.mp4", "kind": "video", "role": "ember_slow_web",
                "format": "mp4", "width": 3456, "height": 2234,
                "duration_seconds": 17_941.0 / 60.0, "frame_rate": 60,
                "codec": "h264", "pixel_format": "yuv420p", "color_space": "srgb",
                "bytes": EMBER_SLOW_WEB_VIDEO_PATH.len(),
                "sha256": fixture_sha256(EMBER_SLOW_WEB_VIDEO_PATH),
            })
        );
        let hq = entry(&manifest, "ember_hq");
        assert_eq!(hq["path"], "videos/hq/ember.mp4");
        assert_eq!((&hq["codec"], &hq["pixel_format"]), (&"hevc".into(), &"yuv422p10le".into()));
        assert_eq!(hq["duration_seconds"], 1_802.0 / 60.0);
        assert_eq!(hq["sha256"], fixture_sha256(EMBER_HQ_VIDEO_PATH).as_str());
    }

    #[test]
    fn test_manifest_main_video_duration_counts_the_encoded_frames() {
        let dir = package_fixture(&[]);
        // 100,000 steps: every 55th step up to 99,990, then the final step 99,999.
        assert_eq!(ember_frame_schedule(100_000).len(), 1_819);
        for steps in [100_000, 1_000_000, 1_234_567] {
            // The ember videos are frame-locked to `main.mp4`: same frames, same duration.
            let frames = ember_frame_schedule(steps).len();
            let slow_frames = (frames - 1) * EMBER_SLOW_FACTOR as usize + 1;
            let ember = EmberManifest {
                frames_emitted: frames,
                slow_frames_emitted: slow_frames,
                frame_rate: constants::DEFAULT_VIDEO_FPS,
                has_video: true,
                fast_encode: false,
            };
            let manifest = written_manifest_of(&dir, steps, false, Some(&ember));
            let duration = |role| entry(&manifest, role)["duration_seconds"].clone();
            // Written from the same f64, the four entries carry the same text.
            for role in ["main_hq", "ember_web", "ember_hq"] {
                assert_eq!(duration(role), duration("main_web"), "{role} at {steps} steps");
            }
            // `serde_json` parses floats exactly (`float_roundtrip`), so the value is the one
            // written, bit for bit.
            let seconds = frames as f64 / f64::from(constants::DEFAULT_VIDEO_FPS);
            let main = duration("main_web").as_f64().expect("duration");
            assert_eq!(main.to_bits(), seconds.to_bits(), "{main} s for {frames} frames");
            assert_eq!(duration("spectral_sweep_web"), 10.0, "the sweep is unchanged");
            // The slow film has its own length.
            let slow = slow_frames as f64 / f64::from(constants::DEFAULT_VIDEO_FPS);
            assert_eq!(duration("ember_slow_web").as_f64().map(f64::to_bits), Some(slow.to_bits()));
        }
    }

    #[test]
    fn test_manifest_records_the_fast_ember_encode() {
        let dir = package_fixture(&[]);
        let ember = EmberManifest {
            frames_emitted: 120,
            slow_frames_emitted: 1_191,
            frame_rate: 60,
            has_video: true,
            fast_encode: true,
        };
        let manifest = written_manifest(&dir, false, Some(&ember));
        let hq = entry(&manifest, "ember_hq");
        assert_eq!((&hq["codec"], &hq["pixel_format"]), (&"h264".into(), &"yuv420p10le".into()));
        assert_eq!(hq["duration_seconds"], 2.0);
        // The fast encode replaces the HQ slot only; the slow film keeps the web settings.
        let slow = entry(&manifest, "ember_slow_web");
        assert_eq!((&slow["codec"], &slow["pixel_format"]), (&"h264".into(), &"yuv420p".into()));
    }

    #[test]
    fn test_manifest_image_only_lists_only_stills() {
        let dir = package_fixture(&[]);
        assert_eq!(roles(&written_manifest(&dir, true, None)), MAIN_STILL_ROLES);

        let still_only = EmberManifest {
            frames_emitted: 0,
            slow_frames_emitted: 0,
            frame_rate: 60,
            has_video: false,
            fast_encode: false,
        };
        assert_eq!(
            roles(&written_manifest(&dir, true, Some(&still_only))),
            [MAIN_STILL_ROLES, EMBER_STILL_ROLES].concat()
        );
    }

    #[test]
    fn test_ember_package_paths_are_the_required_package_files() {
        // `run.py` requires exactly these files as EMBER_PACKAGE_FILES (and removes them from a
        // package whose ember edition failed); keep the two lists in sync.
        let run_py = include_str!("../run.py");
        let listed = run_py
            .split_once("\nEMBER_PACKAGE_FILES = (\n")
            .and_then(|(_, rest)| rest.split_once("\n)"))
            .map(|(list, _)| list)
            .expect("run.py defines EMBER_PACKAGE_FILES");
        let listed: Vec<&str> = listed
            .lines()
            .map(|line| line.trim().trim_end_matches(',').trim_matches('"'))
            .collect();
        assert_eq!(listed, EMBER_OUTPUT_PATHS, "run.py must require exactly the ember outputs");
    }

    #[test]
    fn test_remove_ember_outputs_keeps_the_rest_of_the_package() {
        let master = "images/source/master.png";
        let traits = "metadata/nft_traits.json";
        let partial = [EMBER_STILL_PATH, EMBER_HQ_VIDEO_PATH, EMBER_CERTIFICATE_PATH];
        let dir = package_fixture(&[&[master, traits][..], &partial[..]].concat());
        let seed_dir = dir.path().to_str().expect("UTF-8 temp path");

        remove_ember_outputs(seed_dir).expect("partial outputs removed");
        for path in EMBER_OUTPUT_PATHS {
            assert!(!dir.path().join(path).exists(), "{path} must be gone");
        }
        assert!(dir.path().join(master).is_file() && dir.path().join(traits).is_file());
        remove_ember_outputs(seed_dir).expect("nothing left to remove is fine");

        // A path that cannot be removed as a file is reported, after the others are removed.
        fs::create_dir_all(dir.path().join(EMBER_STILL_PATH)).expect("blocking directory");
        fs::write(dir.path().join(EMBER_CERTIFICATE_PATH), b"{").expect("truncated certificate");
        match remove_ember_outputs(seed_dir) {
            Err(AppError::Config(ConfigError::FileSystem { operation, path, .. })) => {
                assert_eq!(operation, "remove");
                assert!(path.ends_with(EMBER_STILL_PATH), "{path}");
            }
            other => panic!("expected the removal error, got {other:?}"),
        }
        assert!(!dir.path().join(EMBER_CERTIFICATE_PATH).exists(), "the others are still removed");
    }

    /// The coarse but complete configuration of the golden test (`tests/ember_determinism.rs`),
    /// with a short pre-roll and valve lead, so that the brief arc of [`TINY_STEPS`] is long
    /// enough to ink: a 96×64 render of it takes seconds.
    fn tiny_ember_config() -> EmberConfig {
        let mut config = EmberConfig::default();
        config.fluid.rows = 64;
        config.fluid.body_radius = 0.08;
        config.fluid.mask_width = 0.02;
        config.fluid.max_dt = 0.01;
        config.fluid.max_snapshot_interval = 0.01;
        config.contact.vorticity_gate = 1.0;
        config.contact.soak_depth = 0.12;
        config.contact.pre_roll = 0.3;
        config.contact.valve_lead = 0.1;
        config.look.hold_fraction = 0.25;
        config.look.fade_fraction = 0.25;
        config.paper.formation_modes = 64;
        config
    }

    /// The golden test's orbit: [`figure_eight_bodies`] slightly tilted out of its plane.
    fn tilted_figure_eight_bodies() -> Vec<Body> {
        let mut bodies = figure_eight_bodies();
        bodies[0].position.z = 0.05;
        bodies[1].position.z = -0.05;
        bodies
    }

    /// Whether every one of `tools` (`ffmpeg`, `ffprobe`) runs here. When one does not, the
    /// calling test `test` should return early: locally it is skipped with a message; under CI
    /// (the `CI` environment variable is set) this panics instead, so that a runner without the
    /// tools fails rather than passes without testing.
    fn media_tools_available(test: &str, tools: &[&str]) -> bool {
        let missing: Vec<&str> = tools
            .iter()
            .copied()
            .filter(|tool| {
                !Command::new(tool)
                    .arg("-version")
                    .output()
                    .is_ok_and(|output| output.status.success())
            })
            .collect();
        if missing.is_empty() {
            return true;
        }
        assert!(
            std::env::var_os("CI").is_none(),
            "{test} needs {missing:?} on PATH, and CI is set: install FFmpeg in this job"
        );
        eprintln!("skipping {test}: {missing:?} not on PATH");
        false
    }

    /// Recorded steps of the tiny ember renders: a short arc of [`tilted_figure_eight_bodies`],
    /// which its frontal view spreads over the 96×64 sheet. There is a frame per step, and the
    /// fluid lands on [`EMBER_SLOW_FACTOR`] snapshots per frame in every mode, so the step count
    /// sets the tests' cost: a few seconds under [`tiny_ember_config`], whose short pre-roll
    /// lets the arc (0.75 fluid time units) ink for half of its length.
    const TINY_STEPS: usize = 240;

    /// A package directory with the ember edition's output paths, each built from its named
    /// constant.
    struct TinyEmberPackage {
        dir: tempfile::TempDir,
        still_png: String,
        full_webp: String,
        preview_webp: String,
        web_video: String,
        slow_web_video: String,
        hq_video: String,
        certificate: String,
    }

    impl TinyEmberPackage {
        fn new() -> Self {
            let dir = package_fixture(&[]);
            for subdir in ["images/source", "images/web", "videos/web", "videos/hq"] {
                fs::create_dir_all(dir.path().join(subdir)).expect("package directory");
            }
            let path = |relative: &str| {
                dir.path().join(relative).to_str().expect("UTF-8 temp path").to_string()
            };
            Self {
                still_png: path(EMBER_STILL_PATH),
                full_webp: path(EMBER_FULL_WEBP_PATH),
                preview_webp: path(EMBER_PREVIEW_WEBP_PATH),
                web_video: path(EMBER_WEB_VIDEO_PATH),
                slow_web_video: path(EMBER_SLOW_WEB_VIDEO_PATH),
                hq_video: path(EMBER_HQ_VIDEO_PATH),
                certificate: path(EMBER_CERTIFICATE_PATH),
                dir,
            }
        }

        fn seed_dir(&self) -> &str {
            self.dir.path().to_str().expect("UTF-8 temp path")
        }

        /// Renders the ember edition of [`tilted_figure_eight_bodies`] under
        /// [`tiny_ember_config`] at 96×64 into this package, in a small pool (the tiny grids gain
        /// nothing from more threads).
        fn render(&self, steps: usize, image_only: bool, fast_encode: bool) -> EmberSummary {
            let config = tiny_ember_config();
            let bodies = tilted_figure_eight_bodies();
            let request = EmberEditionRequest {
                seed_hex: "46205528",
                seed_bytes: &[0x46, 0x20, 0x55, 0x28],
                bodies: &bodies,
                view: &frontal_view(&bodies, steps, 96, 64),
                steps,
                width: 96,
                height: 64,
                image_only,
                fast_encode,
                config: &config,
                paths: EmberOutputPaths {
                    still_png: &self.still_png,
                    full_webp: &self.full_webp,
                    preview_webp: &self.preview_webp,
                    web_video: &self.web_video,
                    slow_web_video: &self.slow_web_video,
                    hq_video: &self.hq_video,
                    certificate: &self.certificate,
                },
            };
            rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .expect("thread pool")
                .install(|| render_ember_edition(&request))
                .expect("the tiny ember edition renders")
        }

        /// The written certificate as JSON.
        fn certificate(&self) -> serde_json::Value {
            serde_json::from_slice(&fs::read(&self.certificate).expect("ember.json")).expect("JSON")
        }

        /// Asserts that `ember.png` decodes to the still the certificate and `summary` record.
        fn assert_png_is_the_certified_still(&self, summary: &EmberSummary) {
            use sha2::{Digest as _, Sha256};
            let certificate = self.certificate();
            let certified =
                certificate["outputs"]["still_rgb48le_sha256"].as_str().expect("still digest");
            let (width, height, rgb48le) = png_as_rgb48le(std::path::Path::new(&self.still_png));
            assert_eq!((width, height), (96, 64));
            assert_eq!(hex::encode(Sha256::digest(&rgb48le)), certified, "ember.png is the still");
            assert_eq!(certified, summary.still_sha256, "the certificate records the render");
            for webp in [&self.full_webp, &self.preview_webp] {
                assert!(fs::metadata(webp).expect("WebP").len() > 0, "{webp}");
            }
        }
    }

    /// `(codec name, frames counted by decoding)` of the first video stream of `path`
    /// (`ffprobe -count_frames`).
    fn probe_video(path: &str) -> (String, usize) {
        let output = Command::new("ffprobe")
            .args(["-v", "error", "-select_streams", "v:0", "-count_frames"])
            .args(["-show_entries", "stream=codec_name,nb_read_frames"])
            .args(["-of", "default=noprint_wrappers=1", path])
            .output()
            .expect("ffprobe runs");
        assert!(output.status.success(), "ffprobe {path}: {output:?}");
        let text = String::from_utf8(output.stdout).expect("UTF-8 ffprobe output");
        let field = |name: &str| {
            text.lines()
                .find_map(|line| line.strip_prefix(name)?.strip_prefix('='))
                .unwrap_or_else(|| panic!("ffprobe reports no {name} for {path}:\n{text}"))
                .to_string()
        };
        (field("codec_name"), field("nb_read_frames").parse().expect("a frame count"))
    }

    /// The PNG at `path` decoded with the `png` crate, as `(width, height, rgb48le bytes)`.
    fn png_as_rgb48le(path: &std::path::Path) -> (u32, u32, Vec<u8>) {
        let file = std::io::BufReader::new(File::open(path).expect("PNG file"));
        let mut reader = png::Decoder::new(file).read_info().expect("PNG header");
        let mut buffer = vec![0; reader.output_buffer_size().expect("PNG buffer size")];
        let info = reader.next_frame(&mut buffer).expect("PNG pixels");
        assert_eq!(
            (info.color_type, info.bit_depth),
            (png::ColorType::Rgb, png::BitDepth::Sixteen),
            "the ember still is 16-bit RGB"
        );
        // PNG stores 16-bit samples big-endian.
        let rgb48le = buffer[..info.buffer_size()]
            .chunks_exact(2)
            .flat_map(|sample| [sample[1], sample[0]])
            .collect();
        (info.width, info.height, rgb48le)
    }

    #[test]
    fn test_render_ember_edition_writes_a_certified_still() {
        use sha2::{Digest as _, Sha256};

        // The still's WebP derivatives need FFmpeg.
        if !media_tools_available("test_render_ember_edition_writes_a_certified_still", &["ffmpeg"])
        {
            return;
        }
        let package = TinyEmberPackage::new();
        let summary = package.render(TINY_STEPS, true, false);
        package.assert_png_is_the_certified_still(&summary);

        // Still only: no frame stream, no videos; the statistics count the still alone.
        let certificate = package.certificate();
        let outputs = &certificate["outputs"];
        assert!(outputs["frames_rgb48le_sha256"].is_null(), "{outputs}");
        assert_eq!((outputs["frames_emitted"].as_u64(), summary.frames_emitted), (Some(0), 0));
        assert!(outputs["slow_frames_rgb48le_sha256"].is_null());
        assert_eq!(
            (outputs["slow_frames_emitted"].as_u64(), summary.slow_frames_emitted),
            (Some(0), 0)
        );
        for video in [&package.web_video, &package.slow_web_video, &package.hq_video] {
            assert!(!std::path::Path::new(video).exists(), "{video}");
        }
        let stats = summary.stats;
        assert!(stats.still_ink_fraction > 0.0 && stats.still_ink_fraction < 1.0, "{stats:?}");
        assert_eq!(
            certificate["stats"]["still_ink_fraction"].as_f64().map(f64::to_bits),
            Some(stats.still_ink_fraction.to_bits())
        );
        assert_eq!(certificate["stats"]["contact_events"], stats.contact_events);
        assert_eq!(certificate["algorithm"], ember::certificate::ALGORITHM_VERSION);
        // The certificate records the request: the configuration under test, the orbit, the
        // schedule of `main.mp4` and the package seed's own paper.
        let inputs = &certificate["inputs"];
        assert_eq!(certificate["config"]["look"]["fade_fraction"], 0.25);
        assert_eq!(certificate["config"]["fluid"]["rows"], 64);
        assert_eq!((&inputs["seed"], &inputs["steps"]), (&"46205528".into(), &TINY_STEPS.into()));
        assert_eq!(inputs["frames"]["count"], ember_frame_schedule(TINY_STEPS).len());
        assert_eq!(
            inputs["paper_seed_sha256"],
            hex::encode(Sha256::digest(ember_paper_seed(&[0x46, 0x20, 0x55, 0x28])))
        );
        assert_eq!(inputs["bodies"][0]["position"][2], 0.05);
    }

    /// The production path end to end, at a tiny size: every frame is shaded and streamed into
    /// the encoders of its film (`--fast-encode`: software H.264 in every slot), each video holds
    /// exactly the frames of its film (the scheduled frames, or the slow film's), the certificate
    /// records both frame streams the render hashed, and the asset manifest lists the three
    /// videos, each with the duration of its frame count.
    #[test]
    fn test_render_ember_edition_writes_its_videos() {
        let test = "test_render_ember_edition_writes_its_videos";
        if !media_tools_available(test, &["ffmpeg", "ffprobe"]) {
            return;
        }
        // The preflight plans what the render then does.
        let config = tiny_ember_config();
        let bodies = tilted_figure_eight_bodies();
        let view = frontal_view(&bodies, TINY_STEPS, 96, 64);
        let plan = preflight_ember_edition(&bodies, &view, TINY_STEPS, 96, 64, &config)
            .expect("the tiny orbit plans");

        let package = TinyEmberPackage::new();
        let summary = package.render(TINY_STEPS, false, true);
        package.assert_png_is_the_certified_still(&summary);

        let frames = ember_frame_schedule(TINY_STEPS).len();
        assert_eq!(summary.frames_emitted, frames);
        let frames_sha256 = summary.frames_sha256.as_deref().expect("a video render hashes frames");
        let certificate = package.certificate();
        let outputs = &certificate["outputs"];
        assert_eq!(outputs["frames_rgb48le_sha256"], frames_sha256, "{outputs}");
        assert_eq!(outputs["frames_emitted"], frames);
        // The slow film: every scheduled frame from its first one on, and the frames between.
        let slow_frames = summary.slow_frames_emitted;
        assert_eq!(plan.slow_frames, slow_frames);
        assert_eq!(
            slow_frames,
            (frames - 1 - summary.slow_first_frame) * EMBER_SLOW_FACTOR as usize + 1
        );
        let slow_sha256 = summary.slow_frames_sha256.as_deref().expect("the slow film is hashed");
        assert_eq!(outputs["slow_frames_rgb48le_sha256"], slow_sha256);
        assert_eq!(outputs["slow_frames_emitted"], slow_frames);
        assert_eq!(certificate["inputs"]["frames"]["slow_factor"], EMBER_SLOW_FACTOR);
        assert_eq!(certificate["derived"]["slow_first_frame"], summary.slow_first_frame);
        let stats = summary.stats;
        assert!(stats.contact_events > 0, "the bodies inked the water: {stats:?}");
        assert_eq!(certificate["stats"]["contact_events"], stats.contact_events);

        // Every video exists and decodes to exactly the frames of its film.
        let [web, hq] = ember_video_options(true);
        let videos = [
            ("ember_web", &package.web_video, &web, frames),
            ("ember_slow_web", &package.slow_web_video, &web, slow_frames),
            ("ember_hq", &package.hq_video, &hq, frames),
        ];
        for (_, video, options, film_frames) in videos {
            assert!(fs::metadata(video).expect("an ember video").len() > 0, "{video}");
            let (codec, decoded) = probe_video(video);
            assert_eq!(codec, manifest_codec(&options.codec), "{video}");
            assert_eq!(decoded, film_frames, "{video}: every frame of the film is encoded");
        }

        // The manifest lists each with the duration of its frame count at the product rate.
        let manifest = EmberManifest::from_summary(&summary, true);
        write_asset_manifest(package.seed_dir(), 96, 64, TINY_STEPS, false, Some(&manifest))
            .expect("manifest");
        let bytes = fs::read(package.dir.path().join("metadata/assets.json")).expect("assets.json");
        let assets: serde_json::Value = serde_json::from_slice(&bytes).expect("JSON");
        for (role, video, _, film_frames) in videos {
            let seconds = film_frames as f64 / f64::from(constants::DEFAULT_VIDEO_FPS);
            let listed = entry(&assets, role);
            assert_eq!(
                listed["duration_seconds"].as_f64().map(f64::to_bits),
                Some(seconds.to_bits()),
                "{listed}"
            );
            assert_eq!(listed["frame_rate"], constants::DEFAULT_VIDEO_FPS, "{listed}");
            assert_eq!(listed["bytes"], fs::metadata(video).expect("video").len(), "{listed}");
        }
        assert_eq!(
            entry(&assets, "main_web")["duration_seconds"],
            entry(&assets, "ember_web")["duration_seconds"],
            "the ember videos are frame-locked to main.mp4"
        );
    }
}
