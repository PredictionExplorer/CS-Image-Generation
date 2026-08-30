//! Video encoding functionality
//!
//! Provides website-compatible H.264 plus high-quality H.265 encoding.

use std::error::Error;
use std::io::Write;
use std::process::{Child, ChildStdin, Command, Stdio};
use tracing::info;

use crate::render::error::{RenderError, Result};

/// Configuration for video encoding
///
/// This struct provides fine-grained control over `FFmpeg` encoding parameters.
/// Use [`VideoEncodingOptions::web_compatible`] for the public website MP4 and
/// [`VideoEncodingOptions::high_quality`] for the archival HEVC copy.
#[derive(Debug, Clone)]
pub struct VideoEncodingOptions {
    /// Output bitrate (only used for hardware encoders or 2-pass encoding)
    /// Leave empty for CRF mode (quality-based variable bitrate)
    pub bitrate: String,

    /// H.264/H.265 preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    /// Slower presets provide better compression at the cost of encoding time
    /// Note: Not used by hardware encoders
    pub preset: String,

    /// Constant Rate Factor (0-51, lower = better quality)
    /// For H.265: CRF 19-20 is visually lossless, 23 is default, 28 is acceptable for web
    /// For H.264: CRF 18 is visually lossless
    pub crf: u32,

    /// Pixel format for color subsampling and bit depth
    /// yuv420p = 8-bit 4:2:0 (most compatible)
    /// yuv422p10le = 10-bit 4:2:2 (high quality, better gradients)
    /// yuv420p10le = 10-bit 4:2:0 (good quality, smaller files)
    pub pixel_format: String,

    /// Video codec to use (e.g., "libx264", "libx265", "`h264_videotoolbox`")
    pub codec: String,

    /// Input pixel format from source frames (rgb24 for 8-bit, rgb48le for 16-bit)
    pub input_pixel_format: String,

    /// Additional `FFmpeg` arguments for advanced customization
    /// These are passed directly to `FFmpeg` after all other options
    pub extra_args: Vec<String>,
}

/// One encoded output produced from the same raw frame stream.
#[derive(Debug, Clone)]
pub struct VideoOutputSpec {
    /// Path to the MP4 file to create.
    pub output_file: String,
    /// `FFmpeg` options used for this output.
    pub options: VideoEncodingOptions,
}

impl Default for VideoEncodingOptions {
    /// Default to the high-quality archival encoder.
    fn default() -> Self {
        Self::high_quality()
    }
}

impl VideoEncodingOptions {
    /// Website-compatible H.264 MP4 for `QuickTime` and broad browser playback.
    #[must_use]
    pub fn web_compatible() -> Self {
        Self {
            codec: "libx264".to_string(),
            preset: "medium".to_string(),
            crf: 18,
            bitrate: String::new(),
            pixel_format: "yuv420p".to_string(),
            input_pixel_format: "rgb48le".to_string(),
            extra_args: vec![
                "-movflags".to_string(),
                "+faststart".to_string(),
                "-colorspace".to_string(),
                "bt709".to_string(),
                "-color_primaries".to_string(),
                "bt709".to_string(),
                "-color_trc".to_string(),
                "bt709".to_string(),
                "-color_range".to_string(),
                "tv".to_string(),
            ],
        }
    }

    /// High-quality HEVC copy preserving the existing 10-bit 4:2:2 encode path.
    #[must_use]
    pub fn high_quality() -> Self {
        Self {
            codec: "libx265".to_string(),
            preset: "slower".to_string(),
            crf: 17,
            bitrate: String::new(),
            pixel_format: "yuv422p10le".to_string(),
            input_pixel_format: "rgb48le".to_string(),
            extra_args: vec![
                "-profile:v".to_string(),
                "main422-10".to_string(),
                "-x265-params".to_string(),
                "bframes=8:ref=6:\
                 rc-lookahead=250:\
                 aq-mode=3:aq-strength=1.0:\
                 psy-rd=2.5:psy-rdoq=1.5:\
                 deblock=-1,-1:\
                 no-sao=0:\
                 colorprim=smpte432:transfer=iec61966-2-1:colormatrix=bt709:\
                 qg-size=8:\
                 rdoq-level=2"
                    .to_string(),
                // Content tuning for gradients and smooth motion
                "-tune".to_string(),
                "grain".to_string(),
                // Web optimization (instant playback while streaming)
                "-movflags".to_string(),
                "+faststart".to_string(),
                // Color accuracy metadata (critical for correct reproduction)
                "-colorspace".to_string(),
                "bt709".to_string(),
                "-color_primaries".to_string(),
                "smpte432".to_string(),
                "-color_trc".to_string(),
                "iec61966-2-1".to_string(),
                "-color_range".to_string(),
                "tv".to_string(),
                "-tag:v".to_string(),
                "hvc1".to_string(),
            ],
        }
    }

    /// Fast encoding mode using hardware acceleration (macOS `VideoToolbox`)
    ///
    /// This configuration prioritizes encoding speed over maximum quality:
    /// - Hardware HEVC encoder: 3-5× faster than software
    /// - 10-bit color: Maintains gradient quality
    /// - 4:2:0 chroma: Standard compatibility
    /// - Quality ~65/100: Roughly equivalent to CRF 20-22
    /// - Fast encoding: Suitable for draft renders or batch generation
    ///
    /// Trade-offs:
    /// - Encoding time: 3-5× faster than default
    /// - File size: ~15-25% larger than software encoder at same quality
    /// - Quality: Very good (5-10% worse than software at peak quality)
    ///
    /// Best for: Preview renders, iteration, batch generation
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn fast_encode() -> Self {
        Self {
            codec: "hevc_videotoolbox".to_string(),
            preset: String::new(),  // Not used by hardware encoder
            crf: 0,                 // Not used by hardware encoder
            bitrate: String::new(), // VBR mode with -q:v
            pixel_format: "yuv420p10le".to_string(),
            input_pixel_format: "rgb48le".to_string(),
            extra_args: vec![
                // Hardware encoder quality (0-100 scale, 65 ≈ CRF 20-22)
                "-q:v".to_string(),
                "60".to_string(),
                // Allow B-frames for better compression
                "-allow_sw".to_string(),
                "1".to_string(),
                // Web optimization
                "-movflags".to_string(),
                "+faststart".to_string(),
                // Color metadata
                "-colorspace".to_string(),
                "bt709".to_string(),
                "-color_primaries".to_string(),
                "smpte432".to_string(),
                "-color_trc".to_string(),
                "iec61966-2-1".to_string(),
                "-color_range".to_string(),
                "tv".to_string(),
                // Compatibility tag
                "-tag:v".to_string(),
                "hvc1".to_string(),
            ],
        }
    }

    /// Fast encoding fallback for non-macOS platforms (H.264 with faster settings)
    #[cfg(not(target_os = "macos"))]
    #[must_use]
    pub fn fast_encode() -> Self {
        Self {
            codec: "libx264".to_string(),
            preset: "fast".to_string(),
            crf: 21,
            bitrate: String::new(),
            pixel_format: "yuv420p10le".to_string(),
            input_pixel_format: "rgb48le".to_string(),
            extra_args: vec![
                "-tune".to_string(),
                "film".to_string(),
                "-movflags".to_string(),
                "+faststart".to_string(),
                "-colorspace".to_string(),
                "bt709".to_string(),
                "-color_primaries".to_string(),
                "smpte432".to_string(),
                "-color_trc".to_string(),
                "iec61966-2-1".to_string(),
                "-color_range".to_string(),
                "tv".to_string(),
            ],
        }
    }
}

struct MultiVideoWriter {
    writers: Vec<ChildStdin>,
}

impl Write for MultiVideoWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        for writer in &mut self.writers {
            writer.write_all(buf)?;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        for writer in &mut self.writers {
            writer.flush()?;
        }
        Ok(())
    }
}

fn build_ffmpeg_command(
    width: u32,
    height: u32,
    frame_rate: u32,
    spec: &VideoOutputSpec,
) -> Command {
    let options = &spec.options;
    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        &options.input_pixel_format,
        "-s",
        &format!("{width}x{height}"),
        "-r",
        &frame_rate.to_string(),
        "-i",
        "-",
    ]);

    cmd.args(["-c:v", &options.codec]);
    if !options.preset.is_empty() && options.codec.starts_with("lib") {
        cmd.args(["-preset", &options.preset]);
    }
    if options.codec.starts_with("lib") && options.crf > 0 {
        cmd.args(["-crf", &options.crf.to_string()]);
    }
    if !options.bitrate.is_empty() {
        cmd.args(["-b:v", &options.bitrate]);
    }
    cmd.args(["-pix_fmt", &options.pixel_format]);
    for arg in &options.extra_args {
        cmd.arg(arg);
    }
    cmd.arg(&spec.output_file);
    cmd
}

fn spawn_encoder(
    width: u32,
    height: u32,
    frame_rate: u32,
    spec: &VideoOutputSpec,
) -> Result<(String, Child, ChildStdin)> {
    info!(
        "Encoding {} with codec: {}, pixel format: {}",
        spec.output_file, spec.options.codec, spec.options.pixel_format
    );

    let mut child = build_ffmpeg_command(width, height, frame_rate, spec)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(RenderError::VideoEncoding)?;
    let stdin = child.stdin.take().ok_or_else(|| {
        RenderError::VideoEncoding(std::io::Error::other("failed to open ffmpeg stdin"))
    })?;
    Ok((spec.output_file.clone(), child, stdin))
}

fn validate_video_params(
    width: u32,
    height: u32,
    frame_rate: u32,
    output_count: usize,
) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(RenderError::InvalidDimensions { width, height });
    }

    if frame_rate == 0 {
        return Err(RenderError::InvalidConfig {
            parameter: "frame_rate".into(),
            reason: "must be greater than 0".into(),
        });
    }

    if output_count == 0 {
        return Err(RenderError::InvalidConfig {
            parameter: "outputs".into(),
            reason: "must include at least one video output".into(),
        });
    }

    Ok(())
}

fn wait_for_encoders(mut children: Vec<(String, Child)>) -> Result<()> {
    for (output_file, child) in children.drain(..) {
        let output = child.wait_with_output().map_err(RenderError::VideoEncoding)?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(RenderError::VideoEncoding(std::io::Error::other(format!(
                "FFmpeg failed for {output_file} with status {:?}. stderr: {}",
                output.status, stderr
            ))));
        }
        info!("   Saved video => {}", output_file);
    }
    Ok(())
}

fn kill_encoders(children: &mut [(String, Child)]) {
    for (_, child) in children {
        let _ = child.kill();
    }
}

/// Create video in a single pass using `FFmpeg` with configurable options
///
/// This function pipes raw RGB frames directly to `FFmpeg`'s stdin, avoiding the need
/// for temporary frame files on disk. Supports both 8-bit (rgb24) and 16-bit (rgb48le)
/// input formats, automatically determined from the `VideoEncodingOptions`.
///
/// # Arguments
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels  
/// * `frame_rate` - Output video framerate (fps)
/// * `frames_iter` - Closure that writes raw RGB frame data to the provided writer
/// * `output_file` - Path to the output video file
/// * `options` - Encoding configuration options
///
/// # Returns
/// * `Ok(())` on success
/// * `Err(RenderError)` if `FFmpeg` fails or encoding parameters are invalid
///
/// # Example
///
/// ```no_run
/// # use three_body_problem::render::video::{VideoEncodingOptions, create_video_from_frames_singlepass};
/// # use std::io::Write;
/// let options = VideoEncodingOptions::default();
/// create_video_from_frames_singlepass(
///     1920, 1080, 60,
///     |writer| { writer.write_all(&[0u8; 3])?; Ok(()) },
///     "output.mp4",
///     &options,
/// ).expect("encoding should succeed");
/// ```
pub fn create_video_from_frames_singlepass(
    width: u32,
    height: u32,
    frame_rate: u32,
    frames_iter: impl FnMut(&mut dyn Write) -> std::result::Result<(), Box<dyn Error>>,
    output_file: &str,
    options: &VideoEncodingOptions,
) -> Result<()> {
    create_videos_from_frames_singlepass(
        width,
        height,
        frame_rate,
        frames_iter,
        &[VideoOutputSpec { output_file: output_file.to_string(), options: options.clone() }],
    )
}

/// Create one or more videos from a single raw frame stream.
pub fn create_videos_from_frames_singlepass(
    width: u32,
    height: u32,
    frame_rate: u32,
    mut frames_iter: impl FnMut(&mut dyn Write) -> std::result::Result<(), Box<dyn Error>>,
    outputs: &[VideoOutputSpec],
) -> Result<()> {
    validate_video_params(width, height, frame_rate, outputs.len())?;

    let mut children = Vec::with_capacity(outputs.len());
    let mut writers = Vec::with_capacity(outputs.len());
    for spec in outputs {
        match spawn_encoder(width, height, frame_rate, spec) {
            Ok((output_file, child, stdin)) => {
                children.push((output_file, child));
                writers.push(stdin);
            }
            Err(e) => {
                kill_encoders(&mut children);
                return Err(e);
            }
        }
    }

    let mut writer = MultiVideoWriter { writers };
    if let Err(e) = frames_iter(&mut writer) {
        let _ = writer.flush();
        drop(writer);
        kill_encoders(&mut children);
        return Err(RenderError::VideoEncoding(std::io::Error::other(e.to_string())));
    }
    let _ = writer.flush();
    drop(writer);

    wait_for_encoders(children)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_options() {
        let options = VideoEncodingOptions::default();
        assert_eq!(options.codec, "libx265");
        assert_eq!(options.preset, "slower");
        assert_eq!(options.crf, 17);
        assert_eq!(options.pixel_format, "yuv422p10le");
        assert_eq!(options.input_pixel_format, "rgb48le");
        assert!(options.bitrate.is_empty());
        assert!(options.extra_args.contains(&"-tune".to_string()));
        assert!(options.extra_args.contains(&"grain".to_string()));
        assert!(options.extra_args.contains(&"-movflags".to_string()));
        assert!(options.extra_args.contains(&"+faststart".to_string()));
        assert!(options.extra_args.contains(&"-colorspace".to_string()));
        assert!(options.extra_args.contains(&"bt709".to_string()));
    }

    #[test]
    fn test_web_compatible_options() {
        let options = VideoEncodingOptions::web_compatible();
        assert_eq!(options.codec, "libx264");
        assert_eq!(options.pixel_format, "yuv420p");
        assert_eq!(options.input_pixel_format, "rgb48le");
        assert!(options.extra_args.contains(&"-movflags".to_string()));
        assert!(options.extra_args.contains(&"+faststart".to_string()));
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_fast_encode_macos() {
        let fast = VideoEncodingOptions::fast_encode();
        assert_eq!(fast.codec, "hevc_videotoolbox");
        assert!(fast.preset.is_empty()); // Hardware encoder doesn't use preset
        assert_eq!(fast.pixel_format, "yuv420p10le");
        assert_eq!(fast.input_pixel_format, "rgb48le");
        assert!(fast.extra_args.contains(&"-q:v".to_string()));
        assert!(fast.extra_args.contains(&"60".to_string()));
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_fast_encode_other_platforms() {
        let fast = VideoEncodingOptions::fast_encode();
        assert_eq!(fast.codec, "libx264");
        assert_eq!(fast.preset, "fast");
        assert_eq!(fast.crf, 21);
        assert_eq!(fast.pixel_format, "yuv420p10le");
        assert_eq!(fast.input_pixel_format, "rgb48le");
    }

    #[test]
    fn test_color_metadata_present() {
        let options = VideoEncodingOptions::default();
        // Verify color metadata is set for accurate reproduction
        let args = &options.extra_args;
        let colorspace_idx =
            args.iter().position(|s| s == "-colorspace").expect("expected -colorspace arg");
        assert_eq!(args[colorspace_idx + 1], "bt709");

        let primaries_idx = args
            .iter()
            .position(|s| s == "-color_primaries")
            .expect("expected -color_primaries arg");
        assert_eq!(args[primaries_idx + 1], "smpte432");

        let trc_idx = args.iter().position(|s| s == "-color_trc").expect("expected -color_trc arg");
        assert_eq!(args[trc_idx + 1], "iec61966-2-1");
    }

    #[test]
    fn test_perceptual_optimization() {
        let options = VideoEncodingOptions::default();
        // Verify x265-params contains perceptual optimizations
        let x265_params = options
            .extra_args
            .iter()
            .position(|s| s == "-x265-params")
            .map(|idx| &options.extra_args[idx + 1]);

        assert!(x265_params.is_some());
        let params = x265_params.expect("expected -x265-params arg");
        assert!(params.contains("aq-mode=3"));
        assert!(params.contains("psy-rd=2.5"));
        assert!(params.contains("rc-lookahead=250"));
        assert!(options.extra_args.contains(&"-profile:v".to_string()));
        assert!(options.extra_args.contains(&"main422-10".to_string()));
        assert!(options.extra_args.contains(&"-tag:v".to_string()));
        assert!(options.extra_args.contains(&"hvc1".to_string()));
    }

    #[test]
    fn test_crf_17_for_museum_quality() {
        let options = VideoEncodingOptions::default();
        assert!(
            options.crf <= 18,
            "CRF should be 18 or lower for museum quality, got {}",
            options.crf
        );
    }

    #[test]
    fn test_422_chroma_subsampling() {
        let options = VideoEncodingOptions::default();
        assert!(
            options.pixel_format.contains("422"),
            "should use 4:2:2 chroma subsampling, got {}",
            options.pixel_format
        );
    }

    #[test]
    fn test_10bit_color_depth() {
        let options = VideoEncodingOptions::default();
        assert!(
            options.pixel_format.contains("10"),
            "should use 10-bit color depth, got {}",
            options.pixel_format
        );
    }

    #[test]
    fn test_x265_profile_matches_pixel_format() {
        let options = VideoEncodingOptions::default();
        let has_422 = options.pixel_format.contains("422");
        let x265_params = options
            .extra_args
            .iter()
            .position(|s| s == "-x265-params")
            .map(|idx| &options.extra_args[idx + 1]);

        if x265_params.is_some() && has_422 {
            assert!(
                options.extra_args.contains(&"main422-10".to_string()),
                "4:2:2 pixel format requires main422-10 profile"
            );
        }
    }
}
