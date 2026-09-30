//! Video encoding functionality
//!
//! Provides website-compatible H.264 plus high-quality H.265 encoding, for the main renderer's
//! Display P3 frames and (the `*_srgb` variants) the ember edition's sRGB frames.

use std::error::Error;
use std::io::Write;
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};
use tracing::info;

use crate::render::error::{RenderError, Result};

/// Configuration for video encoding
///
/// This struct provides fine-grained control over `FFmpeg` encoding parameters.
/// Use [`VideoEncodingOptions::web_compatible`] for the public website MP4 and
/// [`VideoEncodingOptions::high_quality`] for the archival HEVC copy (and their `*_srgb`
/// counterparts for sRGB frames).
#[derive(Debug, Clone)]
pub struct VideoEncodingOptions {
    /// Output bitrate (for a bitrate-targeted encode). Leave empty for CRF mode (quality-based
    /// variable bitrate), which every encode of the generator uses
    pub bitrate: String,

    /// H.264/H.265 preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    /// Slower presets provide better compression at the cost of encoding time
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

    /// Video codec to use: a software encoder (`libx264`, `libx265`); the generator never uses
    /// a hardware encoder
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

/// Colour handling of the ember edition's sRGB video variants.
///
/// The frames piped to `FFmpeg` are sRGB-encoded RGB (BT.709 primaries, D65 white, IEC 61966-2-1
/// transfer). The variants convert them to `Y'CbCr` with the BT.709 matrix at limited ("tv")
/// range and tag the bitstream and the MP4 `colr` box as BT.709 primaries / IEC 61966-2-1
/// transfer / BT.709 matrix / tv range. Two details make this hold on every `FFmpeg` the
/// product meets:
///
/// * The RGB→`Y'CbCr` conversion is an explicit `scale=out_color_matrix=bt709:out_range=tv`.
///   `FFmpeg` 7.1 derives the matrix of its automatically inserted scaler from `-colorspace`,
///   but older releases (the production host runs 6.1) convert RGB with BT.601 while still
///   tagging BT.709. That mismatch moves saturated colours by up to 24 levels (8-bit scale,
///   measured on primaries and the warm red `(177, 34, 16)`); the matched round trip stays
///   within 3 levels at 8-bit 4:2:0 and 0.25 levels at 10 bits.
/// * `setparams` stamps every frame with the sRGB tags. `FFmpeg` 7.1 takes the encoder's
///   primaries and transfer from the frames and drops the `-color_primaries`/`-color_trc`
///   output options, so without it H.264 streams and the `colr` box say "unspecified". The
///   output options stay for older releases, whose encoders read them instead.
///
/// Measured with `FFmpeg` 7.1.1 by the ignored test `srgb_variants_round_trip_through_bt709`
/// (`cargo test --lib srgb_variants_round_trip -- --ignored`, needs `ffmpeg` and `ffprobe`).
mod srgb {
    /// `scale` + `format` + `setparams`: sRGB RGB → BT.709 limited-range `Y'CbCr` in
    /// `pixel_format`, every frame tagged sRGB.
    pub(super) fn filter(pixel_format: &str) -> String {
        format!(
            "scale=out_color_matrix=bt709:out_range=tv,format={pixel_format},\
             setparams=color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709:range=tv"
        )
    }

    /// `-vf` with the conversion [`filter`], then the container/stream tags.
    pub(super) fn args(pixel_format: &str) -> Vec<String> {
        let mut args = vec!["-vf".to_string(), filter(pixel_format)];
        args.extend(super::color_tag_args("bt709", "iec61966-2-1"));
        args
    }
}

/// `-colorspace bt709 -color_primaries <primaries> -color_trc <transfer> -color_range tv`.
fn color_tag_args(primaries: &str, transfer: &str) -> Vec<String> {
    ["-colorspace", "bt709", "-color_primaries", primaries, "-color_trc", transfer]
        .into_iter()
        .chain(["-color_range", "tv"])
        .map(str::to_string)
        .collect()
}

impl VideoEncodingOptions {
    /// Browser-compatible H.264: `libx264`, preset medium, CRF 18, 8-bit 4:2:0, fast start,
    /// then `color_args`.
    fn h264_web(color_args: Vec<String>) -> Self {
        let mut extra_args = vec!["-movflags".to_string(), "+faststart".to_string()];
        extra_args.extend(color_args);
        Self {
            codec: "libx264".to_string(),
            preset: "medium".to_string(),
            crf: 18,
            bitrate: String::new(),
            pixel_format: "yuv420p".to_string(),
            input_pixel_format: "rgb48le".to_string(),
            extra_args,
        }
    }

    /// Archival HEVC: `libx265`, preset slower, CRF 17, 10-bit 4:2:2 (Main 4:2:2 10) with
    /// perceptual tuning and `colorprim` in the x265 VUI, then `conversion_args` (a `-vf`
    /// chain, if any), `color_args` and the `hvc1` tag.
    fn hevc_archival(
        colorprim: &str,
        conversion_args: Vec<String>,
        color_args: Vec<String>,
    ) -> Self {
        let mut extra_args = vec![
            "-profile:v".to_string(),
            "main422-10".to_string(),
            "-x265-params".to_string(),
            format!(
                "bframes=8:ref=6:\
                 rc-lookahead=250:\
                 aq-mode=3:aq-strength=1.0:\
                 psy-rd=2.5:psy-rdoq=1.5:\
                 deblock=-1,-1:\
                 no-sao=0:\
                 colorprim={colorprim}:transfer=iec61966-2-1:colormatrix=bt709:\
                 qg-size=8:\
                 rdoq-level=2"
            ),
            // Content tuning for gradients and smooth motion
            "-tune".to_string(),
            "grain".to_string(),
            // Web optimization (instant playback while streaming)
            "-movflags".to_string(),
            "+faststart".to_string(),
        ];
        extra_args.extend(conversion_args);
        // Color accuracy metadata (critical for correct reproduction)
        extra_args.extend(color_args);
        extra_args.extend(["-tag:v".to_string(), "hvc1".to_string()]);
        Self {
            codec: "libx265".to_string(),
            preset: "slower".to_string(),
            crf: 17,
            bitrate: String::new(),
            pixel_format: "yuv422p10le".to_string(),
            input_pixel_format: "rgb48le".to_string(),
            extra_args,
        }
    }

    /// Website-compatible H.264 MP4 for `QuickTime` and broad browser playback.
    #[must_use]
    pub fn web_compatible() -> Self {
        Self::h264_web(color_tag_args("bt709", "bt709"))
    }

    /// High-quality HEVC copy preserving the existing 10-bit 4:2:2 encode path.
    #[must_use]
    pub fn high_quality() -> Self {
        Self::hevc_archival("smpte432", Vec::new(), color_tag_args("smpte432", "iec61966-2-1"))
    }

    /// [`web_compatible`](Self::web_compatible) for sRGB frames: H.264, CRF 18 by default,
    /// 8-bit 4:2:0, converted with the BT.709 matrix at limited range and tagged BT.709
    /// primaries, IEC 61966-2-1 transfer, BT.709 matrix, tv range.
    ///
    /// The ember edition encodes `videos/web/ember.mp4` with these options but overrides the
    /// rate factor with [`app::EMBER_WEB_CRF`](crate::app::EMBER_WEB_CRF) (22): its textured
    /// paper frames would be about 1.7 times larger at CRF 18 for no visible gain.
    ///
    /// See the notes on the private `srgb` module for why the conversion is explicit.
    #[must_use]
    pub fn web_compatible_srgb() -> Self {
        Self::h264_web(srgb::args("yuv420p"))
    }

    /// [`high_quality`](Self::high_quality) for sRGB frames (the ember edition): HEVC, preset
    /// slower, CRF 17, 10-bit 4:2:2 with the same tuning, converted with the BT.709 matrix at
    /// limited range and tagged sRGB in the container, the stream and the x265 VUI
    /// (`colorprim=bt709:transfer=iec61966-2-1:colormatrix=bt709`).
    #[must_use]
    pub fn high_quality_srgb() -> Self {
        Self::hevc_archival("bt709", srgb::args("yuv422p10le"), Vec::new())
    }

    /// Software-only fast encode of sRGB frames (the ember edition under `--fast-encode`):
    /// `libx264`, preset fast, CRF 21, 10-bit 4:2:0 (High 10), with the sRGB conversion and tags.
    ///
    /// Like [`fast_encode`](Self::fast_encode) it is software only, with the sRGB conversion and
    /// tags of the ember edition instead of Display P3.
    #[must_use]
    pub fn software_fast_srgb() -> Self {
        let pixel_format = "yuv420p10le";
        let mut extra_args: Vec<String> =
            ["-tune", "film", "-movflags", "+faststart"].map(str::to_string).to_vec();
        extra_args.extend(srgb::args(pixel_format));
        Self {
            codec: "libx264".to_string(),
            preset: "fast".to_string(),
            crf: 21,
            bitrate: String::new(),
            pixel_format: pixel_format.to_string(),
            input_pixel_format: "rgb48le".to_string(),
            extra_args,
        }
    }

    /// Fast encode of the main edition's Display P3 frames (`--fast-encode`, for drafts):
    /// `libx264`, preset fast, CRF 21, 10-bit 4:2:0, tagged Display P3.
    ///
    /// Software only, on every platform: nothing in the generator uses a GPU or a hardware
    /// encoder, so a draft runs the same command everywhere.
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
    spawn_piped(build_ffmpeg_command(width, height, frame_rate, spec), &spec.output_file)
}

/// Starts an encoder process that reads frames from a pipe on its stdin (stdout discarded,
/// stderr inherited so its diagnostics reach the log).
fn spawn_piped(mut command: Command, output_file: &str) -> Result<(String, Child, ChildStdin)> {
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(RenderError::VideoEncoding)?;
    let Some(stdin) = child.stdin.take() else {
        kill_encoders(&mut [(output_file.to_string(), child)]);
        return Err(RenderError::VideoEncoding(std::io::Error::other(
            "failed to open ffmpeg stdin",
        )));
    };
    Ok((output_file.to_string(), child, stdin))
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

/// Waits for every encoder to finish its video after the frame stream ended normally (their stdin
/// pipes are closed) and reaps each one, even after another has failed, so that no encoder
/// outlives the call.
///
/// Returns one error naming every encoder that failed, each with its exit status (for example
/// `FFmpeg failed for hq.mp4 with exit status: 1`), in output order. The encoders' own
/// diagnostics are already in the log: their stderr is inherited, not captured.
fn wait_for_encoders(children: Vec<(String, Child)>) -> Result<()> {
    let mut failures = Vec::new();
    for (output_file, mut child) in children {
        match child.wait() {
            Ok(status) if status.success() => info!("   Saved video => {output_file}"),
            Ok(status) => failures.push(format!("FFmpeg failed for {output_file} with {status}")),
            Err(error) => {
                failures.push(format!("could not wait for FFmpeg encoding {output_file}: {error}"));
            }
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(RenderError::VideoEncoding(std::io::Error::other(failures.join("; "))))
    }
}

/// Kills the encoders and reaps them (no zombie processes outlive a failed encode).
fn kill_encoders(children: &mut [(String, Child)]) {
    for (_, child) in children {
        let _ = child.kill();
        let _ = child.wait();
    }
}

/// How long a frame stream that stopped early waits for an encoder that quit reading to exit,
/// so that its exit status (usually the cause) can be reported. An encoder that dies on its
/// own only closes its end of the pipe while shutting down, so by the time a write fails it has
/// all but exited; the grace only delays the error when every encoder is healthy (the frames
/// themselves failed).
const ENCODER_EXIT_GRACE: Duration = Duration::from_secs(1);

/// Ends a frame stream that stopped early with `stream_error` and returns the error to report.
///
/// An encoder that exits on its own (an option its `FFmpeg` rejects, a full disk, a crash)
/// breaks the pipe, so the stream's own error is only a symptom: every encoder that has exited
/// within `grace` is named with its exit status, followed by the stream's error. The stdin
/// pipes stay open until every encoder has been killed and reaped, so a healthy encoder never
/// sees the end of its input and never finalises a truncated video.
fn abort_encoders(
    mut children: Vec<(String, Child)>,
    writer: MultiVideoWriter,
    stream_error: &str,
    grace: Duration,
) -> RenderError {
    let deadline = Instant::now() + grace;
    let mut exited: Vec<Option<ExitStatus>> = vec![None; children.len()];
    loop {
        for (status, (_, child)) in exited.iter_mut().zip(&mut children) {
            if status.is_none() {
                *status = child.try_wait().ok().flatten();
            }
        }
        if exited.iter().any(Option::is_some) || Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    // Kill before closing the pipes: a healthy encoder that saw the end of its input would
    // start finalising a truncated video.
    kill_encoders(&mut children);
    drop(writer);

    let quit: Vec<String> = children
        .iter()
        .zip(&exited)
        .filter_map(|((output_file, _), status)| {
            status.map(|status| format!("FFmpeg for {output_file} exited early with {status}"))
        })
        .collect();
    let message = if quit.is_empty() {
        stream_error.to_string()
    } else {
        format!("{}; the frame stream then failed: {stream_error}", quit.join("; "))
    };
    RenderError::VideoEncoding(std::io::Error::other(message))
}

/// `error` followed by its sources, `": "`-separated (wrapper errors such as
/// [`RenderError::VideoEncoding`] keep their details in the source).
fn error_chain(error: &dyn Error) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(inner) = source {
        text = format!("{text}: {inner}");
        source = inner.source();
    }
    text
}

/// Feeds the frames written by `frames_iter` to every encoder at once, then waits for all of
/// them to finish ([`wait_for_encoders`]). See [`abort_encoders`] for what happens when the
/// stream stops early.
fn stream_to_encoders(
    encoders: Vec<(String, Child, ChildStdin)>,
    mut frames_iter: impl FnMut(&mut dyn Write) -> std::result::Result<(), Box<dyn Error>>,
    grace: Duration,
) -> Result<()> {
    let mut children = Vec::with_capacity(encoders.len());
    let mut writers = Vec::with_capacity(encoders.len());
    for (output_file, child, stdin) in encoders {
        children.push((output_file, child));
        writers.push(stdin);
    }

    let mut writer = MultiVideoWriter { writers };
    if let Err(e) = frames_iter(&mut writer) {
        return Err(abort_encoders(children, writer, &error_chain(e.as_ref()), grace));
    }
    let _ = writer.flush();
    drop(writer);

    wait_for_encoders(children)
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
///
/// If `frames_iter` fails, the error names every encoder that had already exited (with its exit
/// status) before the stream's own error, the encoders are killed and no video is finalised.
/// If the stream completes, every encoder is waited for, and an encoder that fails while
/// finishing its video does not stop the others: the error names each failed output with its
/// exit status.
pub fn create_videos_from_frames_singlepass(
    width: u32,
    height: u32,
    frame_rate: u32,
    frames_iter: impl FnMut(&mut dyn Write) -> std::result::Result<(), Box<dyn Error>>,
    outputs: &[VideoOutputSpec],
) -> Result<()> {
    validate_video_params(width, height, frame_rate, outputs.len())?;

    let mut encoders: Vec<(String, Child, ChildStdin)> = Vec::with_capacity(outputs.len());
    for spec in outputs {
        match spawn_encoder(width, height, frame_rate, spec) {
            Ok(encoder) => encoders.push(encoder),
            Err(e) => {
                let (mut children, stdins): (Vec<_>, Vec<_>) = encoders
                    .into_iter()
                    .map(|(output_file, child, stdin)| ((output_file, child), stdin))
                    .unzip();
                kill_encoders(&mut children);
                drop(stdins);
                return Err(e);
            }
        }
    }

    stream_to_encoders(encoders, frames_iter, ENCODER_EXIT_GRACE)
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

    /// The draft encode is software on every platform (the generator is CPU-only).
    #[test]
    fn test_fast_encode_is_software_everywhere() {
        let fast = VideoEncodingOptions::fast_encode();
        assert_eq!(fast.codec, "libx264");
        assert_eq!(fast.preset, "fast");
        assert_eq!(fast.crf, 21);
        assert_eq!(fast.pixel_format, "yuv420p10le");
        assert_eq!(fast.input_pixel_format, "rgb48le");
        assert!(!fast.extra_args.iter().any(|arg| arg.contains("videotoolbox") || arg == "-q:v"));
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

    /// The value following `flag` in `args`.
    fn value_of<'a>(args: &'a [String], flag: &str) -> &'a str {
        let index = args.iter().position(|arg| arg == flag).unwrap_or_else(|| panic!("no {flag}"));
        &args[index + 1]
    }

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(ToString::to_string).collect()
    }

    /// The main renderer's encodes keep their exact arguments (the shared builders only
    /// deduplicate code).
    #[test]
    fn test_existing_options_are_unchanged() {
        let web = VideoEncodingOptions::web_compatible();
        assert_eq!(
            (web.codec.as_str(), web.preset.as_str(), web.crf, web.pixel_format.as_str()),
            ("libx264", "medium", 18, "yuv420p")
        );
        assert_eq!(
            web.extra_args,
            strings(&[
                "-movflags",
                "+faststart",
                "-colorspace",
                "bt709",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-color_range",
                "tv",
            ])
        );

        let hq = VideoEncodingOptions::high_quality();
        assert_eq!(
            (hq.codec.as_str(), hq.preset.as_str(), hq.crf, hq.pixel_format.as_str()),
            ("libx265", "slower", 17, "yuv422p10le")
        );
        assert_eq!(
            hq.extra_args,
            strings(&[
                "-profile:v",
                "main422-10",
                "-x265-params",
                "bframes=8:ref=6:rc-lookahead=250:aq-mode=3:aq-strength=1.0:psy-rd=2.5:\
                 psy-rdoq=1.5:deblock=-1,-1:no-sao=0:colorprim=smpte432:transfer=iec61966-2-1:\
                 colormatrix=bt709:qg-size=8:rdoq-level=2",
                "-tune",
                "grain",
                "-movflags",
                "+faststart",
                "-colorspace",
                "bt709",
                "-color_primaries",
                "smpte432",
                "-color_trc",
                "iec61966-2-1",
                "-color_range",
                "tv",
                "-tag:v",
                "hvc1",
            ])
        );
    }

    fn srgb_variants() -> [(&'static str, VideoEncodingOptions); 3] {
        [
            ("web", VideoEncodingOptions::web_compatible_srgb()),
            ("hq", VideoEncodingOptions::high_quality_srgb()),
            ("fast", VideoEncodingOptions::software_fast_srgb()),
        ]
    }

    #[test]
    fn test_srgb_variants_convert_with_bt709_and_tag_srgb() {
        for (label, options) in srgb_variants() {
            let args = &options.extra_args;
            assert_eq!(options.input_pixel_format, "rgb48le", "{label}");
            assert_eq!(value_of(args, "-colorspace"), "bt709", "{label}");
            assert_eq!(value_of(args, "-color_primaries"), "bt709", "{label}");
            assert_eq!(value_of(args, "-color_trc"), "iec61966-2-1", "{label}");
            assert_eq!(value_of(args, "-color_range"), "tv", "{label}");
            assert_eq!(value_of(args, "-movflags"), "+faststart", "{label}");
            let filter = value_of(args, "-vf");
            assert!(
                filter.starts_with("scale=out_color_matrix=bt709:out_range=tv,"),
                "{label}: the RGB→YUV conversion must use BT.709 explicitly: {filter}"
            );
            assert!(filter.contains(&format!(",format={},", options.pixel_format)), "{label}");
            assert!(
                filter.ends_with(
                    "setparams=color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709:\
                     range=tv"
                ),
                "{label}: frames must carry the sRGB tags: {filter}"
            );
            assert_eq!(args.iter().filter(|arg| *arg == "-vf").count(), 1, "{label}");
        }
    }

    #[test]
    fn test_srgb_web_variant_matches_web_settings() {
        let web = VideoEncodingOptions::web_compatible();
        let srgb = VideoEncodingOptions::web_compatible_srgb();
        assert_eq!(
            (srgb.codec, srgb.preset, srgb.crf, srgb.pixel_format),
            (web.codec, web.preset, web.crf, web.pixel_format)
        );
    }

    #[test]
    fn test_srgb_hq_variant_matches_hq_tuning() {
        let hq = VideoEncodingOptions::high_quality();
        let srgb = VideoEncodingOptions::high_quality_srgb();
        assert_eq!(
            (&srgb.codec, &srgb.preset, srgb.crf, &srgb.pixel_format),
            (&hq.codec, &hq.preset, hq.crf, &hq.pixel_format)
        );
        let params = value_of(&srgb.extra_args, "-x265-params");
        assert!(params.contains(":colorprim=bt709:transfer=iec61966-2-1:colormatrix=bt709:"));
        assert_eq!(
            params.replace("colorprim=bt709", "colorprim=smpte432"),
            value_of(&hq.extra_args, "-x265-params"),
            "only the primaries differ from the archival tuning"
        );
        assert_eq!(value_of(&srgb.extra_args, "-profile:v"), "main422-10");
        assert_eq!(value_of(&srgb.extra_args, "-tune"), "grain");
        assert_eq!(value_of(&srgb.extra_args, "-tag:v"), "hvc1");
    }

    #[test]
    fn test_software_fast_srgb_never_uses_hardware() {
        let fast = VideoEncodingOptions::software_fast_srgb();
        assert_eq!(fast.codec, "libx264");
        assert_eq!(fast.preset, "fast");
        assert_eq!(fast.crf, 21);
        assert_eq!(fast.pixel_format, "yuv420p10le");
        assert!(fast.bitrate.is_empty());
        assert!(!fast.extra_args.iter().any(|arg| arg.contains("videotoolbox") || arg == "-q:v"));
    }

    #[test]
    fn test_srgb_conversion_filter_is_an_output_option() {
        let spec = VideoOutputSpec {
            output_file: "out.mp4".to_string(),
            options: VideoEncodingOptions::high_quality_srgb(),
        };
        let command = build_ffmpeg_command(64, 32, 60, &spec);
        let args: Vec<String> =
            command.get_args().map(|arg| arg.to_string_lossy().into_owned()).collect();
        let input = args.iter().position(|arg| arg == "-").expect("stdin input");
        let filter = args.iter().position(|arg| arg == "-vf").expect("-vf");
        assert!(input < filter, "-vf must follow the input to apply to the output");
        assert_eq!(args.last().map(String::as_str), Some("out.mp4"));
        assert_eq!(value_of(&args, "-c:v"), "libx265");
        assert_eq!(value_of(&args, "-crf"), "17");
        assert_eq!(value_of(&args, "-pix_fmt"), "rgb48le", "input format first");
        let output_format = args.iter().rposition(|arg| arg == "-pix_fmt").expect("-pix_fmt");
        assert_eq!(args[output_format + 1], "yuv422p10le");
        assert!(input < output_format && output_format < filter);
    }

    /// Solid sRGB test colours (8-bit code values): a saturated warm red, paper and ink black,
    /// the primaries, white and mid grey.
    const ROUND_TRIP_COLOURS: [[u8; 3]; 8] = [
        [177, 34, 16],
        [240, 234, 221],
        [21, 20, 19],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 255],
        [128, 128, 128],
    ];
    const PATCH: u32 = 32;

    /// Largest channel error (8-bit levels) at the patch centres of a decoded `rgb48le` frame.
    fn worst_patch_error(decoded: &[u8], width: u32) -> f64 {
        let mut worst = 0.0f64;
        for (index, colour) in ROUND_TRIP_COLOURS.iter().enumerate() {
            let (x, y) = (index as u32 * PATCH + PATCH / 2, PATCH / 2);
            let offset = ((y * width + x) * 6) as usize;
            for (channel, &expected) in colour.iter().enumerate() {
                let at = offset + 2 * channel;
                let value = u16::from_le_bytes([decoded[at], decoded[at + 1]]);
                worst = worst.max((f64::from(value) / 257.0 - f64::from(expected)).abs());
            }
        }
        worst
    }

    /// Decodes the first frame to `rgb48le`, optionally forcing the `Y'CbCr` matrix.
    fn decode_first_frame(path: &std::path::Path, matrix: Option<&str>) -> Vec<u8> {
        let mut command = Command::new("ffmpeg");
        command.args(["-v", "error", "-i"]).arg(path);
        if let Some(matrix) = matrix {
            command.args(["-vf", &format!("scale=in_color_matrix={matrix}:in_range=tv")]);
        }
        command.args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"]);
        let output = command.output().expect("ffmpeg must be on PATH");
        assert!(
            output.status.success(),
            "decode failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        output.stdout
    }

    /// `(primaries, transfer, matrix, full_range)` of the MP4 `colr` (`nclx`) box.
    fn colr_box(path: &std::path::Path) -> Option<(u16, u16, u16, bool)> {
        let bytes = std::fs::read(path).expect("encoded file");
        let at = bytes.windows(8).position(|window| window == b"colrnclx")? + 8;
        let field =
            |offset: usize| u16::from_be_bytes([bytes[at + offset], bytes[at + offset + 1]]);
        Some((field(0), field(2), field(4), bytes[at + 6] & 0x80 != 0))
    }

    fn probe_stream_tags(path: &std::path::Path) -> String {
        let output = Command::new("ffprobe")
            .args(["-v", "error", "-select_streams", "v:0", "-show_entries"])
            .arg("stream=color_range,color_space,color_transfer,color_primaries")
            .args(["-of", "compact=p=0"])
            .arg(path)
            .output()
            .expect("ffprobe must be on PATH");
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    /// Empirical check of the sRGB variants with the local `FFmpeg`: solid colours piped as
    /// `rgb48le` through each option set must decode back (BT.709, limited range) within the
    /// codec's quantisation, while a BT.601 decode must be far off (so the check discriminates),
    /// and the stream and container must both carry the sRGB tags.
    #[test]
    #[ignore = "needs ffmpeg and ffprobe on PATH; run with --ignored"]
    fn srgb_variants_round_trip_through_bt709() {
        let width = PATCH * ROUND_TRIP_COLOURS.len() as u32;
        let height = PATCH;
        let mut frame = Vec::with_capacity((width * height * 6) as usize);
        for _ in 0..height {
            for colour in ROUND_TRIP_COLOURS {
                for _ in 0..PATCH {
                    for channel in colour {
                        frame.extend_from_slice(&(u16::from(channel) * 257).to_le_bytes());
                    }
                }
            }
        }
        let dir = tempfile::tempdir().expect("temp dir");
        for (label, options) in srgb_variants() {
            let tolerance = if options.pixel_format.contains("10") { 1.0 } else { 4.0 };
            let path = dir.path().join(format!("{label}.mp4"));
            let spec = VideoOutputSpec {
                output_file: path.to_string_lossy().into_owned(),
                options: options.clone(),
            };
            let log = std::fs::File::create(dir.path().join(format!("{label}.log"))).expect("log");
            let mut child = build_ffmpeg_command(width, height, 60, &spec)
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(log)
                .spawn()
                .expect("ffmpeg must be on PATH");
            let mut stdin = child.stdin.take().expect("stdin");
            for _ in 0..6 {
                stdin.write_all(&frame).expect("frame");
            }
            drop(stdin);
            assert!(child.wait().expect("ffmpeg").success(), "{label}: encode failed");

            let honoured = worst_patch_error(&decode_first_frame(&path, None), width);
            let bt709 = worst_patch_error(&decode_first_frame(&path, Some("bt709")), width);
            let bt601 = worst_patch_error(&decode_first_frame(&path, Some("bt601")), width);
            eprintln!(
                "{label}: worst |error| {bt709:.2} BT.709, {honoured:.2} tags, {bt601:.2} BT.601"
            );
            assert!(bt709 <= tolerance, "{label}: BT.709 round trip off by {bt709:.2} levels");
            assert!(
                honoured <= tolerance,
                "{label}: tag-driven decode off by {honoured:.2} levels"
            );
            assert!(bt601 > 10.0, "{label}: a BT.601 decode should be clearly wrong ({bt601:.2})");

            let tags = probe_stream_tags(&path);
            for expected in [
                "color_range=tv",
                "color_space=bt709",
                "color_transfer=iec61966-2-1",
                "color_primaries=bt709",
            ] {
                assert!(tags.contains(expected), "{label}: stream tags {tags} lack {expected}");
            }
            assert_eq!(colr_box(&path), Some((1, 13, 1, false)), "{label}: MP4 colr box");
        }
    }

    /// The encoder lifecycle around a frame stream, with POSIX `sh`/`cat` stand-ins for `FFmpeg`.
    #[cfg(unix)]
    mod encoder_lifecycle {
        use super::*;

        /// A stand-in encoder running the shell `script` with the frame pipe on its stdin.
        fn stand_in(output_file: &str, script: &str) -> (String, Child, ChildStdin) {
            let mut command = Command::new("sh");
            command.args(["-c", script]);
            spawn_piped(command, output_file).expect("sh must be available")
        }

        fn chain(error: &RenderError) -> String {
            error_chain(error)
        }

        /// Writes up to 64 MiB of frames, stopping at the first write error (wrapped the way
        /// the main render's frame sink wraps it).
        fn flood(out: &mut dyn Write) -> std::result::Result<(), Box<dyn Error>> {
            let chunk = vec![0u8; 1 << 16];
            for _ in 0..1024 {
                out.write_all(&chunk).map_err(RenderError::VideoEncoding)?;
            }
            Ok(())
        }

        #[test]
        fn test_an_encoder_that_quits_is_named_with_its_exit_status() {
            let encoders = vec![stand_in("healthy.mp4", "cat"), stand_in("broken.mp4", "exit 3")];
            let error = stream_to_encoders(encoders, flood, ENCODER_EXIT_GRACE)
                .expect_err("the broken encoder breaks the pipe");
            let error = chain(&error);
            assert!(
                error.contains("FFmpeg for broken.mp4 exited early with exit status: 3"),
                "{error}"
            );
            assert!(error.contains("the frame stream then failed"), "{error}");
            assert!(error.contains("Broken pipe"), "the pipe error is kept: {error}");
            assert!(!error.contains("healthy.mp4"), "the healthy encoder is not blamed: {error}");
        }

        #[test]
        fn test_a_failed_frame_stream_kills_healthy_encoders_before_they_finalise() {
            let dir = tempfile::tempdir().expect("temp dir");
            let marker = dir.path().join("finalised");
            // "Finalises" (creates the marker) only if its input ends normally.
            let script = format!("cat >/dev/null && touch '{}'", marker.display());
            let encoders = vec![stand_in("a.mp4", &script), stand_in("b.mp4", &script)];
            let error = stream_to_encoders(
                encoders,
                |out| {
                    out.write_all(&[0u8; 4096])?;
                    Err("ember render: non-finite value".into())
                },
                Duration::from_millis(50),
            )
            .expect_err("the frames failed");
            let error = chain(&error);
            assert!(error.contains("ember render: non-finite value"), "{error}");
            assert!(!error.contains("exited early"), "healthy encoders are not blamed: {error}");
            assert!(!marker.exists(), "a killed encoder must not finalise its video");
        }

        #[test]
        fn test_encoder_exit_status_is_checked_after_the_last_frame() {
            let frames = |out: &mut dyn Write| -> std::result::Result<(), Box<dyn Error>> {
                out.write_all(&[1, 2, 3])?;
                Ok(())
            };
            assert!(
                stream_to_encoders(vec![stand_in("ok.mp4", "cat")], frames, ENCODER_EXIT_GRACE)
                    .is_ok()
            );
            let error = stream_to_encoders(
                vec![stand_in("ok.mp4", "cat"), stand_in("late.mp4", "cat >/dev/null; exit 5")],
                frames,
                ENCODER_EXIT_GRACE,
            )
            .expect_err("the second encoder fails when finishing");
            let error = chain(&error);
            assert!(error.contains("FFmpeg failed for late.mp4 with exit status: 5"), "{error}");
            assert!(!error.contains("ok.mp4"), "the healthy encoder is not blamed: {error}");
            assert!(!error.contains("stderr"), "stderr is inherited, not captured: {error}");
        }

        #[test]
        fn test_every_encoder_is_awaited_and_every_failure_reported() {
            let dir = tempfile::tempdir().expect("temp dir");
            let finished = dir.path().join("slow-finished");
            let frames = |out: &mut dyn Write| -> std::result::Result<(), Box<dyn Error>> {
                out.write_all(&[0u8; 4096])?;
                Ok(())
            };
            // The first encoder fails at once; the others are still finishing their videos.
            let encoders = vec![
                stand_in("first.mp4", "cat >/dev/null; exit 4"),
                stand_in("healthy.mp4", "cat >/dev/null; sleep 0.1"),
                stand_in(
                    "slow.mp4",
                    &format!("cat >/dev/null; sleep 0.3; touch '{}'; exit 6", finished.display()),
                ),
            ];
            let error = stream_to_encoders(encoders, frames, ENCODER_EXIT_GRACE)
                .expect_err("two encoders fail when finishing");
            assert!(finished.exists(), "the slow encoder must be waited for, not abandoned");
            assert_eq!(
                chain(&error),
                "Video encoding failed: FFmpeg failed for first.mp4 with exit status: 4; \
                 FFmpeg failed for slow.mp4 with exit status: 6"
            );
        }

        /// The same with the real `FFmpeg`: an encoder that rejects its options mid-setup is
        /// named with its exit status when the frame pipe breaks.
        #[test]
        #[ignore = "needs ffmpeg on PATH; run with --ignored"]
        fn test_ffmpeg_that_rejects_its_options_is_named_with_its_exit_status() {
            let dir = tempfile::tempdir().expect("temp dir");
            let path = |name: &str| dir.path().join(name).to_string_lossy().into_owned();
            let healthy = VideoEncodingOptions::web_compatible_srgb();
            let broken = VideoEncodingOptions {
                extra_args: vec!["-vf".into(), "no_such_filter".into()],
                ..healthy.clone()
            };
            let outputs = [
                VideoOutputSpec { output_file: path("ok.mp4"), options: healthy },
                VideoOutputSpec { output_file: path("broken.mp4"), options: broken },
            ];
            let frame = vec![0u8; 64 * 32 * 6];
            let error = create_videos_from_frames_singlepass(
                64,
                32,
                60,
                |out| {
                    for _ in 0..1_000 {
                        out.write_all(&frame).map_err(RenderError::VideoEncoding)?;
                    }
                    Ok(())
                },
                &outputs,
            )
            .expect_err("the broken encoder exits");
            let error = chain(&error);
            assert!(error.contains("broken.mp4 exited early with exit status"), "{error}");
            assert!(!error.contains("ok.mp4"), "the healthy encoder is not blamed: {error}");
        }
    }
}
