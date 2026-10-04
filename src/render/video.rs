//! Video encoding functionality
//!
//! Provides website-compatible H.264 plus high-quality H.265 encoding: sRGB frames (the
//! `*_srgb` variants: every web video, and the ember edition's archival copy) and the main
//! renderer's Display P3 frames (its archival copies).
//!
//! Frames are piped to the encoders as they are rendered, without temporary files: one stream
//! to several encoders ([`create_videos_from_frames_singlepass`]), or several streams at once,
//! each to its own group of encoders ([`create_video_groups_from_frames`]).

use std::error::Error;
use std::io::Write;
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};
use tracing::info;

use crate::render::error::{RenderError, Result};

/// Configuration for video encoding
///
/// This struct provides fine-grained control over `FFmpeg` encoding parameters.
/// Use [`VideoEncodingOptions::web_compatible_srgb`] for a public website MP4 (sRGB frames) and
/// [`VideoEncodingOptions::high_quality`] for the main renderer's archival HEVC copy (Display
/// P3 frames; [`VideoEncodingOptions::high_quality_srgb`] for sRGB frames).
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

/// Colour handling of every encode: RGB frames in, BT.709 limited-range `Y'CbCr` out.
///
/// The frames piped to `FFmpeg` are RGB with the IEC 61966-2-1 (sRGB) transfer curve and one of
/// two sets of primaries, both with the D65 white: BT.709 (sRGB frames: every web video, and
/// the ember edition's archival copy) or Display P3 (`smpte432`: the main renderer's archival
/// copies). Each encode converts them to `Y'CbCr` with the BT.709 matrix at limited ("tv")
/// range and tags the bitstream and the MP4 `colr` box with the frames' primaries, the
/// IEC 61966-2-1 transfer, the BT.709 matrix and tv range. Two details make this hold on every
/// `FFmpeg` the product meets:
///
/// * The RGB→`Y'CbCr` conversion is an explicit `scale=out_color_matrix=bt709:out_range=tv`.
///   `FFmpeg` 7.1 derives the matrix of its automatically inserted scaler from `-colorspace`,
///   but older releases (the production host runs 6.1) convert RGB with BT.601 while still
///   tagging BT.709. That mismatch moves saturated colours by up to 24 levels (8-bit scale,
///   measured on primaries and the warm red `(177, 34, 16)`); the matched round trip stays
///   within 3 levels at 8-bit 4:2:0 and 0.25 levels at 10 bits. (Main-edition films encoded
///   before the conversion was explicit decode closer to their still with BT.601 than with
///   their BT.709 tag.)
/// * `setparams` stamps every frame with the tags. `FFmpeg` 7.1 takes the encoder's primaries
///   and transfer from the frames and drops the `-color_primaries`/`-color_trc` output options,
///   so without it H.264 streams and the `colr` box say "unspecified". The output options stay
///   for older releases, whose encoders read them instead.
///
/// Checked on every test run with `ffmpeg` and `ffprobe` on PATH (required under CI) by
/// `variants_round_trip_through_bt709`; measured with `FFmpeg` 7.1.1 and 6.1.1
/// (`cargo test --release --lib variants_round_trip -- --nocapture` prints the errors).
mod ycbcr {
    /// `FFmpeg` name of the BT.709 (sRGB) primaries.
    pub(super) const SRGB_PRIMARIES: &str = "bt709";
    /// `FFmpeg` name of the Display P3 primaries (SMPTE EG 432-1, D65 white).
    pub(super) const DISPLAY_P3_PRIMARIES: &str = "smpte432";

    /// `scale` + `format` + `setparams`: RGB with `primaries` → BT.709 limited-range `Y'CbCr`
    /// in `pixel_format`, every frame tagged with `primaries` and the sRGB transfer.
    pub(super) fn filter(pixel_format: &str, primaries: &str) -> String {
        format!(
            "scale=out_color_matrix=bt709:out_range=tv,format={pixel_format},\
             setparams=color_primaries={primaries}:color_trc=iec61966-2-1:colorspace=bt709:\
             range=tv"
        )
    }

    /// `-vf` with the conversion [`filter`], then the container/stream tags.
    pub(super) fn args(pixel_format: &str, primaries: &str) -> Vec<String> {
        let mut args = vec!["-vf".to_string(), filter(pixel_format, primaries)];
        args.extend(super::color_tag_args(primaries, "iec61966-2-1"));
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

    /// Software-only fast encode (`--fast-encode`, for drafts): `libx264`, preset fast, CRF 21,
    /// 10-bit 4:2:0 (High 10), converted and tagged for frames with `primaries`.
    ///
    /// Software only, on every platform: nothing in the generator uses a GPU or a hardware
    /// encoder, so a draft runs the same command everywhere.
    fn software_fast(primaries: &str) -> Self {
        let pixel_format = "yuv420p10le";
        let mut extra_args: Vec<String> =
            ["-tune", "film", "-movflags", "+faststart"].map(str::to_string).to_vec();
        extra_args.extend(ycbcr::args(pixel_format, primaries));
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

    /// Archival HEVC of the main renderer's Display P3 frames: preset slower, CRF 17, 10-bit
    /// 4:2:2 (Main 4:2:2 10), converted with the BT.709 matrix at limited range and tagged
    /// Display P3 (`smpte432`) with the sRGB transfer in the container, the stream and the
    /// x265 VUI.
    #[must_use]
    pub fn high_quality() -> Self {
        let primaries = ycbcr::DISPLAY_P3_PRIMARIES;
        Self::hevc_archival(primaries, ycbcr::args("yuv422p10le", primaries), Vec::new())
    }

    /// Website-compatible H.264 MP4 of sRGB frames, for `QuickTime` and broad browser playback:
    /// CRF 18 by default, 8-bit 4:2:0, converted with the BT.709 matrix at limited range and
    /// tagged BT.709 primaries, IEC 61966-2-1 transfer, BT.709 matrix, tv range.
    ///
    /// Every web video is sRGB: the main renderer converts its Display P3 frames first
    /// ([`display_p3::to_srgb_samples`](crate::render::display_p3::to_srgb_samples)). The
    /// ember edition encodes `videos/web/ember.mp4` with these options but overrides the rate
    /// factor with [`app::EMBER_WEB_CRF`](crate::app::EMBER_WEB_CRF) (22): its textured paper
    /// frames would be about 1.7 times larger at CRF 18 for no visible gain.
    ///
    /// See the notes on the private `ycbcr` module for why the conversion is explicit.
    #[must_use]
    pub fn web_compatible_srgb() -> Self {
        Self::h264_web(ycbcr::args("yuv420p", ycbcr::SRGB_PRIMARIES))
    }

    /// [`high_quality`](Self::high_quality) for sRGB frames (the ember edition): HEVC, preset
    /// slower, CRF 17, 10-bit 4:2:2 with the same tuning, converted with the BT.709 matrix at
    /// limited range and tagged sRGB in the container, the stream and the x265 VUI
    /// (`colorprim=bt709:transfer=iec61966-2-1:colormatrix=bt709`).
    #[must_use]
    pub fn high_quality_srgb() -> Self {
        let primaries = ycbcr::SRGB_PRIMARIES;
        Self::hevc_archival(primaries, ycbcr::args("yuv422p10le", primaries), Vec::new())
    }

    /// Software-only fast encode of sRGB frames (the ember edition under `--fast-encode`):
    /// `libx264`, preset fast, CRF 21, 10-bit 4:2:0 (High 10), with the sRGB conversion and tags.
    #[must_use]
    pub fn software_fast_srgb() -> Self {
        Self::software_fast(ycbcr::SRGB_PRIMARIES)
    }

    /// Fast encode of the main renderer's Display P3 frames (`--fast-encode`, for drafts):
    /// [`software_fast_srgb`](Self::software_fast_srgb) converted and tagged Display P3.
    #[must_use]
    pub fn fast_encode() -> Self {
        Self::software_fast(ycbcr::DISPLAY_P3_PRIMARIES)
    }
}

/// The frame pipe of one group of encoders: every byte written goes to each encoder of the
/// group, in output order.
///
/// [`create_video_groups_from_frames`] hands its frame closure one writer per group. Once a
/// write has failed (an encoder exited and broke its pipe) the encoders of the group no longer
/// hold the same stream, so the writer stays failed: every later write returns the same error,
/// and the call ends as a failed frame stream even if the closure ignored the error.
#[derive(Debug)]
pub struct GroupWriter {
    /// The stdin pipes of the group's encoders, in output order.
    stdins: Vec<ChildStdin>,
    /// Kind and text of the first write or flush error, if any.
    failure: Option<(std::io::ErrorKind, String)>,
}

impl GroupWriter {
    fn new(stdins: Vec<ChildStdin>) -> Self {
        Self { stdins, failure: None }
    }

    /// Runs `operation` on the pipe of every encoder of the group, in output order, and
    /// remembers the first error; a writer that has already failed returns that error again.
    fn for_each_pipe(
        &mut self,
        mut operation: impl FnMut(&mut ChildStdin) -> std::io::Result<()>,
    ) -> std::io::Result<()> {
        if let Some((kind, text)) = &self.failure {
            return Err(std::io::Error::new(*kind, text.clone()));
        }
        for stdin in &mut self.stdins {
            if let Err(error) = operation(stdin) {
                self.failure = Some((error.kind(), error.to_string()));
                return Err(error);
            }
        }
        Ok(())
    }

    /// Flushes the pipes after the last frame and returns the text of the writer's first failed
    /// write or flush, if any (the frame closure may not have reported it).
    fn finish(&mut self) -> Option<String> {
        // A failed flush is remembered like a failed write.
        let _ = self.flush();
        self.failure.as_ref().map(|(_, text)| text.clone())
    }
}

impl Write for GroupWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.for_each_pipe(|stdin| stdin.write_all(buf))?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.for_each_pipe(Write::flush)
    }
}

/// A running encoder: its output file, its process and the frame pipe on its stdin.
type Encoder = (String, Child, ChildStdin);

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
) -> Result<Encoder> {
    info!(
        "Encoding {} with codec: {}, pixel format: {}",
        spec.output_file, spec.options.codec, spec.options.pixel_format
    );
    spawn_piped(build_ffmpeg_command(width, height, frame_rate, spec), &spec.output_file)
}

/// Starts an encoder process that reads frames from a pipe on its stdin (stdout discarded,
/// stderr inherited so its diagnostics reach the log).
fn spawn_piped(mut command: Command, output_file: &str) -> Result<Encoder> {
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

/// Starts one encoder per output of every group with `spawn`, in group order and, within a
/// group, in output order.
///
/// If an encoder cannot be started, those already running (of every group) are killed and
/// reaped before their pipes are closed, so none of them writes a video, and the spawn error is
/// returned.
fn spawn_encoder_groups(
    groups: &[&[VideoOutputSpec]],
    mut spawn: impl FnMut(&VideoOutputSpec) -> Result<Encoder>,
) -> Result<Vec<Vec<Encoder>>> {
    let mut started: Vec<Vec<Encoder>> = Vec::with_capacity(groups.len());
    for group in groups {
        let mut encoders = Vec::with_capacity(group.len());
        for spec in *group {
            match spawn(spec) {
                Ok(encoder) => encoders.push(encoder),
                Err(error) => {
                    started.push(encoders);
                    let (mut children, stdins): (Vec<_>, Vec<_>) = started
                        .into_iter()
                        .flatten()
                        .map(|(output_file, child, stdin)| ((output_file, child), stdin))
                        .unzip();
                    kill_encoders(&mut children);
                    drop(stdins);
                    return Err(error);
                }
            }
        }
        started.push(encoders);
    }
    Ok(started)
}

/// Checks the frame geometry and rate shared by every encoder of a call.
fn validate_frame_params(width: u32, height: u32, frame_rate: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(RenderError::InvalidDimensions { width, height });
    }

    if frame_rate == 0 {
        return Err(RenderError::InvalidConfig {
            parameter: "frame_rate".into(),
            reason: "must be greater than 0".into(),
        });
    }

    Ok(())
}

fn validate_video_params(
    width: u32,
    height: u32,
    frame_rate: u32,
    output_count: usize,
) -> Result<()> {
    validate_frame_params(width, height, frame_rate)?;

    if output_count == 0 {
        return Err(RenderError::InvalidConfig {
            parameter: "outputs".into(),
            reason: "must include at least one video output".into(),
        });
    }

    Ok(())
}

/// [`validate_frame_params`], then: at least one group, and at least one output in every group.
fn validate_video_groups(
    width: u32,
    height: u32,
    frame_rate: u32,
    groups: &[&[VideoOutputSpec]],
) -> Result<()> {
    validate_frame_params(width, height, frame_rate)?;

    if groups.is_empty() {
        return Err(RenderError::InvalidConfig {
            parameter: "groups".into(),
            reason: "must include at least one group of video outputs".into(),
        });
    }

    if let Some(index) = groups.iter().position(|group| group.is_empty()) {
        return Err(RenderError::InvalidConfig {
            parameter: "groups".into(),
            reason: format!("group {index} must include at least one video output"),
        });
    }

    Ok(())
}

/// Waits for every encoder (of every group) to finish its video after the frame streams ended
/// normally (their stdin pipes are closed) and reaps each one, even after another has failed, so
/// that no encoder outlives the call.
///
/// Returns one error naming every encoder that failed, each with its exit status (for example
/// `FFmpeg failed for hq.mp4 with exit status: 1`), in group then output order. The encoders'
/// own diagnostics are already in the log: their stderr is inherited, not captured.
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

/// How long frame streams that stopped early wait for an encoder that quit reading to exit,
/// so that its exit status (usually the cause) can be reported. An encoder that dies on its
/// own only closes its end of the pipe while shutting down, so by the time a write fails it has
/// all but exited; the grace only delays the error when every encoder is healthy (the frames
/// themselves failed).
const ENCODER_EXIT_GRACE: Duration = Duration::from_secs(1);

/// Ends frame streams that stopped early with `stream_error` and returns the error to report.
/// `children` are the encoders of every group and `writers` the pipes of every group: there is
/// one abort for the whole call, whichever group the failure came from.
///
/// An encoder that exits on its own (an option its `FFmpeg` rejects, a full disk, a crash)
/// breaks the pipe, so the stream's own error is only a symptom: every encoder that has exited
/// within `grace` is named with its exit status (in group then output order), followed by the
/// stream's error. The stdin pipes of every group stay open until every encoder has been killed
/// and reaped, so a healthy encoder, of the failed group or of another, never sees the end of
/// its input and never finalises a truncated video.
fn abort_encoders(
    mut children: Vec<(String, Child)>,
    writers: Vec<GroupWriter>,
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
    drop(writers);

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

/// Hands `frames` one [`GroupWriter`] per group of `groups`, in order, each feeding every
/// encoder of its group at once, then waits for the encoders of every group to finish
/// ([`wait_for_encoders`]).
///
/// The streams stop early if `frames` returns an error or if a write to any encoder failed
/// (whether or not `frames` reported it). See [`abort_encoders`] for what happens then: no
/// encoder of any group finalises its video.
fn stream_to_encoder_groups(
    groups: Vec<Vec<Encoder>>,
    frames: impl FnOnce(&mut [GroupWriter]) -> std::result::Result<(), Box<dyn Error>>,
    grace: Duration,
) -> Result<()> {
    let mut children = Vec::with_capacity(groups.iter().map(Vec::len).sum());
    let mut writers = Vec::with_capacity(groups.len());
    for encoders in groups {
        let mut stdins = Vec::with_capacity(encoders.len());
        for (output_file, child, stdin) in encoders {
            children.push((output_file, child));
            stdins.push(stdin);
        }
        writers.push(GroupWriter::new(stdins));
    }

    let stream_error = match frames(&mut writers) {
        Err(error) => Some(error_chain(error.as_ref())),
        Ok(()) => writers.iter_mut().find_map(GroupWriter::finish),
    };
    if let Some(stream_error) = stream_error {
        return Err(abort_encoders(children, writers, &stream_error, grace));
    }
    drop(writers);

    wait_for_encoders(children)
}

/// The frame closure of a single stream as the closure of one group.
fn single_group(
    mut frames_iter: impl FnMut(&mut dyn Write) -> std::result::Result<(), Box<dyn Error>>,
) -> impl FnOnce(&mut [GroupWriter]) -> std::result::Result<(), Box<dyn Error>> {
    move |writers: &mut [GroupWriter]| match writers {
        [writer] => frames_iter(writer),
        _ => Err(format!("a single frame stream needs one group, not {}", writers.len()).into()),
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
///
/// This is [`create_video_groups_from_frames`] with one group, `outputs`.
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
    create_video_groups_from_frames(
        width,
        height,
        frame_rate,
        &[outputs],
        single_group(frames_iter),
    )
}

/// Create several films from one render pass: every group of `groups` is its own raw frame
/// stream, encoded to each output of the group, and all the streams are written at once.
///
/// `frames` receives one [`GroupWriter`] per group, in `groups` order, and writes each film's
/// frames to its writer; a writer feeds every encoder of its group, as the single stream of
/// [`create_videos_from_frames_singlepass`] does. The films share `width`, `height` and
/// `frame_rate` but not their frames: the ember edition writes each frame to the film it belongs to,
/// and every tenth frame of its slow film is a frame of its normal film too.
///
/// The encoders of every group succeed or fail together:
///
/// * If `frames` fails, or a write to any encoder fails, the encoders of every group are killed
///   and reaped before any of their pipes is closed, so no video is finalised, not even those of
///   a group whose encoders were healthy. The error names every encoder that had already exited
///   (its output file and exit status, in group then output order) before the stream's own
///   error.
/// * If the streams complete, every encoder of every group is waited for, and an encoder that
///   fails while finishing its video does not stop the others: the error names each failed
///   output with its exit status.
/// * If an encoder cannot be started, those already started are killed and reaped and `frames`
///   is never called.
///
/// # Arguments
/// * `width` - Frame width in pixels (every group)
/// * `height` - Frame height in pixels (every group)
/// * `frame_rate` - Output video framerate (fps, every group)
/// * `groups` - The outputs of each film: at least one group, each with at least one output
/// * `frames` - Closure that writes each film's raw RGB frames to the writer of its group
///
/// # Returns
/// * `Ok(())` once every video of every group is complete
/// * `Err(RenderError)` if `frames` or an encoder fails or the parameters are invalid
///
/// # Example
///
/// ```no_run
/// # use three_body_problem::render::video::{
/// #     VideoEncodingOptions, VideoOutputSpec, create_video_groups_from_frames,
/// # };
/// # use std::io::Write;
/// let output = |file: &str, options| VideoOutputSpec { output_file: file.to_string(), options };
/// let film = [
///     output("film.mp4", VideoEncodingOptions::web_compatible_srgb()),
///     output("film-hq.mp4", VideoEncodingOptions::high_quality_srgb()),
/// ];
/// let slow_film = [output("film-slow.mp4", VideoEncodingOptions::web_compatible_srgb())];
/// let frame = vec![0u8; 1920 * 1080 * 6];
/// create_video_groups_from_frames(1920, 1080, 60, &[&film, &slow_film], |writers| {
///     let [normal, slow] = writers else { return Err("expected two writers".into()) };
///     for index in 0..600 {
///         slow.write_all(&frame)?;
///         if index % 10 == 0 {
///             normal.write_all(&frame)?;
///         }
///     }
///     Ok(())
/// })
/// .expect("encoding should succeed");
/// ```
pub fn create_video_groups_from_frames(
    width: u32,
    height: u32,
    frame_rate: u32,
    groups: &[&[VideoOutputSpec]],
    frames: impl FnOnce(&mut [GroupWriter]) -> std::result::Result<(), Box<dyn Error>>,
) -> Result<()> {
    validate_video_groups(width, height, frame_rate, groups)?;
    let encoders =
        spawn_encoder_groups(groups, |spec| spawn_encoder(width, height, frame_rate, spec))?;
    stream_to_encoder_groups(encoders, frames, ENCODER_EXIT_GRACE)
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
        let options = VideoEncodingOptions::web_compatible_srgb();
        assert_eq!(
            (options.codec.as_str(), options.preset.as_str(), options.crf),
            ("libx264", "medium", 18)
        );
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

    /// The main renderer's archival encode, argument for argument: the archival tuning, then
    /// the explicit BT.709 conversion with the Display P3 tags.
    #[test]
    fn test_main_archival_options_are_pinned() {
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
                "-vf",
                "scale=out_color_matrix=bt709:out_range=tv,format=yuv422p10le,\
                 setparams=color_primaries=smpte432:color_trc=iec61966-2-1:colorspace=bt709:\
                 range=tv",
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

    /// Every encode of the generator with the primaries of the frames it receives.
    fn every_variant() -> [(&'static str, VideoEncodingOptions, &'static str); 5] {
        [
            ("web", VideoEncodingOptions::web_compatible_srgb(), "bt709"),
            ("hq", VideoEncodingOptions::high_quality_srgb(), "bt709"),
            ("fast", VideoEncodingOptions::software_fast_srgb(), "bt709"),
            ("main-hq", VideoEncodingOptions::high_quality(), "smpte432"),
            ("main-fast", VideoEncodingOptions::fast_encode(), "smpte432"),
        ]
    }

    #[test]
    fn test_every_variant_converts_with_bt709_and_tags_its_primaries() {
        for (label, options, primaries) in every_variant() {
            let args = &options.extra_args;
            assert_eq!(options.input_pixel_format, "rgb48le", "{label}");
            assert_eq!(value_of(args, "-colorspace"), "bt709", "{label}");
            assert_eq!(value_of(args, "-color_primaries"), primaries, "{label}");
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
                filter.ends_with(&format!(
                    "setparams=color_primaries={primaries}:color_trc=iec61966-2-1:\
                     colorspace=bt709:range=tv"
                )),
                "{label}: frames must carry their tags: {filter}"
            );
            assert_eq!(args.iter().filter(|arg| *arg == "-vf").count(), 1, "{label}");
        }
    }

    /// The H.264 encodes, argument for argument: every public web film (the main film, the
    /// sweep and the ember films) and the archival copy of a `--fast-encode` draft.
    #[test]
    fn test_h264_options_are_pinned() {
        let web = VideoEncodingOptions::web_compatible_srgb();
        assert_eq!(
            (web.codec.as_str(), web.preset.as_str(), web.crf, web.pixel_format.as_str()),
            ("libx264", "medium", 18, "yuv420p")
        );
        assert_eq!(
            web.extra_args,
            strings(&[
                "-movflags",
                "+faststart",
                "-vf",
                "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,\
                 setparams=color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709:range=tv",
                "-colorspace",
                "bt709",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "iec61966-2-1",
                "-color_range",
                "tv",
            ])
        );
        let fast = VideoEncodingOptions::fast_encode();
        assert_eq!(
            (fast.codec.as_str(), fast.preset.as_str(), fast.crf, fast.pixel_format.as_str()),
            ("libx264", "fast", 21, "yuv420p10le")
        );
        assert_eq!(
            fast.extra_args,
            strings(&[
                "-tune",
                "film",
                "-movflags",
                "+faststart",
                "-vf",
                "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p10le,\
                 setparams=color_primaries=smpte432:color_trc=iec61966-2-1:colorspace=bt709:\
                 range=tv",
                "-colorspace",
                "bt709",
                "-color_primaries",
                "smpte432",
                "-color_trc",
                "iec61966-2-1",
                "-color_range",
                "tv",
            ])
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
        let swap = |args: &[String]| -> Vec<String> {
            args.iter()
                .map(|arg| arg.replace("bt709", "PRIM").replace("smpte432", "PRIM"))
                .collect()
        };
        assert_eq!(swap(&srgb.extra_args), swap(&hq.extra_args), "same arguments otherwise");
    }

    #[test]
    fn test_main_fast_encode_matches_srgb_fast_encode() {
        let p3 = VideoEncodingOptions::fast_encode();
        let srgb = VideoEncodingOptions::software_fast_srgb();
        assert_eq!(
            (&p3.codec, &p3.preset, p3.crf, &p3.pixel_format),
            (&srgb.codec, &srgb.preset, srgb.crf, &srgb.pixel_format)
        );
        let swap = |args: &[String]| -> Vec<String> {
            args.iter().map(|arg| arg.replace("smpte432", "bt709")).collect()
        };
        assert_eq!(swap(&p3.extra_args), srgb.extra_args, "only the primaries differ");
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
    /// and the stream and container must both carry the frames' primaries with the sRGB
    /// transfer. (The test colours are code values: the matrix does not depend on the
    /// primaries.)
    #[test]
    fn variants_round_trip_through_bt709() {
        let test = "variants_round_trip_through_bt709";
        if !crate::test_support::media_tools_available(test, &["ffmpeg", "ffprobe"]) {
            return;
        }
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
        for (label, options, primaries) in every_variant() {
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
            let primaries_tag = format!("color_primaries={primaries}");
            for expected in [
                "color_range=tv",
                "color_space=bt709",
                "color_transfer=iec61966-2-1",
                primaries_tag.as_str(),
            ] {
                assert!(tags.contains(expected), "{label}: stream tags {tags} lack {expected}");
            }
            // H.273 code points: primaries 1 = BT.709, 12 = Display P3; transfer 13 = sRGB.
            let primaries_code = if primaries == "smpte432" { 12 } else { 1 };
            assert_eq!(
                colr_box(&path),
                Some((primaries_code, 13, 1, false)),
                "{label}: MP4 colr box"
            );
        }
    }

    fn default_output(output_file: &str) -> VideoOutputSpec {
        VideoOutputSpec {
            output_file: output_file.to_string(),
            options: VideoEncodingOptions::default(),
        }
    }

    /// The error of a call with invalid `groups`, which must neither start an encoder nor ask
    /// for a frame.
    fn rejected_groups(
        width: u32,
        height: u32,
        frame_rate: u32,
        groups: &[&[VideoOutputSpec]],
    ) -> String {
        let mut asked_for_frames = false;
        let error = create_video_groups_from_frames(width, height, frame_rate, groups, |_| {
            asked_for_frames = true;
            Ok(())
        })
        .expect_err("the parameters are invalid");
        assert!(!asked_for_frames, "no frame may be rendered for invalid parameters");
        error.to_string()
    }

    #[test]
    fn test_a_call_without_groups_is_rejected() {
        assert_eq!(
            rejected_groups(64, 32, 60, &[]),
            "Invalid configuration for 'groups': must include at least one group of video outputs"
        );
    }

    #[test]
    fn test_an_empty_group_is_rejected_before_any_encoder_starts() {
        let film = [default_output("film.mp4")];
        assert_eq!(
            rejected_groups(64, 32, 60, &[&film, &[]]),
            "Invalid configuration for 'groups': group 1 must include at least one video output"
        );
        assert_eq!(
            rejected_groups(64, 32, 60, &[&[]]),
            "Invalid configuration for 'groups': group 0 must include at least one video output"
        );
    }

    #[test]
    fn test_groups_keep_the_frame_parameter_checks() {
        let film = [default_output("film.mp4")];
        assert_eq!(rejected_groups(0, 32, 60, &[&film]), "Invalid dimensions: width=0, height=32");
        assert_eq!(rejected_groups(64, 0, 60, &[&film]), "Invalid dimensions: width=64, height=0");
        assert_eq!(
            rejected_groups(64, 32, 0, &[&film]),
            "Invalid configuration for 'frame_rate': must be greater than 0"
        );
        // The frame parameters are checked first, as for a single stream.
        assert_eq!(rejected_groups(0, 32, 60, &[]), "Invalid dimensions: width=0, height=32");
    }

    /// A single stream without outputs keeps its own error (not the group form's).
    #[test]
    fn test_a_single_stream_without_outputs_is_rejected_as_before() {
        let error = create_videos_from_frames_singlepass(64, 32, 60, |_| Ok(()), &[])
            .expect_err("there is no output");
        assert_eq!(
            error.to_string(),
            "Invalid configuration for 'outputs': must include at least one video output"
        );
    }

    /// The encoder lifecycle around the frame streams, with POSIX `sh`/`cat` stand-ins for
    /// `FFmpeg`.
    #[cfg(unix)]
    mod encoder_lifecycle {
        use super::*;
        use std::path::Path;

        /// A stand-in encoder running the shell `script` with the frame pipe on its stdin.
        fn stand_in(output_file: &str, script: &str) -> Encoder {
            let mut command = Command::new("sh");
            command.args(["-c", script]);
            spawn_piped(command, output_file).expect("sh must be available")
        }

        /// The script of a stand-in that "finalises" (creates `marker`) only if its input ends
        /// normally.
        fn finalising(marker: &Path) -> String {
            format!("cat >/dev/null && touch '{}'", marker.display())
        }

        /// The script of a stand-in that copies its input to `file`.
        fn recording(file: &Path) -> String {
            format!("cat >'{}'", file.display())
        }

        /// A stand-in that reads up to 64 KiB of frames, then dies with exit status 3.
        const DIES_MID_STREAM: &str = "dd bs=1024 count=64 >/dev/null 2>&1; exit 3";

        /// The process ids of the encoders (of one group).
        fn pids(encoders: &[Encoder]) -> Vec<u32> {
            encoders.iter().map(|(_, child, _)| child.id()).collect()
        }

        /// Whether the process still exists: running, or exited but not reaped.
        fn exists(pid: u32) -> bool {
            Command::new("sh")
                .args(["-c", &format!("kill -0 {pid} 2>/dev/null")])
                .status()
                .expect("sh must be available")
                .success()
        }

        /// Asserts that the encoders were killed and reaped, not left to finish or abandoned.
        fn assert_gone(pids: &[u32]) {
            for &pid in pids {
                assert!(!exists(pid), "encoder process {pid} must be killed and reaped");
            }
        }

        fn chain(error: &RenderError) -> String {
            error_chain(error)
        }

        /// [`stream_to_encoder_groups`] with `encoders` as its one group, fed the way
        /// [`create_videos_from_frames_singlepass`] feeds its outputs.
        fn stream_to_encoders(
            encoders: Vec<Encoder>,
            frames_iter: impl FnMut(&mut dyn Write) -> std::result::Result<(), Box<dyn Error>>,
            grace: Duration,
        ) -> Result<()> {
            stream_to_encoder_groups(vec![encoders], single_group(frames_iter), grace)
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

        /// [`flood`] for several groups: up to 64 MiB of frames to each, a chunk to every group
        /// in turn, stopping at the first write error.
        fn flood_groups(writers: &mut [GroupWriter]) -> std::result::Result<(), Box<dyn Error>> {
            let chunk = vec![0u8; 1 << 16];
            for _ in 0..1024 {
                for writer in &mut *writers {
                    writer.write_all(&chunk).map_err(RenderError::VideoEncoding)?;
                }
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
            let script = finalising(&marker);
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

        /// The ember edition's layout: every frame goes to the slow film (one encoder), every
        /// tenth also to the normal film (two encoders).
        #[test]
        fn test_each_group_receives_exactly_its_own_stream() {
            const FRAME_BYTES: usize = 4096;
            const SLOW_FRAMES: usize = 200;
            // Frame `index` of the slow film; no two frames in a row are alike.
            let frame = |index: usize| vec![(index % 251) as u8; FRAME_BYTES];
            let normal_film: Vec<u8> = (0..SLOW_FRAMES).step_by(10).flat_map(frame).collect();
            let slow_film: Vec<u8> = (0..SLOW_FRAMES).flat_map(frame).collect();

            let dir = tempfile::tempdir().expect("temp dir");
            let file = |name: &str| dir.path().join(name);
            let groups = vec![
                vec![
                    stand_in("web.mp4", &recording(&file("web"))),
                    stand_in("hq.mp4", &recording(&file("hq"))),
                ],
                vec![stand_in("slow.mp4", &recording(&file("slow")))],
            ];
            stream_to_encoder_groups(
                groups,
                |writers| {
                    let [normal, slow] = writers else {
                        return Err(format!("{} writers for two groups", writers.len()).into());
                    };
                    for index in 0..SLOW_FRAMES {
                        slow.write_all(&frame(index))?;
                        if index % 10 == 0 {
                            normal.write_all(&frame(index))?;
                        }
                    }
                    Ok(())
                },
                ENCODER_EXIT_GRACE,
            )
            .expect("every encoder completes");

            let recorded = |name: &str| std::fs::read(file(name)).expect("recorded stream");
            assert_eq!(slow_film.len(), 10 * normal_film.len());
            assert!(recorded("web") == normal_film, "web.mp4 holds exactly the normal film");
            assert!(recorded("hq") == normal_film, "hq.mp4 holds exactly the normal film");
            assert!(recorded("slow") == slow_film, "slow.mp4 holds exactly the slow film");
        }

        #[test]
        fn test_an_encoder_dying_in_the_second_group_kills_the_first_group_unfinalised() {
            let dir = tempfile::tempdir().expect("temp dir");
            let (web, hq) = (dir.path().join("web-finalised"), dir.path().join("hq-finalised"));
            let normal =
                vec![stand_in("web.mp4", &finalising(&web)), stand_in("hq.mp4", &finalising(&hq))];
            let slow = vec![stand_in("slow.mp4", DIES_MID_STREAM)];
            let healthy = pids(&normal);

            let error =
                stream_to_encoder_groups(vec![normal, slow], flood_groups, ENCODER_EXIT_GRACE)
                    .expect_err("the slow film's encoder breaks its pipe");
            let error = chain(&error);
            assert_eq!(
                error,
                "Video encoding failed: FFmpeg for slow.mp4 exited early with exit status: 3; \
                 the frame stream then failed: Video encoding failed: Broken pipe (os error 32)"
            );
            assert!(!web.exists(), "the other group's web encoder must not finalise its video");
            assert!(!hq.exists(), "the other group's hq encoder must not finalise its video");
            assert_gone(&healthy);
        }

        #[test]
        fn test_an_encoder_dying_in_the_first_group_kills_the_second_group_unfinalised() {
            let dir = tempfile::tempdir().expect("temp dir");
            let (hq, slow) = (dir.path().join("hq-finalised"), dir.path().join("slow-finalised"));
            let normal =
                vec![stand_in("web.mp4", DIES_MID_STREAM), stand_in("hq.mp4", &finalising(&hq))];
            let slow_group = vec![stand_in("slow.mp4", &finalising(&slow))];
            let healthy = [pids(&normal[1..]), pids(&slow_group)].concat();

            let error = stream_to_encoder_groups(
                vec![normal, slow_group],
                flood_groups,
                ENCODER_EXIT_GRACE,
            )
            .expect_err("the normal film's web encoder breaks its pipe");
            let error = chain(&error);
            assert_eq!(
                error,
                "Video encoding failed: FFmpeg for web.mp4 exited early with exit status: 3; \
                 the frame stream then failed: Video encoding failed: Broken pipe (os error 32)"
            );
            assert!(!slow.exists(), "the other group's encoder must not finalise its video");
            assert!(!hq.exists(), "the failed group's healthy encoder must not finalise either");
            assert_gone(&healthy);
        }

        #[test]
        fn test_a_failed_frame_closure_kills_every_encoder_of_every_group() {
            let dir = tempfile::tempdir().expect("temp dir");
            let markers = ["web", "hq", "slow"].map(|name| dir.path().join(name));
            let [web, hq, slow] = &markers;
            let normal =
                vec![stand_in("web.mp4", &finalising(web)), stand_in("hq.mp4", &finalising(hq))];
            let slow_group = vec![stand_in("slow.mp4", &finalising(slow))];
            let all = [pids(&normal), pids(&slow_group)].concat();

            let error = stream_to_encoder_groups(
                vec![normal, slow_group],
                |writers| {
                    for writer in writers {
                        writer.write_all(&[0u8; 4096])?;
                    }
                    Err("ember render: non-finite value".into())
                },
                Duration::from_millis(50),
            )
            .expect_err("the frames failed");
            assert_eq!(chain(&error), "Video encoding failed: ember render: non-finite value");
            for marker in &markers {
                assert!(!marker.exists(), "{} must not be finalised", marker.display());
            }
            assert_gone(&all);
        }

        /// A frame closure that carries on after a failed write cannot finalise any film: the
        /// writer stays failed, passes nothing more to its healthy encoder, and the streams are
        /// aborted.
        #[test]
        fn test_a_write_error_the_closure_ignores_still_aborts_every_group() {
            let dir = tempfile::tempdir().expect("temp dir");
            let recorded = dir.path().join("web-frames");
            let (web, slow) = (dir.path().join("web-finalised"), dir.path().join("slow-finalised"));
            let recording_web = format!("{} && touch '{}'", recording(&recorded), web.display());
            let normal =
                vec![stand_in("web.mp4", &recording_web), stand_in("hq.mp4", DIES_MID_STREAM)];
            let slow_group = vec![stand_in("slow.mp4", &finalising(&slow))];
            let healthy = [pids(&normal[..1]), pids(&slow_group)].concat();

            let error = stream_to_encoder_groups(
                vec![normal, slow_group],
                |writers| {
                    let [normal, slow] = writers else {
                        return Err(format!("{} writers for two groups", writers.len()).into());
                    };
                    let chunk = vec![0u8; 1 << 16];
                    let first_failure = (0..64)
                        .find_map(|_| {
                            let _ = slow.write_all(&chunk);
                            normal.write_all(&chunk).err()
                        })
                        .expect("the hq encoder breaks its pipe");
                    // The careless frame source carries on: more than a pipe can buffer for the
                    // failed group (so it would reach web.mp4's recording), then the other group.
                    let later_failure = normal
                        .write_all(&vec![0xFF; 1 << 20])
                        .expect_err("a failed writer accepts no more frames");
                    assert_eq!(later_failure.kind(), first_failure.kind());
                    assert_eq!(later_failure.to_string(), first_failure.to_string());
                    let _ = slow.write_all(&chunk);
                    Ok(())
                },
                ENCODER_EXIT_GRACE,
            )
            .expect_err("a write failed");
            assert_eq!(
                chain(&error),
                "Video encoding failed: FFmpeg for hq.mp4 exited early with exit status: 3; \
                 the frame stream then failed: Broken pipe (os error 32)"
            );
            assert!(!web.exists(), "the failed group's healthy encoder must not finalise");
            assert!(!slow.exists(), "the healthy group must not finalise its video");
            assert_gone(&healthy);
            let frames = std::fs::read(&recorded).unwrap_or_default();
            assert!(!frames.contains(&0xFF), "nothing is written to a group after its failure");
        }

        #[test]
        fn test_failures_after_the_last_frame_are_reported_across_groups() {
            let groups = vec![
                vec![stand_in("web.mp4", "cat >/dev/null; exit 4"), stand_in("hq.mp4", "cat")],
                vec![stand_in("slow.mp4", "cat >/dev/null; exit 6")],
            ];
            let error = stream_to_encoder_groups(
                groups,
                |writers| {
                    for writer in writers {
                        writer.write_all(&[0u8; 4096])?;
                    }
                    Ok(())
                },
                ENCODER_EXIT_GRACE,
            )
            .expect_err("an encoder of each group fails when finishing");
            assert_eq!(
                chain(&error),
                "Video encoding failed: FFmpeg failed for web.mp4 with exit status: 4; \
                 FFmpeg failed for slow.mp4 with exit status: 6"
            );
        }

        #[test]
        fn test_a_spawn_failure_kills_the_encoders_already_started_in_every_group() {
            let dir = tempfile::tempdir().expect("temp dir");
            let marker = |spec: &VideoOutputSpec| dir.path().join(&spec.output_file);
            let normal = [default_output("web"), default_output("hq")];
            let slow = [default_output("slow"), default_output("unstartable")];

            let mut started = Vec::new();
            let error = spawn_encoder_groups(&[&normal, &slow], |spec| {
                if spec.output_file == "unstartable" {
                    return Err(RenderError::VideoEncoding(std::io::Error::other("no ffmpeg")));
                }
                let encoder = stand_in(&spec.output_file, &finalising(&marker(spec)));
                started.push(encoder.1.id());
                Ok(encoder)
            })
            .map(|_| ())
            .expect_err("the last encoder cannot be started");
            assert_eq!(chain(&error), "Video encoding failed: no ffmpeg");
            assert_eq!(started.len(), 3, "both groups had started encoders");
            for spec in normal.iter().chain(&slow) {
                assert!(!marker(spec).exists(), "{} must not be finalised", spec.output_file);
            }
            assert_gone(&started);
        }

        #[test]
        fn test_encoders_are_started_in_group_then_output_order() {
            let normal = [default_output("web"), default_output("hq")];
            let slow = [default_output("slow")];
            let groups = spawn_encoder_groups(&[&normal, &slow], |spec| {
                Ok(stand_in(&spec.output_file, "cat >/dev/null"))
            })
            .expect("every stand-in starts");
            let outputs: Vec<Vec<&str>> = groups
                .iter()
                .map(|group| group.iter().map(|(output_file, ..)| output_file.as_str()).collect())
                .collect();
            assert_eq!(outputs, [vec!["web", "hq"], vec!["slow"]]);
            stream_to_encoder_groups(groups, |_| Ok(()), ENCODER_EXIT_GRACE)
                .expect("every stand-in finishes");
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

        /// The number of frames `ffprobe` decodes from the video.
        fn decoded_frames(path: &str) -> usize {
            let output = Command::new("ffprobe")
                .args(["-v", "error", "-select_streams", "v:0", "-count_frames"])
                .args(["-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path])
                .output()
                .expect("ffprobe must be on PATH");
            String::from_utf8_lossy(&output.stdout).trim().parse().expect("a frame count")
        }

        /// Two films from one pass with the real `FFmpeg`, in the ember edition's layout: the
        /// slow film has ten times the frames of the normal film's two encodes.
        #[test]
        #[ignore = "needs ffmpeg and ffprobe on PATH; run with --ignored"]
        fn test_ffmpeg_encodes_two_films_from_one_pass() {
            let dir = tempfile::tempdir().expect("temp dir");
            let output = |name: &str, options| VideoOutputSpec {
                output_file: dir.path().join(name).to_string_lossy().into_owned(),
                options,
            };
            let normal = [
                output("ember.mp4", VideoEncodingOptions::web_compatible_srgb()),
                output("ember-hq.mp4", VideoEncodingOptions::high_quality_srgb()),
            ];
            let slow = [output("ember-slow.mp4", VideoEncodingOptions::web_compatible_srgb())];
            let frame = vec![0x80u8; 64 * 32 * 6];
            create_video_groups_from_frames(64, 32, 60, &[&normal, &slow], |writers| {
                let [normal, slow] = writers else {
                    return Err(format!("{} writers for two groups", writers.len()).into());
                };
                for index in 0..120 {
                    slow.write_all(&frame)?;
                    if index % 10 == 0 {
                        normal.write_all(&frame)?;
                    }
                }
                Ok(())
            })
            .expect("both films are encoded");
            assert_eq!(decoded_frames(&normal[0].output_file), 12);
            assert_eq!(decoded_frames(&normal[1].output_file), 12);
            assert_eq!(decoded_frames(&slow[0].output_file), 120);
        }
    }
}
