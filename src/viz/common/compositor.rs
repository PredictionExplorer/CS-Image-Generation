//! Film assembly (master plan II.13): segments cut or crossfaded into one
//! timeline through a single `FFmpeg` `filter_complex` graph, with optional
//! title cards (stills held for a duration) and one pre-mixed WAV bed.
//!
//! The argument vector is a pure function of the inputs (unit-tested), so
//! assembled films are reproducible. `amix`/`adelay` audio graphs are
//! deferred until V49 needs them -- modes bake their mix into one WAV.

use crate::error::Result;
use crate::render::{VideoEncodingOptions, VideoOutputSpec};
use std::fmt::Write as _;
use std::process::Command;

/// How a segment joins the previous one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Transition {
    /// Hard cut.
    Cut,
    /// Crossfade over the given seconds.
    Xfade(f64),
}

/// One timeline segment: a trimmed video or a held title-card still.
#[derive(Clone, Debug)]
pub struct Segment {
    /// Source path (mp4, or png when `is_card`).
    pub source: String,
    /// Still card held for `duration` instead of a video trim.
    pub is_card: bool,
    /// Trim start in seconds (videos only).
    pub trim_start: f64,
    /// Segment duration in seconds.
    pub duration: f64,
    /// EDL label.
    pub label: String,
    /// Join with the previous segment.
    pub transition_in: Transition,
}

/// Build the deterministic `FFmpeg` argument vector for one output.
#[must_use]
pub fn build_args(
    segments: &[Segment],
    audio_wav: Option<&str>,
    width: u32,
    height: u32,
    fps: u32,
    options: &VideoEncodingOptions,
    output_file: &str,
) -> Vec<String> {
    let mut args: Vec<String> = vec!["-y".into(), "-loglevel".into(), "error".into()];
    for segment in segments {
        if segment.is_card {
            args.extend([
                "-loop".into(),
                "1".into(),
                "-t".into(),
                format!("{:.3}", segment.duration),
                "-i".into(),
                segment.source.clone(),
            ]);
        } else {
            args.extend([
                "-ss".into(),
                format!("{:.3}", segment.trim_start),
                "-t".into(),
                format!("{:.3}", segment.duration),
                "-i".into(),
                segment.source.clone(),
            ]);
        }
    }
    if let Some(wav) = audio_wav {
        args.extend(["-i".into(), wav.to_string()]);
    }

    // Normalize every input, then chain concat/xfade.
    let mut graph = String::new();
    for (index, _) in segments.iter().enumerate() {
        // `settb=AVTB` aligns timebases across video and looped-still
        // inputs; xfade refuses mismatched timebases.
        let _ = write!(
            graph,
            "[{index}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,\
             pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},\
             format=yuv420p,settb=AVTB[v{index}];"
        );
    }
    let mut current = "v0".to_string();
    let mut timeline = segments[0].duration;
    for (index, segment) in segments.iter().enumerate().skip(1) {
        let next = format!("j{index}");
        match segment.transition_in {
            Transition::Cut => {
                let _ = write!(graph, "[{current}][v{index}]concat=n=2:v=1:a=0[{next}];");
                timeline += segment.duration;
            }
            Transition::Xfade(fade) => {
                let offset = (timeline - fade).max(0.0);
                let _ = write!(
                    graph,
                    "[{current}][v{index}]xfade=transition=fade:duration={fade:.3}:\
                     offset={offset:.3}[{next}];"
                );
                timeline += segment.duration - fade;
            }
        }
        current = next;
    }
    // Trim the graph's trailing semicolon.
    if graph.ends_with(';') {
        graph.pop();
    }

    args.extend(["-filter_complex".into(), graph, "-map".into(), format!("[{current}]")]);
    if audio_wav.is_some() {
        args.extend([
            "-map".into(),
            format!("{}:a:0", segments.len()),
            "-c:a".into(),
            "aac".into(),
            "-b:a".into(),
            "192k".into(),
            "-shortest".into(),
        ]);
    }
    args.extend(["-c:v".into(), options.codec.clone()]);
    if !options.preset.is_empty() && !options.codec.contains("videotoolbox") {
        args.extend(["-preset".into(), options.preset.clone()]);
    }
    if options.bitrate.is_empty() {
        if !options.codec.contains("videotoolbox") {
            args.extend(["-crf".into(), options.crf.to_string()]);
        }
    } else {
        args.extend(["-b:v".into(), options.bitrate.clone()]);
    }
    args.extend(["-pix_fmt".into(), options.pixel_format.clone()]);
    args.extend(options.extra_args.iter().cloned());
    args.push(output_file.to_string());
    args
}

/// Assemble the timeline into every requested output.
pub fn assemble(
    segments: &[Segment],
    audio_wav: Option<&str>,
    width: u32,
    height: u32,
    fps: u32,
    outputs: &[VideoOutputSpec],
) -> Result<()> {
    assert!(!segments.is_empty(), "compositor needs at least one segment");
    for spec in outputs {
        let args =
            build_args(segments, audio_wav, width, height, fps, &spec.options, &spec.output_file);
        let status = Command::new("ffmpeg")
            .args(&args)
            .status()
            .map_err(|e| std::io::Error::other(format!("compositor ffmpeg spawn failed: {e}")))?;
        if !status.success() {
            return Err(std::io::Error::other(format!(
                "compositor ffmpeg failed ({status}) for {}",
                spec.output_file
            ))
            .into());
        }
    }
    Ok(())
}

/// Total timeline duration in seconds (cuts add, xfades overlap).
#[must_use]
pub fn timeline_seconds(segments: &[Segment]) -> f64 {
    let mut total = 0.0;
    for (index, segment) in segments.iter().enumerate() {
        total += segment.duration;
        if index > 0
            && let Transition::Xfade(fade) = segment.transition_in
        {
            total -= fade;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo_segments() -> Vec<Segment> {
        vec![
            Segment {
                source: "a.mp4".into(),
                is_card: false,
                trim_start: 1.0,
                duration: 3.0,
                label: "open".into(),
                transition_in: Transition::Cut,
            },
            Segment {
                source: "card.png".into(),
                is_card: true,
                trim_start: 0.0,
                duration: 2.0,
                label: "title".into(),
                transition_in: Transition::Xfade(0.5),
            },
            Segment {
                source: "b.mp4".into(),
                is_card: false,
                trim_start: 0.0,
                duration: 4.0,
                label: "montage".into(),
                transition_in: Transition::Cut,
            },
        ]
    }

    #[test]
    fn args_are_a_pure_function_of_inputs() {
        let options = VideoEncodingOptions::web_compatible();
        let first =
            build_args(&demo_segments(), Some("mix.wav"), 1728, 1116, 30, &options, "out.mp4");
        let second =
            build_args(&demo_segments(), Some("mix.wav"), 1728, 1116, 30, &options, "out.mp4");
        assert_eq!(first, second);
        let joined = first.join(" ");
        assert!(joined.contains("xfade=transition=fade:duration=0.500:offset=2.500"));
        assert!(joined.contains("concat=n=2:v=1:a=0"));
        assert!(joined.contains("settb=AVTB"), "xfade needs aligned timebases");
        assert!(joined.contains("-map [j2]"));
        assert!(joined.contains("3:a:0"), "audio input index follows segments");
    }

    #[test]
    fn timeline_accounts_for_xfade_overlap() {
        let total = timeline_seconds(&demo_segments());
        assert!((total - (3.0 + 2.0 + 4.0 - 0.5)).abs() < 1e-9);
    }
}
