//! Verify or render the ember edition outside the full generation pipeline.
//!
//! **Verify** a package on any machine (`x86_64`, `aarch64`, …): re-render it from its certificate
//! and compare the SHA-256 digests of the raw frame stream and of the still.
//!
//! ```text
//! cargo run --release --example ember_render -- verify output/<name>/metadata/ember.json
//! ```
//!
//! The re-render takes as long as the original, so `verify` checks everything it can first. It
//! reads the certificate strictly (`EmberCertificate::from_json`), which among other things
//! rejects outputs that disagree about the frames: frames emitted but no frames digest (which
//! would otherwise verify the still alone), or a frames digest over another number of frames
//! than `inputs.frames.count`. It then checks that this build can reproduce the certificate at
//! all, and stops with a message naming every field that rules it out: the rendering algorithm
//! version, the integrator, the time step and the gravitational constant (bit for bit), the
//! paper-seed digest, the frame-schedule digest, and whether the recorded configuration
//! round-trips through this build's `EmberConfig`. Exit status: 0 when both digests match, 1 on
//! a digest mismatch, 2 on any error (a failed check included).
//!
//! **Render** an orbit given by its initial conditions (for look development), optionally with a
//! partial configuration override:
//!
//! ```text
//! cargo run --release --example ember_render -- render --bodies bodies.json --out /tmp/ember \
//!     --resolution 1728x1117 --config '{"tidal": {"max_aspect": 2.0}}' --frame-every 100 --video
//! ```
//!
//! `bodies.json` is either an ember certificate (its `inputs.bodies[*].bits` and `inputs.seed`)
//! or an object with `bodies_f64_bits: [[mass, px, py, pz, vx, vy, vz]; 3]` as `u64` bit patterns
//! (and optionally `seed`, used for the paper texture).

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use nalgebra::Vector3;
use serde_json::Value;
use three_body_problem::app;
use three_body_problem::ember::certificate::{self, EmberCertificate};
use three_body_problem::ember::{
    EmberConfig, EmberError, EmberFrame, EmberMode, EmberRequest, EmberSummary, render_ember,
};
use three_body_problem::render::{
    self, VideoEncodingOptions, VideoOutputSpec, constants, create_videos_from_frames_singlepass,
};
use three_body_problem::sim::{self, Body, get_positions};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Parser)]
#[command(about = "Verify or render the ember edition of an orbit")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Re-render a package from `metadata/ember.json` and compare its digests.
    Verify {
        /// Path of the certificate.
        certificate: PathBuf,
    },
    /// Render an orbit from its initial conditions.
    Render {
        /// Initial conditions (certificate or `bodies_f64_bits` JSON).
        #[arg(long)]
        bodies: PathBuf,
        /// Output directory.
        #[arg(long)]
        out: PathBuf,
        /// Recorded orbit steps (the still shows step `steps - 1`).
        #[arg(long, default_value_t = 1_000_000)]
        steps: usize,
        /// Output size.
        #[arg(long, default_value = "3456x2234")]
        resolution: String,
        /// Partial configuration override (JSON object merged into the defaults).
        #[arg(long)]
        config: Option<String>,
        /// Package seed (hex) for the paper texture; defaults to the file's `seed`.
        #[arg(long)]
        seed: Option<String>,
        /// Save every Nth frame (and the last) as a PNG; 0 renders the still only.
        #[arg(long, default_value_t = 0)]
        frame_every: usize,
        /// Also encode `ember.mp4` (browser H.264).
        #[arg(long)]
        video: bool,
    },
}

fn main() -> ExitCode {
    tracing_subscriber::fmt().with_target(false).init();
    let result = match Cli::parse().command {
        Command::Verify { certificate } => verify(&certificate),
        Command::Render { bodies, out, steps, resolution, config, seed, frame_every, video } => {
            render_orbit(&RenderArgs {
                bodies: &bodies,
                out: &out,
                steps,
                resolution: &resolution,
                config: config.as_deref(),
                seed: seed.as_deref(),
                frame_every,
                video,
            })
        }
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

/// Re-renders a package from its certificate; `Ok(true)` when both digests match.
fn verify(path: &Path) -> Result<bool> {
    let text = fs::read_to_string(path)?;
    let certificate = EmberCertificate::from_json(&text)?;
    let recorded: Value = serde_json::from_str(&text)?;
    let inputs = &certificate.inputs;
    let paper_seed = app::ember_paper_seed(&app::parse_seed(&inputs.seed)?);
    let frame_steps = app::ember_frame_schedule(inputs.steps);

    let problems =
        reproducibility_problems(&certificate, &recorded["config"], &paper_seed, &frame_steps)?;
    if !problems.is_empty() {
        return Err(format!(
            "this build cannot reproduce and verify {}:\n  - {}",
            path.display(),
            problems.join("\n  - ")
        )
        .into());
    }
    println!(
        "checked  {} with {}; dt, G, paper seed, {} scheduled frames and the configuration \
         match this build",
        certificate.algorithm,
        inputs.integrator,
        frame_steps.len()
    );

    // The reader has checked that the outputs agree about the frames, so what is verified
    // follows from the frames digest: without one (a still-only certificate) only the still is.
    let expected_frames = certificate.outputs.frames_rgb48le_sha256.as_deref();
    let expected_still = certificate.outputs.still_rgb48le_sha256.as_str();
    let bodies = inputs.bodies();
    let masses = app::ember_masses(&bodies)?;
    let positions = get_positions(bodies, inputs.steps).positions;
    let request = EmberRequest {
        positions: &positions,
        masses,
        frame_steps: &frame_steps,
        width: inputs.width,
        height: inputs.height,
        paper_seed: &paper_seed,
        config: &certificate.config,
        mode: if expected_frames.is_some() { EmberMode::Video } else { EmberMode::StillOnly },
    };
    let summary = render_ember(&request, &mut |_| Ok(()))?;
    let still_ok = summary.still_sha256 == expected_still;
    let frames_ok = expected_frames.is_none_or(|e| summary.frames_sha256.as_deref() == Some(e));
    println!(
        "still    {}  {}\nframes   {}  {}",
        if still_ok { "MATCH   " } else { "MISMATCH" },
        summary.still_sha256,
        if frames_ok { "MATCH   " } else { "MISMATCH" },
        summary.frames_sha256.as_deref().unwrap_or("(still only)"),
    );
    println!(
        "machine  {} {} with {} threads, render {:.1}s",
        std::env::consts::ARCH,
        std::env::consts::OS,
        rayon::current_num_threads(),
        summary.timings.total_seconds
    );
    Ok(still_ok && frames_ok)
}

/// Everything that stops this build from reproducing and verifying `certificate`, found before
/// the render: one entry per offending field, with the recorded value and this build's (or the
/// field it contradicts).
///
/// `recorded_config` is the certificate's `config` exactly as parsed from the file, so that a
/// field this build's [`EmberConfig`] would default or re-encode is caught.
fn reproducibility_problems(
    certificate: &EmberCertificate,
    recorded_config: &Value,
    paper_seed: &[u8],
    frame_steps: &[usize],
) -> Result<Vec<String>> {
    let inputs = &certificate.inputs;
    let mut problems = Vec::new();
    if certificate.algorithm != certificate::ALGORITHM_VERSION {
        problems.push(format!(
            "algorithm: the certificate was rendered by {:?}, this build implements {:?}",
            certificate.algorithm,
            certificate::ALGORITHM_VERSION
        ));
    }
    if inputs.integrator != certificate::INTEGRATOR {
        problems.push(format!(
            "inputs.integrator: the certificate records {:?}, this build integrates with {:?}",
            inputs.integrator,
            certificate::INTEGRATOR
        ));
    }
    for (field, recorded, ours) in [
        ("inputs.dt", inputs.dt, constants::DEFAULT_DT),
        ("inputs.gravitational_constant", inputs.gravitational_constant, sim::G),
    ] {
        if recorded.to_bits() != ours.to_bits() {
            problems.push(format!(
                "{field}: the certificate records {recorded:?} ({:#018x}), this build uses \
                 {ours:?} ({:#018x})",
                recorded.to_bits(),
                ours.to_bits()
            ));
        }
    }
    let paper = certificate::paper_seed_sha256(paper_seed);
    if inputs.paper_seed_sha256 != paper {
        problems.push(format!(
            "inputs.paper_seed_sha256: the certificate records {}, this build derives {paper} \
             from seed {} (app::ember_paper_seed)",
            inputs.paper_seed_sha256, inputs.seed
        ));
    }
    let schedule = certificate::schedule_sha256(frame_steps);
    if inputs.frames.sha256 != schedule {
        problems.push(format!(
            "inputs.frames.sha256: the certificate records {} ({} frames), this build schedules \
             {} frames for {} steps (app::ember_frame_schedule), digest {schedule}",
            inputs.frames.sha256,
            inputs.frames.count,
            frame_steps.len(),
            inputs.steps
        ));
    }
    let reread = serde_json::to_value(&certificate.config)?;
    if let Some(difference) = first_difference("config", recorded_config, &reread) {
        problems.push(format!(
            "{difference} (the recorded configuration does not round-trip through this build's \
             EmberConfig)"
        ));
    }
    Ok(problems)
}

/// The first place where `recorded` and `reread` differ, as `path: recorded X, read back as Y`,
/// or `None` if they are equal. Numbers compare by value and bit pattern: `3` equals `3.0`,
/// `-0.0` differs from `0.0`.
fn first_difference(path: &str, recorded: &Value, reread: &Value) -> Option<String> {
    match (recorded, reread) {
        (Value::Object(a), Value::Object(b)) => a.keys().chain(b.keys()).find_map(|key| {
            let child = format!("{path}.{key}");
            match (a.get(key), b.get(key)) {
                (Some(x), Some(y)) => first_difference(&child, x, y),
                (x, y) => Some(difference(&child, x, y)),
            }
        }),
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => a
            .iter()
            .zip(b)
            .enumerate()
            .find_map(|(i, (x, y))| first_difference(&format!("{path}[{i}]"), x, y)),
        (Value::Number(a), Value::Number(b))
            if a.as_f64().map(f64::to_bits) == b.as_f64().map(f64::to_bits) =>
        {
            None
        }
        _ if recorded == reread => None,
        _ => Some(difference(path, Some(recorded), Some(reread))),
    }
}

/// `path: recorded X, read back as Y`, with `(absent)` for a missing value.
fn difference(path: &str, recorded: Option<&Value>, reread: Option<&Value>) -> String {
    let show =
        |value: Option<&Value>| value.map_or_else(|| "(absent)".to_owned(), Value::to_string);
    format!("{path}: recorded {}, read back as {}", show(recorded), show(reread))
}

/// Arguments of the `render` subcommand.
struct RenderArgs<'a> {
    bodies: &'a Path,
    out: &'a Path,
    steps: usize,
    resolution: &'a str,
    config: Option<&'a str>,
    seed: Option<&'a str>,
    frame_every: usize,
    video: bool,
}

/// Initial conditions, and the package seed if the file records one, from an ember certificate
/// or a `bodies_f64_bits` file.
fn read_orbit(path: &Path) -> Result<(Vec<Body>, Option<String>)> {
    let text = fs::read_to_string(path)?;
    let json: Value = serde_json::from_str(&text)?;
    if let Some(rows) = json.get("bodies_f64_bits") {
        let rows: Vec<[u64; 7]> = serde_json::from_value(rows.clone())?;
        let bodies = rows
            .into_iter()
            .map(|row| {
                let [mass, px, py, pz, vx, vy, vz] = row.map(f64::from_bits);
                Body::new(mass, Vector3::new(px, py, pz), Vector3::new(vx, vy, vz))
            })
            .collect();
        return Ok((bodies, json.get("seed").and_then(Value::as_str).map(str::to_owned)));
    }
    let certificate = EmberCertificate::from_json(&text)?;
    Ok((certificate.inputs.bodies(), Some(certificate.inputs.seed)))
}

/// Deep-merges `patch` into `base` (objects recursively, everything else replaced).
fn merge(base: &mut Value, patch: &Value) {
    match (base, patch) {
        (Value::Object(base), Value::Object(patch)) => {
            for (key, value) in patch {
                merge(base.entry(key.clone()).or_insert(Value::Null), value);
            }
        }
        (base, patch) => *base = patch.clone(),
    }
}

/// Renders an orbit for look development: the still, optionally every Nth frame as a PNG and
/// the browser video, and a `summary.json` of the render.
fn render_orbit(args: &RenderArgs<'_>) -> Result<bool> {
    let (bodies, recorded_seed) = read_orbit(args.bodies)?;
    let (width, height) = args
        .resolution
        .split_once('x')
        .ok_or("resolution must be WIDTHxHEIGHT")
        .and_then(|(w, h)| {
            Ok((w.parse::<u32>().map_err(|_| "width")?, h.parse::<u32>().map_err(|_| "height")?))
        })?;
    let mut config_json = serde_json::to_value(EmberConfig::default())?;
    if let Some(patch) = args.config {
        merge(&mut config_json, &serde_json::from_str(patch)?);
    }
    let config: EmberConfig = serde_json::from_value(config_json)?;
    let seed = args.seed.map(str::to_owned).or(recorded_seed).unwrap_or_else(|| "0x00".to_owned());
    let paper_seed = app::ember_paper_seed(&app::parse_seed(&seed)?);
    fs::create_dir_all(args.out.join("frames"))?;

    let masses = app::ember_masses(&bodies)?;
    let positions = get_positions(bodies, args.steps).positions;
    let frame_steps = app::ember_frame_schedule(args.steps);
    let video_mode = args.video || args.frame_every > 0;
    let request = EmberRequest {
        positions: &positions,
        masses,
        frame_steps: &frame_steps,
        width,
        height,
        paper_seed: &paper_seed,
        config: &config,
        mode: if video_mode { EmberMode::Video } else { EmberMode::StillOnly },
    };
    let frames_dir = args.out.join("frames");
    let save_frame = |frame: &EmberFrame<'_>| -> Result<()> {
        let is_last = frame.index + 1 == frame.count;
        if args.frame_every > 0 && (frame.index.is_multiple_of(args.frame_every) || is_last) {
            let image = image::ImageBuffer::from_raw(frame.width, frame.height, frame.rgb.to_vec())
                .ok_or("frame size")?;
            let path =
                frames_dir.join(format!("{:05}_step{:07}.png", frame.index, frame.orbit_step));
            render::save_image_as_srgb_png_16bit(&image, &path.to_string_lossy())?;
        }
        Ok(())
    };

    let summary: EmberSummary = if args.video {
        let output = VideoOutputSpec {
            output_file: args.out.join("ember.mp4").to_string_lossy().into_owned(),
            options: VideoEncodingOptions::web_compatible_srgb(),
        };
        let mut result = None;
        create_videos_from_frames_singlepass(
            width,
            height,
            constants::DEFAULT_VIDEO_FPS,
            |pipe| {
                let rendered = render_ember(&request, &mut |frame| {
                    save_frame(frame).map_err(|e| EmberError::Sink(e.to_string()))?;
                    pipe.write_all(frame.rgb48le).map_err(|e| EmberError::Sink(e.to_string()))
                });
                let status = rendered.as_ref().map(|_| ()).map_err(|e| e.to_string().into());
                result = Some(rendered);
                status
            },
            &[output],
        )?;
        result.ok_or("no render")??
    } else {
        render_ember(&request, &mut |frame| {
            save_frame(frame).map_err(|e| EmberError::Sink(e.to_string()))
        })?
    };

    let still = args.out.join("ember.png");
    render::save_image_as_srgb_png_16bit(&summary.still_image(), &still.to_string_lossy())?;
    let report = serde_json::json!({
        "duration": summary.duration,
        "valve_time": summary.valve_time,
        "fluid_grid": summary.fluid_grid,
        "ink_grid": summary.ink_grid,
        "projection": summary.projection,
        "frames": frame_steps.len(),
        "stats": summary.stats,
        "timings_seconds": summary.timings,
        "still_sha256": summary.still_sha256,
        "frames_sha256": summary.frames_sha256,
        "config": config,
    });
    fs::write(args.out.join("summary.json"), serde_json::to_string_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(true)
}
