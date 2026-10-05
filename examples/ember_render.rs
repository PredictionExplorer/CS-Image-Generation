//! Verify or render the ember edition outside the full generation pipeline.
//!
//! **Verify** a package on any machine (`x86_64`, `aarch64`, …): re-render it from its certificate
//! and compare the SHA-256 digests of the still and of the raw frame streams of every film (the
//! normal one and each slow one), whichever the certificate records, and with them everything
//! else the certificate says the render produced: the frame counts, the derived quantities
//! (`derived`) and the statistics (`stats`), which are as deterministic as the pixels.
//!
//! ```text
//! cargo run --release --example ember_render -- verify output/<name>/metadata/ember.json
//! ```
//!
//! The re-render takes as long as the original, so `verify` checks everything it can first. It
//! reads the certificate strictly (`EmberCertificate::from_json`), which among other things
//! rejects outputs that disagree about the frames: frames emitted but no frames digest (which
//! would otherwise verify the still alone), a frames digest over another number of frames than
//! `inputs.frames.count`, or slow films other than those of `inputs.frames.slow_factors`, some
//! rendered and some not, or of another length than their factor and first frame give. It then
//! checks that this build can reproduce the certificate at
//! all, and stops with a message naming every field that rules it out: the rendering algorithm
//! version, the integrator, the time step and the gravitational constant (bit for bit), the
//! paper-seed digest, the frame-schedule digest, and whether the recorded configuration
//! round-trips through this build's `EmberConfig`. The view (`inputs.view`) and the slow factors
//! (`inputs.frames.slow_factors`) are inputs like the bodies: the re-render follows what the
//! certificate records. Exit status: 0 when everything matches, 1 on a mismatch, 2 on any error
//! (a failed check included).
//!
//! **Render** an orbit given by its initial conditions (for look development), optionally with a
//! partial configuration override:
//!
//! ```text
//! cargo run --release --example ember_render -- render --bodies bodies.json --out /tmp/ember \
//!     --resolution 1728x1117 --config '{"tidal": {"max_aspect": 2.0}}' --frame-every 100 \
//!     --video --slow-videos
//! ```
//!
//! `bodies.json` is either an ember certificate (its `inputs.bodies[*].bits`, `inputs.seed` and
//! `inputs.view`; a certificate of an earlier look will do, so that a published edition can be
//! rendered again in this one) or an object with
//! `bodies_f64_bits: [[mass, px, py, pz, vx, vy, vz]; 3]` as `u64` bit patterns (and optionally
//! `seed`, used for the paper texture).
//!
//! The orbit is drawn through the certificate's view — the main edition's projection, viewing
//! rotation, drift and frame — which fits only the orbit it was recorded for: the same `--steps`
//! and an output of the same aspect ratio. With `--frontal`, for a `bodies_f64_bits` file, and
//! for a certificate that records no view (layouts before 4, `ember-v1` and `ember-v2`; a note
//! says so), the orbit is drawn as it is (plain positions seen along the z axis, no drift), not
//! as the main edition shows it.
//!
//! The request is planned before any encoder starts, so a request the renderer refuses (e.g.
//! `--slow-factors 10,4`) leaves the videos already in `--out` untouched.

use std::error::Error;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use nalgebra::Vector3;
use serde_json::Value;
use three_body_problem::app;
use three_body_problem::ember::certificate::{
    ALGORITHM_VERSION, BodyRecord, CertificateDerived, CertificateOutputs, EmberCertificate,
    INTEGRATOR, paper_seed_sha256, schedule_sha256,
};
use three_body_problem::ember::{
    EmberConfig, EmberError, EmberFrame, EmberMode, EmberRequest, EmberSummary, View, plan_ember,
    render_ember,
};
use three_body_problem::render::{
    self, GroupWriter, VideoEncodingOptions, VideoOutputSpec, constants,
    create_video_groups_from_frames,
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
        #[arg(value_name = "CERTIFICATE")]
        path: PathBuf,
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
        /// Save every Nth scheduled frame (and the last) as a PNG; 0 saves none.
        #[arg(long, default_value_t = 0)]
        frame_every: usize,
        /// Also encode `ember.mp4` (browser H.264, as `videos/web/ember.mp4`).
        #[arg(long)]
        video: bool,
        /// Also encode each slow film as `ember_<factor>x.mp4` (browser H.264, as the package's).
        #[arg(long)]
        slow_videos: bool,
        /// How many times slower each slow film is, strictly increasing (comma-separated). The
        /// snapshot lattice of every render is a multiple of every factor, so they change the
        /// still as well.
        #[arg(long, value_delimiter = ',', default_values_t = app::EMBER_SLOW_FACTORS)]
        slow_factors: Vec<u32>,
        /// Draw the orbit as it is, even if `--bodies` is a certificate that records a view.
        #[arg(long)]
        frontal: bool,
    },
}

fn main() -> ExitCode {
    tracing_subscriber::fmt().with_target(false).init();
    let result = match Cli::parse().command {
        Command::Verify { path } => verify(&path),
        Command::Render {
            bodies,
            out,
            steps,
            resolution,
            config,
            seed,
            frame_every,
            video,
            slow_videos,
            slow_factors,
            frontal,
        } => render_orbit(&RenderArgs {
            bodies: &bodies,
            out: &out,
            steps,
            resolution: &resolution,
            config: config.as_deref(),
            seed: seed.as_deref(),
            frame_every,
            video,
            slow_videos,
            slow_factors: &slow_factors,
            frontal,
        }),
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

/// Re-renders a package from its certificate; `Ok(true)` when every recorded digest matches and
/// the re-render's frame counts, derived quantities and statistics are the recorded ones.
fn verify(path: &Path) -> Result<bool> {
    let text = fs::read_to_string(path)?;
    let record = EmberCertificate::from_json(&text)?;
    let recorded: Value = serde_json::from_str(&text)?;
    let inputs = &record.inputs;
    let paper_seed = app::ember_paper_seed(&app::parse_seed(&inputs.seed)?);
    let frame_steps = app::ember_frame_schedule(inputs.steps);

    let problems =
        reproducibility_problems(&record, &recorded["config"], &paper_seed, &frame_steps)?;
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
        record.algorithm,
        inputs.integrator,
        frame_steps.len()
    );

    // The reader has checked that the outputs agree about the frames, so what is verified
    // follows from the digests: without a frames digest (a still-only certificate) only the
    // still is, and the slow films only if their digests are recorded.
    let outputs = &record.outputs;
    let expected_frames = outputs.frames_rgb48le_sha256.as_deref();
    let slow_rendered = outputs.slow_films.iter().any(|film| film.frames_rgb48le_sha256.is_some());
    let bodies = inputs.bodies();
    let masses = app::ember_masses(&bodies)?;
    let positions = get_positions(bodies, inputs.steps).positions;
    let request = EmberRequest {
        positions: &positions,
        dt: inputs.dt,
        masses,
        view: &inputs.view,
        frame_steps: &frame_steps,
        slow_factors: &inputs.frames.slow_factors,
        width: inputs.width,
        height: inputs.height,
        paper_seed: &paper_seed,
        config: &record.config,
        mode: match (expected_frames, slow_rendered) {
            (Some(_), true) => EmberMode::VideoAndSlow,
            (Some(_), false) => EmberMode::Video,
            (None, _) => EmberMode::StillOnly,
        },
    };
    let summary = render_ember(&request, &mut |_| Ok(()))?;
    let still = Some(outputs.still_rgb48le_sha256.as_str());
    let mut verified = digest_matches("still", still, Some(&summary.still_sha256));
    verified &= digest_matches("frames", expected_frames, summary.frames_sha256.as_deref());
    for (recorded, rendered) in outputs.slow_films.iter().zip(&summary.slow_films) {
        verified &= digest_matches(
            &format!("{}x", recorded.factor),
            recorded.frames_rgb48le_sha256.as_deref(),
            rendered.frames_sha256.as_deref(),
        );
    }
    // What else the certificate says of the render: a certificate whose digests match but whose
    // counts, derived quantities or statistics are not this render's misstates it. The frame
    // counts and the slow films' records stand for the outputs here.
    let counts = |outputs: &Value| {
        serde_json::json!({
            "frames_emitted": outputs["frames_emitted"],
            "slow_films": outputs["slow_films"],
        })
    };
    let rendered = serde_json::json!({
        "outputs": CertificateOutputs::from(&summary),
        "derived": CertificateDerived::from(&summary),
        "stats": summary.stats,
    });
    let sections = ["outputs", "derived", "stats"].map(|name| {
        let section = |document: &Value| match name {
            "outputs" => counts(&document[name]),
            _ => document[name].clone(),
        };
        (name, section(&recorded), section(&rendered))
    });
    for (name, recorded, rendered) in sections {
        match first_difference(name, &recorded, &rendered, "rendered") {
            None => println!("{name:<8} MATCH"),
            Some(difference) => {
                verified = false;
                println!("{name:<8} MISMATCH  {difference}");
            }
        }
    }
    println!(
        "machine  {} {} with {} threads, render {:.1}s",
        std::env::consts::ARCH,
        std::env::consts::OS,
        rayon::current_num_threads(),
        summary.timings.total_seconds
    );
    Ok(verified)
}

/// Prints the re-rendered digest `name` and whether it is the recorded one. `true` if it is, or
/// if the certificate records none (a film that was not rendered).
fn digest_matches(name: &str, recorded: Option<&str>, rendered: Option<&str>) -> bool {
    let Some(recorded) = recorded else {
        println!("{name:<8} (not recorded)");
        return true;
    };
    let matches = rendered == Some(recorded);
    println!(
        "{name:<8} {}  {}",
        if matches { "MATCH   " } else { "MISMATCH" },
        rendered.unwrap_or("(not rendered)"),
    );
    matches
}

/// Everything that stops this build from reproducing and verifying `record`, found before
/// the render: one entry per offending field, with the recorded value and this build's (or the
/// field it contradicts).
///
/// `recorded_config` is the certificate's `config` exactly as parsed from the file, so that a
/// field this build's [`EmberConfig`] would default or re-encode is caught.
fn reproducibility_problems(
    record: &EmberCertificate,
    recorded_config: &Value,
    paper_seed: &[u8],
    frame_steps: &[usize],
) -> Result<Vec<String>> {
    let inputs = &record.inputs;
    let mut problems = Vec::new();
    if record.algorithm != ALGORITHM_VERSION {
        problems.push(format!(
            "algorithm: the certificate was rendered by {:?}, this build implements {:?}",
            record.algorithm, ALGORITHM_VERSION
        ));
    }
    if inputs.integrator != INTEGRATOR {
        problems.push(format!(
            "inputs.integrator: the certificate records {:?}, this build integrates with {:?}",
            inputs.integrator, INTEGRATOR
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
    let paper = paper_seed_sha256(paper_seed);
    if inputs.paper_seed_sha256 != paper {
        problems.push(format!(
            "inputs.paper_seed_sha256: the certificate records {}, this build derives {paper} \
             from seed {} (app::ember_paper_seed)",
            inputs.paper_seed_sha256, inputs.seed
        ));
    }
    let schedule = schedule_sha256(frame_steps);
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
    let reread = serde_json::to_value(&record.config)?;
    if let Some(difference) = first_difference("config", recorded_config, &reread, "read back as") {
        problems.push(format!(
            "{difference} (the recorded configuration does not round-trip through this build's \
             EmberConfig)"
        ));
    }
    Ok(problems)
}

/// The first place where `recorded` and `other` differ, as `path: recorded X, <how> Y` (`how`
/// says what the other value is, e.g. `read back as`), or `None` if they are equal. Numbers
/// compare by value and bit pattern: `3` equals `3.0`, `-0.0` differs from `0.0`.
fn first_difference(path: &str, recorded: &Value, other: &Value, how: &str) -> Option<String> {
    match (recorded, other) {
        (Value::Object(a), Value::Object(b)) => a.keys().chain(b.keys()).find_map(|key| {
            let child = format!("{path}.{key}");
            match (a.get(key), b.get(key)) {
                (Some(x), Some(y)) => first_difference(&child, x, y, how),
                (x, y) => Some(difference(&child, x, y, how)),
            }
        }),
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => a
            .iter()
            .zip(b)
            .enumerate()
            .find_map(|(i, (x, y))| first_difference(&format!("{path}[{i}]"), x, y, how)),
        (Value::Number(a), Value::Number(b))
            if a.as_f64().map(f64::to_bits) == b.as_f64().map(f64::to_bits) =>
        {
            None
        }
        _ if recorded == other => None,
        _ => Some(difference(path, Some(recorded), Some(other), how)),
    }
}

/// `path: recorded X, <how> Y`, with `(absent)` for a missing value.
fn difference(path: &str, recorded: Option<&Value>, other: Option<&Value>, how: &str) -> String {
    let show =
        |value: Option<&Value>| value.map_or_else(|| "(absent)".to_owned(), Value::to_string);
    format!("{path}: recorded {}, {how} {}", show(recorded), show(other))
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
    slow_videos: bool,
    slow_factors: &'a [u32],
    frontal: bool,
}

/// An orbit to render: its initial conditions, and what else the file records of it.
struct Orbit {
    bodies: Vec<Body>,
    /// The package seed (for the paper texture).
    seed: Option<String>,
    /// The main edition's view of the orbit (certificates only).
    view: Option<View>,
}

/// The orbit of an ember certificate or of a `bodies_f64_bits` file.
fn read_orbit(path: &Path) -> Result<Orbit> {
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
        let seed = json.get("seed").and_then(Value::as_str).map(str::to_owned);
        return Ok(Orbit { bodies, seed, view: None });
    }
    // Only the orbit is read from a certificate, so one of any layout that records it (the view
    // since layout 4) will do; `verify` is the strict reader.
    let inputs = json.get("inputs").ok_or("neither `bodies_f64_bits` nor a certificate")?;
    let field = |name: &'static str| {
        move |e: serde_json::Error| format!("{}: inputs.{name}: {e}", path.display())
    };
    let bodies: Vec<BodyRecord> =
        serde_json::from_value(inputs["bodies"].clone()).map_err(field("bodies"))?;
    let view = inputs
        .get("view")
        .map(|view| serde_json::from_value(view.clone()).map_err(field("view")))
        .transpose()?;
    if view.is_none() {
        eprintln!(
            "note: {} records no view (a certificate before layout 4): the orbit is drawn \
             frontally, not as the main edition shows it",
            path.display()
        );
    }
    let seed = inputs.get("seed").and_then(Value::as_str).map(str::to_owned);
    Ok(Orbit { bodies: bodies.iter().map(BodyRecord::body).collect(), seed, view })
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

/// Renders an orbit for look development: the still, optionally every Nth scheduled frame as a
/// PNG, the browser video and the slow films, and a `summary.json` of the render.
fn render_orbit(args: &RenderArgs<'_>) -> Result<bool> {
    let orbit = read_orbit(args.bodies)?;
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
    let seed = args.seed.map(str::to_owned).or(orbit.seed).unwrap_or_else(|| "0x00".to_owned());
    let paper_seed = app::ember_paper_seed(&app::parse_seed(&seed)?);
    fs::create_dir_all(args.out.join("frames"))?;

    let masses = app::ember_masses(&orbit.bodies)?;
    let positions = get_positions(orbit.bodies, args.steps).positions;
    let view = match orbit.view {
        Some(view) if !args.frontal => view,
        _ => app::ember_frontal_view(&positions, width, height),
    };
    let frame_steps = app::ember_frame_schedule(args.steps);
    let request = EmberRequest {
        positions: &positions,
        dt: constants::DEFAULT_DT,
        masses,
        view: &view,
        frame_steps: &frame_steps,
        slow_factors: args.slow_factors,
        width,
        height,
        paper_seed: &paper_seed,
        config: &config,
        mode: if args.slow_videos {
            EmberMode::VideoAndSlow
        } else if args.video || args.frame_every > 0 {
            EmberMode::Video
        } else {
            EmberMode::StillOnly
        },
    };
    // Cheap, and it must pass before the encoders truncate any file in `--out`.
    plan_ember(&request)?;
    let frames_dir = args.out.join("frames");
    let save_frame = |frame: &EmberFrame<'_>| -> Result<()> {
        // Only scheduled frames are saved: the slow films' in-between frames have no index.
        let (Some(index), Some(step)) = (frame.index, frame.orbit_step) else { return Ok(()) };
        let is_last = index + 1 == frame.count;
        if args.frame_every > 0 && (index.is_multiple_of(args.frame_every) || is_last) {
            let image = image::ImageBuffer::from_raw(frame.width, frame.height, frame.rgb.to_vec())
                .ok_or("frame size")?;
            let path = frames_dir.join(format!("{index:05}_step{step:07}.png"));
            render::save_image_as_srgb_png_16bit(&image, &path.to_string_lossy())?;
        }
        Ok(())
    };

    // The package's web encode, so that a film here is as large as the published one.
    let film = |name: &str| VideoOutputSpec {
        output_file: args.out.join(name).to_string_lossy().into_owned(),
        options: VideoEncodingOptions {
            crf: app::EMBER_WEB_CRF,
            ..VideoEncodingOptions::web_compatible_srgb()
        },
    };
    // One encoder group per film that is asked for; a film without an encoder is rendered (its
    // frames are needed for the PNGs or the other films) but goes nowhere.
    let normal = if args.video { vec![film("ember.mp4")] } else { Vec::new() };
    let slow: Vec<Vec<VideoOutputSpec>> = if args.slow_videos {
        args.slow_factors.iter().map(|factor| vec![film(&format!("ember_{factor}x.mp4"))]).collect()
    } else {
        Vec::new()
    };
    let groups: Vec<&[VideoOutputSpec]> = std::iter::once(normal.as_slice())
        .chain(slow.iter().map(Vec::as_slice))
        .filter(|group| !group.is_empty())
        .collect();
    let sink_error = |e: Box<dyn Error>| EmberError::Sink(e.to_string());

    let summary: EmberSummary = if groups.is_empty() {
        render_ember(&request, &mut |frame| save_frame(frame).map_err(sink_error))?
    } else {
        let mut result = None;
        let encoded = create_video_groups_from_frames(
            width,
            height,
            constants::DEFAULT_VIDEO_FPS,
            &groups,
            |writers| {
                // The writers come in group order: the normal film's (if asked for), then one per
                // slow film in the order of the frames' `slow_indices`.
                let (mut normal_film, slow_films) = match writers.split_first_mut() {
                    Some((first, rest)) if args.video => (Some(first), rest),
                    _ => (None, writers),
                };
                let rendered = render_ember(&request, &mut |frame| {
                    save_frame(frame).map_err(sink_error)?;
                    let write = |film: &mut GroupWriter| {
                        film.write_all(frame.rgb48le).map_err(|e| sink_error(e.into()))
                    };
                    if let (Some(film), Some(_)) = (normal_film.as_deref_mut(), frame.index) {
                        write(film)?;
                    }
                    for (film, index) in slow_films.iter_mut().zip(frame.slow_indices) {
                        if index.is_some() {
                            write(film)?;
                        }
                    }
                    Ok(())
                });
                let status = rendered.as_ref().map(|_| ()).map_err(|e| e.to_string().into());
                result = Some(rendered);
                status
            },
        );
        // A failed render is what stopped the encoders, so it is the error to report.
        match result {
            Some(Err(error)) => return Err(error.into()),
            Some(Ok(summary)) => encoded.map(|()| summary)?,
            None => return Err(encoded.err().map_or_else(|| "no render".into(), Into::into)),
        }
    };

    let still = args.out.join("ember.png");
    render::save_image_as_srgb_png_16bit(&summary.still_image(), &still.to_string_lossy())?;
    let report = serde_json::json!({
        "duration": summary.duration,
        "valve_time": summary.valve_time,
        "fluid_grid": summary.fluid_grid,
        "ink_grid": summary.ink_grid,
        "view": view,
        "frames": frame_steps.len(),
        "slow_films": summary
            .slow_films
            .iter()
            .map(|film| serde_json::json!({
                "factor": film.factor,
                "first_frame": film.first_frame,
                "frames": film.frames_emitted,
                "frames_sha256": film.frames_sha256,
            }))
            .collect::<Vec<_>>(),
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
