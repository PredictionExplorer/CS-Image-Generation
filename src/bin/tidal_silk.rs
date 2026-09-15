//! Independent CPU-only Tidal Silk generation, baking, rendering and encoding.
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;
use three_body_problem::silk::{
    SilkResult, cache, comparison, guides, normal, orbit, render, simulation,
};

#[derive(Parser)]
#[command(
    about = "Tidal Silk — three-body motion, continuous fabric, CPU light transport",
    version
)]
struct Args {
    /// CPU worker count for independent rendering work.
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Action,
}

#[derive(Subcommand)]
enum Action {
    /// Replay the normal accumulated-light movie from a recorded selected orbit.
    Normal {
        #[arg(long)]
        orbit: PathBuf,
        #[arg(long)]
        generation_record: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        width: Option<u32>,
        #[arg(long)]
        height: Option<u32>,
        #[arg(long)]
        fast_encode: bool,
    },
    /// Export the three physical attachment positions through the silk camera.
    Guides {
        #[arg(long)]
        orbit: PathBuf,
        #[arg(long)]
        bake: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
    /// Add body-position markers to a separate copy of a PNG sequence.
    Overlay {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        markers: PathBuf,
    },
    /// Generate or reconstruct original physical motion once.
    Orbit {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        seed: Option<String>,
        #[arg(long)]
        sims: Option<usize>,
        #[arg(long)]
        steps: Option<usize>,
        /// Replay the recorded selected candidate without running the search.
        #[arg(long)]
        generation_record: Option<PathBuf>,
    },
    /// Simulate cloth and save a reusable full-precision geometry cache.
    Bake {
        #[arg(long)]
        orbit: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        frames: Option<u32>,
        #[arg(long)]
        subdivisions: Option<usize>,
    },
    /// Render 16-bit PNG frames; completed compatible frames can be resumed.
    Render {
        #[arg(long)]
        bake: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        width: Option<u32>,
        #[arg(long)]
        height: Option<u32>,
        #[arg(long)]
        samples: Option<u32>,
        #[arg(long)]
        palette: Option<String>,
        /// Render one particular source frame.
        #[arg(long,conflicts_with_all=["start","end"])]
        frame: Option<usize>,
        #[arg(long)]
        start: Option<usize>,
        /// Exclusive end frame.
        #[arg(long)]
        end: Option<usize>,
        /// Frame stride for inexpensive full-motion previews.
        #[arg(long, default_value_t = 1)]
        every: usize,
        /// Replace an earlier rendering in this output directory.
        #[arg(long)]
        overwrite: bool,
    },
    /// Encode a rendered sequence with the existing `FFmpeg` dependency.
    Encode {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        hq: bool,
        #[arg(long, default_value_t = 16)]
        encoder_threads: usize,
    },
    /// Print cache dimensions, source recipe and simulation diagnostics.
    Inspect {
        #[arg(long)]
        bake: PathBuf,
    },
    /// Compare decoded RGB16 image values for repeatability checks.
    Compare {
        #[arg(long)]
        first: PathBuf,
        #[arg(long)]
        second: PathBuf,
    },
}

#[derive(Clone, Serialize, Deserialize)]
struct RenderManifest {
    schema_version: u32,
    bake_sha256: String,
    binary_sha256: String,
    config: render::RenderConfig,
    fps: u32,
    frames: Vec<usize>,
    complete: bool,
}

#[derive(Serialize, Deserialize)]
struct FrameReceipt {
    recipe_sha256: String,
    png_sha256: String,
}

fn sidecar(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".json");
    PathBuf::from(name)
}

fn read_config<T: DeserializeOwned + Default>(path: Option<&Path>) -> SilkResult<T> {
    match path {
        Some(path) => Ok(serde_json::from_slice(&fs::read(path)?)?),
        None => Ok(T::default()),
    }
}

fn write_json(path: &Path, value: &impl Serialize) -> SilkResult<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.partial");
    fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn frame_path(directory: &Path, frame: usize) -> PathBuf {
    directory.join(format!("frame_{frame:06}.png"))
}

fn main() -> SilkResult<()> {
    let args = Args::parse();
    if let Some(threads) = args.threads {
        if threads == 0 {
            return Err("--threads must be positive".into());
        }
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global()?;
    }
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .init();
    let started = Instant::now();
    match args.command {
        Action::Normal { orbit, generation_record, output, config, width, height, fast_encode } => {
            let orbit = cache::read_orbit(&orbit)?;
            let record = serde_json::from_slice(&fs::read(generation_record)?)?;
            let mut config: normal::NormalConfig = read_config(config.as_deref())?;
            if let Some(width) = width {
                config.width = width;
            }
            if let Some(height) = height {
                config.height = height;
            }
            if fast_encode {
                config.fast_encode = true;
            }
            let report = normal::render_from_record(&orbit, &record, &config, &output)?;
            eprintln!(
                "Normal movie: {} frames at {} fps, {} seconds",
                report["frame_count"], report["fps"], report["duration_seconds"]
            );
        }
        Action::Guides { orbit, bake, config, output } => {
            let orbit = cache::read_orbit(&orbit)?;
            let bake = cache::read_bake(&bake)?;
            let config: render::RenderConfig = read_config(config.as_deref())?;
            write_json(&output, &guides::generate_silk_markers(&orbit, &bake, &config)?)?;
        }
        Action::Overlay { input, output, markers } => {
            let markers = serde_json::from_slice(&fs::read(markers)?)?;
            guides::overlay_frames(&input, &output, &markers)?;
        }
        Action::Orbit { output, config, seed, sims, steps, generation_record } => {
            let mut config: orbit::OrbitConfig = read_config(config.as_deref())?;
            if let Some(seed) = seed {
                config.seed = seed;
            }
            if let Some(sims) = sims {
                config.sims = sims;
            }
            if let Some(steps) = steps {
                config.steps = steps;
            }
            if generation_record.is_some() {
                config.generation_record = generation_record;
            }
            let data = orbit::generate(&config)?;
            cache::write_orbit(&output, &data)?;
            write_json(
                &sidecar(&output),
                &serde_json::json!({"schema_version":1,"seed":data.seed,"dt":data.dt,"samples":data.samples.len(),"masses":data.masses,"sha256":cache::file_hash(&output)?,"provenance":data.provenance}),
            )?;
            eprintln!("Orbit cache: {} ({} samples)", output.display(), data.samples.len());
        }
        Action::Bake { orbit, output, config, frames, subdivisions } => {
            let mut config: simulation::SimulationConfig = read_config(config.as_deref())?;
            if let Some(frames) = frames {
                config.frames = frames;
            }
            if let Some(subdivisions) = subdivisions {
                config.subdivisions = subdivisions;
            }
            let data = cache::read_orbit(&orbit)?;
            let mut bake = simulation::bake(&data, &config)?;
            bake.recipe["orbit_cache_sha256"] = cache::file_hash(&orbit)?.into();
            cache::write_bake(&output, &bake)?;
            write_json(
                &sidecar(&output),
                &serde_json::json!({"schema_version":1,"vertices":bake.mesh.positions.len(),"triangles":bake.mesh.triangles.len(),"frames":bake.frames.len(),"fps":bake.fps,"sha256":cache::file_hash(&output)?,"recipe":bake.recipe,"stats":bake.stats}),
            )?;
            eprintln!(
                "Cloth cache: {} ({} vertices, {} frames)",
                output.display(),
                bake.mesh.positions.len(),
                bake.frames.len()
            );
        }
        Action::Render {
            bake,
            output,
            config,
            width,
            height,
            samples,
            palette,
            frame,
            start,
            end,
            every,
            overwrite,
        } => {
            let mut config: render::RenderConfig = read_config(config.as_deref())?;
            if let Some(width) = width {
                config.width = width;
            }
            if let Some(height) = height {
                config.height = height;
            }
            if let Some(samples) = samples {
                config.samples_per_pixel = samples;
            }
            if let Some(palette) = palette {
                config.palette = palette;
            }
            if every == 0 {
                return Err("--every must be positive".into());
            }
            let data = cache::read_bake(&bake)?;
            let frames = if let Some(frame) = frame {
                if frame >= data.frames.len() {
                    return Err("Selected frame is outside the cloth cache".into());
                }
                vec![frame]
            } else {
                let start = start.unwrap_or(0);
                let end = end.unwrap_or(data.frames.len());
                if start >= end || end > data.frames.len() {
                    return Err("Invalid frame interval".into());
                }
                (start..end).step_by(every).collect()
            };
            fs::create_dir_all(&output)?;
            let mut manifest = RenderManifest {
                schema_version: 1,
                bake_sha256: cache::file_hash(&bake)?,
                binary_sha256: cache::file_hash(&std::env::current_exe()?)?,
                config: config.clone(),
                fps: data.fps,
                frames,
                complete: false,
            };
            let manifest_path = output.join("render.json");
            if manifest_path.exists() && !overwrite {
                let mut existing: RenderManifest =
                    serde_json::from_slice(&fs::read(&manifest_path)?)?;
                existing.complete = false;
                if serde_json::to_value(&existing)? != serde_json::to_value(&manifest)? {
                    return Err("This directory contains a different render recipe or executable; choose a new output directory or --overwrite".into());
                }
            } else if !overwrite && manifest.frames.iter().any(|&f| frame_path(&output, f).exists())
            {
                return Err("Output images exist without a matching manifest; choose a new directory or --overwrite".into());
            }
            let recipe_hash = hex::encode(Sha256::digest(serde_json::to_vec(&manifest)?));
            write_json(&manifest_path, &manifest)?;
            for (index, &frame) in manifest.frames.iter().enumerate() {
                let path = frame_path(&output, frame);
                if path.exists() && !overwrite && sidecar(&path).exists() {
                    let receipt: FrameReceipt = serde_json::from_slice(&fs::read(sidecar(&path))?)?;
                    if receipt.recipe_sha256 == recipe_hash
                        && receipt.png_sha256 == cache::file_hash(&path)?
                    {
                        let existing = image::open(&path)?;
                        if existing.width() == config.width && existing.height() == config.height {
                            eprintln!("Reusing frame {frame}");
                            continue;
                        }
                    }
                }
                let frame_start = Instant::now();
                let image = render::render_frame16(&data, frame, &config)?;
                let temporary = path.with_extension("png.partial");
                image.save_with_format(&temporary, image::ImageFormat::Png)?;
                fs::rename(temporary, &path)?;
                write_json(
                    &sidecar(&path),
                    &FrameReceipt {
                        recipe_sha256: recipe_hash.clone(),
                        png_sha256: cache::file_hash(&path)?,
                    },
                )?;
                eprintln!(
                    "Frame {frame} ({}/{}) rendered in {:.2}s",
                    index + 1,
                    manifest.frames.len(),
                    frame_start.elapsed().as_secs_f64()
                );
            }
            manifest.complete = true;
            write_json(&manifest_path, &manifest)?;
        }
        Action::Encode { input, output, hq, encoder_threads } => {
            encode(&input, &output, hq, encoder_threads)?;
        }
        Action::Inspect { bake } => {
            let data = cache::read_bake(&bake)?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &serde_json::json!({"vertices":data.mesh.positions.len(),"triangles":data.mesh.triangles.len(),"frames":data.frames.len(),"fps":data.fps,"max_stretch":data.stats.iter().map(|s|s.max_stretch).fold(0.0,f64::max),"max_pin_error":data.stats.iter().map(|s|s.pin_error).fold(0.0,f64::max),"contacts":data.stats.iter().map(|s|s.contacts).sum::<usize>(),"recipe":data.recipe})
                )?
            );
        }
        Action::Compare { first, second } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&comparison::compare_images(&first, &second)?)?
            );
        }
    }
    eprintln!("Completed in {:.2}s", started.elapsed().as_secs_f64());
    Ok(())
}

fn encode(input: &Path, output: &Path, hq: bool, threads: usize) -> SilkResult<()> {
    let manifest: RenderManifest = serde_json::from_slice(&fs::read(input.join("render.json"))?)?;
    if !manifest.complete || manifest.frames.len() < 2 || threads == 0 {
        return Err(
            "Encoding requires a complete multi-frame render and positive thread count".into()
        );
    }
    let stride =
        manifest.frames[1].checked_sub(manifest.frames[0]).ok_or("Frame indices must increase")?;
    if stride == 0 || manifest.frames.windows(2).any(|w| w[1].checked_sub(w[0]) != Some(stride)) {
        return Err("Frame sequence must be strictly increasing at a fixed interval".into());
    }
    if !manifest.config.width.is_multiple_of(2) || !manifest.config.height.is_multiple_of(2) {
        return Err("Video requires even pixel dimensions".into());
    }
    let mut normalized_manifest = manifest.clone();
    normalized_manifest.complete = false;
    let recipe_hash = hex::encode(Sha256::digest(serde_json::to_vec(&normalized_manifest)?));
    for &frame in &manifest.frames {
        let path = frame_path(input, frame);
        if !path.is_file() {
            return Err(format!("Missing source frame {frame}").into());
        }
        let receipt: FrameReceipt = serde_json::from_slice(&fs::read(sidecar(&path))?)?;
        if receipt.recipe_sha256 != recipe_hash || receipt.png_sha256 != cache::file_hash(&path)? {
            return Err(
                format!("Frame {frame} failed render recipe or integrity verification").into()
            );
        }
    }
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let temporary = output.with_extension("partial.mp4");
    let mut command = Command::new("ffmpeg");
    command.args([
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb48le",
        "-video_size",
        &format!("{}x{}", manifest.config.width, manifest.config.height),
        "-framerate",
        &format!("{}/{}", manifest.fps, stride),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        if hq { "libx265" } else { "libx264" },
        "-preset",
        "slow",
        "-crf",
        if hq { "16" } else { "17" },
        "-threads",
        &threads.to_string(),
        "-vf",
        "scale=out_color_matrix=bt709:out_range=tv:flags=bilinear+accurate_rnd+bitexact",
        "-sws_dither",
        "none",
        "-pix_fmt",
        if hq { "yuv420p10le" } else { "yuv420p" },
        "-color_primaries",
        "bt709",
        "-color_trc",
        "iec61966-2-1",
        "-colorspace",
        "bt709",
        "-color_range",
        "tv",
        "-flags",
        "+bitexact",
        "-fflags",
        "+bitexact",
        "-map_metadata",
        "-1",
        "-movflags",
        "+faststart",
    ]);
    if hq {
        command.args([
            "-tag:v",
            "hvc1",
            "-x265-params",
            &format!("pools={threads}:frame-threads=2:log-level=error"),
        ]);
    } else {
        command.args(["-x264-params", "cpu-independent=1"]);
    }
    command.arg(&temporary).stdin(Stdio::piped()).stdout(Stdio::null());
    let mut child = command.spawn()?;
    let result = (|| -> SilkResult<()> {
        let mut stdin = child.stdin.take().ok_or("Could not open encoder input")?;
        for &frame in &manifest.frames {
            let image = image::open(frame_path(input, frame))?.to_rgb16();
            if image.width() != manifest.config.width || image.height() != manifest.config.height {
                return Err(format!("Frame {frame} dimensions do not match manifest").into());
            }
            let bytes: Vec<u8> = image.as_raw().iter().flat_map(|v| v.to_le_bytes()).collect();
            stdin.write_all(&bytes)?;
        }
        drop(stdin);
        Ok(())
    })();
    if result.is_err() {
        let _ = child.kill();
        let _ = child.wait();
        let _ = fs::remove_file(&temporary);
        return result;
    }
    if !child.wait()?.success() {
        let _ = fs::remove_file(&temporary);
        return Err("FFmpeg failed to encode the film".into());
    }
    fs::rename(&temporary, output)?;
    let encoder_version = Command::new("ffmpeg").arg("-version").output()?;
    let encoder_version = String::from_utf8_lossy(&encoder_version.stdout)
        .lines()
        .next()
        .unwrap_or("unavailable")
        .to_owned();
    write_json(
        &sidecar(output),
        &serde_json::json!({"schema_version":1,"source_manifest":manifest,"codec":if hq{"hevc"}else{"h264"},"encoder_threads":threads,"encoder_version":encoder_version,"cpu_independent":!hq,"color_conversion":"bilinear+accurate_rnd+bitexact; no dither; BT.709 matrix; sRGB transfer","sha256":cache::file_hash(output)?}),
    )?;
    eprintln!("Film: {}", output.display());
    Ok(())
}
