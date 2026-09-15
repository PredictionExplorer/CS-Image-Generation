//! Render and archive six artistic interpretations of a frozen physical orbit.
use clap::{Parser, Subcommand};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{ErrorKind, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Instant,
};
use three_body_problem::{
    atelier::{
        Camera, OrbitSeries, RenderConfig, Scene, SilkResult, V3, aurora, calligraphy, eclipse,
        engraving, light, loom, render,
    },
    silk::cache,
};

#[derive(Parser)]
#[command(about = "High-resolution CPU art from one recorded three-body motion")]
struct Args {
    /// CPU workers; pixel results do not depend on worker count.
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Action,
}

#[derive(Subcommand)]
enum Action {
    /// Bound full-source Eclipse motion and the fixed camera crop.
    AuditEclipse {
        #[arg(long)]
        orbit: PathBuf,
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Write the fully specified default art recipe.
    Config {
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "default")]
        preset: String,
    },
    /// Render original RGB16 PNG frames, with compatible interruption recovery.
    Render {
        #[arg(long)]
        orbit: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, conflicts_with_all = ["start", "end"])]
        frame: Option<usize>,
        #[arg(long)]
        start: Option<usize>,
        /// Exclusive end frame.
        #[arg(long)]
        end: Option<usize>,
        #[arg(long, default_value_t = 1)]
        every: usize,
        #[arg(long)]
        width: Option<u32>,
        #[arg(long)]
        height: Option<u32>,
        #[arg(long)]
        aa: Option<u32>,
        #[arg(long)]
        overwrite: bool,
        /// Save a separately labelled progressive preview while expensive exposures finish.
        #[arg(long)]
        progress: bool,
    },
    /// Encode a completed, verified frame sequence with the existing `FFmpeg`.
    Encode {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        /// Encode an HEVC 10-bit master instead of browser H.264.
        #[arg(long)]
        hq: bool,
        #[arg(long, default_value_t = 16)]
        encoder_threads: usize,
        #[arg(long)]
        overwrite: bool,
    },
    /// Verify parallel render chunks and assemble one complete frame sequence.
    Assemble {
        /// Completed chunk directory; repeat for every part of the film.
        #[arg(long, required = true)]
        input: Vec<PathBuf>,
        /// Separate destination, or a compatible interrupted assembly.
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct StudyConfig {
    kind: String,
    frames: usize,
    fps: u32,
    temporal_samples: usize,
    shutter_fraction: f64,
    prelude_fraction: f64,
    camera: Camera,
    render: RenderConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    calligraphy: Option<calligraphy::CalligraphyConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    loom: Option<loom::LoomConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    aurora: Option<aurora::AuroraConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    light: Option<light::LightConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    engraving: Option<engraving::EngravingConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    eclipse: Option<eclipse::EclipseConfig>,
}

impl Default for StudyConfig {
    fn default() -> Self {
        Self {
            kind: "calligraphy".into(),
            frames: 1802,
            fps: 60,
            temporal_samples: 4,
            shutter_fraction: 0.5,
            prelude_fraction: 0.36,
            camera: Camera::default(),
            render: RenderConfig::default(),
            calligraphy: Some(calligraphy::CalligraphyConfig::default()),
            loom: None,
            aurora: None,
            light: None,
            engraving: None,
            eclipse: None,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct Manifest {
    schema_version: u32,
    seed: String,
    orbit_sha256: String,
    executable_sha256: String,
    source_history_start_fraction: f64,
    config: StudyConfig,
    rendered_frames: Vec<usize>,
    complete: bool,
}

#[derive(Clone, Serialize, Deserialize)]
struct Receipt {
    recipe_sha256: String,
    png_sha256: String,
    source_fraction: f64,
    shutter_start_fraction: f64,
    shutter_end_fraction: f64,
    vertices: usize,
    triangles: usize,
    strands: usize,
    seconds: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    light_samples: Option<Vec<light::LightDiagnostics>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    engraving_samples: Option<Vec<engraving::EngravingDiagnostics>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    eclipse_samples: Option<Vec<eclipse::EclipseDiagnostics>>,
}

fn json(path: &Path, value: &impl Serialize) -> SilkResult<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.partial");
    fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn sidecar(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".json");
    PathBuf::from(name)
}

fn frame_path(directory: &Path, frame: usize) -> PathBuf {
    directory.join(format!("frame_{frame:06}.png"))
}

fn write_progress_preview(
    directory: &Path,
    frame: usize,
    completed: usize,
    config: &StudyConfig,
    hash: &str,
    accumulated: &[V3],
) -> SilkResult<()> {
    let path = directory.join(format!(".progress-{frame:06}.png"));
    let temporary = path.with_extension("png.partial");
    let scale = config.temporal_samples as f64 / completed as f64;
    let linear = accumulated.par_iter().map(|pixel| *pixel * scale).collect();
    render::finish_linear(linear, &config.render)?
        .save_with_format(&temporary, image::ImageFormat::Png)?;
    fs::rename(temporary, &path)?;
    json(
        &sidecar(&path),
        &serde_json::json!({
            "development_preview":true,"final_frame":false,"frame":frame,
            "completed_exposure_samples":completed,"total_exposure_samples":config.temporal_samples,
            "nominal_source_fraction":frame as f64/(config.frames-1) as f64,
            "latest_source_fraction":expected_sample_time(config,frame,completed-1),
            "recipe_sha256":hash,"png_sha256":cache::file_hash(&path)?,
            "note":"Progressive exposure preview. Canonical frame and receipt are published separately after all samples finish."
        }),
    )?;
    Ok(())
}

fn recipe_hash(manifest: &Manifest) -> SilkResult<String> {
    let mut normalized = manifest.clone();
    normalized.complete = false;
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(&normalized)?)))
}

fn art_scene(source: &OrbitSeries, time: f64, config: &StudyConfig) -> SilkResult<Scene> {
    match config.kind.as_str() {
        "calligraphy" => calligraphy::scene(
            source,
            time,
            config.calligraphy.as_ref().ok_or("Missing Calligraphy parameters")?,
        ),
        "loom" => loom::scene(source, time, config.loom.as_ref().ok_or("Missing Loom parameters")?),
        "aurora" => {
            aurora::scene(source, time, config.aurora.as_ref().ok_or("Missing Aurora parameters")?)
        }
        _ => Err(format!("Unsupported study: {}", config.kind).into()),
    }
}

struct ArtFrame {
    pixels: Vec<V3>,
    geometry: [usize; 3],
    light: Option<light::LightDiagnostics>,
    engraving: Option<engraving::EngravingDiagnostics>,
    eclipse: Option<eclipse::EclipseDiagnostics>,
}

fn art_frame(
    source: &OrbitSeries,
    time: f64,
    raw_cell_interval: [f64; 2],
    config: &StudyConfig,
) -> SilkResult<ArtFrame> {
    if config.kind == "eclipse" {
        let frame = eclipse::render_linear(
            source,
            time,
            config.eclipse.as_ref().ok_or("Missing Eclipse Garden parameters")?,
            &config.camera,
            &config.render,
        )?;
        frame.diagnostics.validate()?;
        Ok(ArtFrame {
            pixels: frame.pixels,
            geometry: [0; 3],
            light: None,
            engraving: None,
            eclipse: Some(frame.diagnostics),
        })
    } else if config.kind == "light" {
        let frame = light::render_linear(
            source,
            time,
            config.light.as_ref().ok_or("Missing Light Cast parameters")?,
            &config.camera,
            &config.render,
        )?;
        frame.diagnostics.validate()?;
        Ok(ArtFrame {
            pixels: frame.pixels,
            geometry: [0; 3],
            light: Some(frame.diagnostics),
            engraving: None,
            eclipse: None,
        })
    } else if config.kind == "engraving" {
        let frame = engraving::render_linear(
            source,
            time,
            raw_cell_interval,
            config.engraving.as_ref().ok_or("Missing Engraving parameters")?,
            &config.camera,
            &config.render,
        )?;
        frame.diagnostics.validate()?;
        Ok(ArtFrame {
            pixels: frame.pixels,
            geometry: [0; 3],
            light: None,
            engraving: Some(frame.diagnostics),
            eclipse: None,
        })
    } else {
        let scene = art_scene(source, time, config)?;
        Ok(ArtFrame {
            pixels: render::render_linear(&scene, &config.camera, &config.render)?,
            geometry: [scene.vertices.len(), scene.triangles.len(), scene.strands.len()],
            light: None,
            engraving: None,
            eclipse: None,
        })
    }
}

fn main() -> SilkResult<()> {
    let args = Args::parse();
    if let Some(threads) = args.threads {
        if threads == 0 {
            return Err("--threads must be positive".into());
        }
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global()?;
    }
    let started = Instant::now();
    match args.command {
        Action::AuditEclipse { orbit, config, output } => {
            let c: StudyConfig = serde_json::from_slice(&fs::read(&config)?)?;
            if c.kind != "eclipse" {
                return Err("Motion audit requires an Eclipse recipe".into());
            }
            let data = cache::read_orbit(&orbit)?;
            let source = OrbitSeries::new(&data)?;
            let audit = eclipse::audit_motion(
                &source,
                c.eclipse.as_ref().ok_or("Missing Eclipse parameters")?,
                &c.camera,
                &c.render,
                c.frames,
                c.shutter_fraction,
                c.temporal_samples,
            )?;
            json(
                &output,
                &serde_json::json!({"source_seed":data.seed,"orbit_sha256":cache::file_hash(&orbit)?,"config_sha256":cache::file_hash(&config)?,"audit":audit}),
            )?;
            eprintln!("Motion audit: {}", output.display());
        }
        Action::Config { output, preset } => {
            let config = match preset.as_str() {
                "eclipse" => StudyConfig {
                    kind: "eclipse".into(),
                    temporal_samples: 128,
                    calligraphy: None,
                    eclipse: Some(eclipse::EclipseConfig::default()),
                    prelude_fraction: 0.0,
                    camera: Camera {
                        position: V3::new(0.0, 0.0, 12.0),
                        target: V3::ZERO,
                        orthographic_height: 7.4,
                        ..Camera::default()
                    },
                    render: RenderConfig {
                        aa: 3,
                        exposure: 0.0,
                        background: V3::new(0.0005, 0.00035, 0.00065),
                        key_strength: 0.0,
                        rim_strength: 0.0,
                        fill_strength: 0.0,
                        bloom_strength: 0.0,
                        ..RenderConfig::default()
                    },
                    ..StudyConfig::default()
                },
                "engraving" => StudyConfig {
                    kind: "engraving".into(),
                    temporal_samples: 16,
                    calligraphy: None,
                    engraving: Some(engraving::EngravingConfig::default()),
                    prelude_fraction: 0.0,
                    camera: Camera {
                        position: V3::new(0.0, 0.0, 12.0),
                        target: V3::ZERO,
                        orthographic_height: 6.8,
                        ..Camera::default()
                    },
                    render: RenderConfig {
                        aa: 2,
                        exposure: 0.0,
                        background: V3::new(0.0030, 0.0018, 0.0048),
                        key_strength: 0.0,
                        rim_strength: 0.0,
                        fill_strength: 0.0,
                        bloom_strength: 0.0,
                        ..RenderConfig::default()
                    },
                    ..StudyConfig::default()
                },
                "light" | "light-ivory" | "light-dark" => {
                    let dark = preset == "light-dark";
                    StudyConfig {
                        kind: "light".into(),
                        temporal_samples: 16,
                        calligraphy: None,
                        light: Some(light::LightConfig {
                            display: if dark {
                                light::LightDisplay::DarkGain
                            } else {
                                light::LightDisplay::Ivory
                            },
                            ..light::LightConfig::default()
                        }),
                        prelude_fraction: 0.0,
                        camera: Camera {
                            position: V3::new(0.0, 0.0, 12.0),
                            target: V3::ZERO,
                            orthographic_height: 6.0,
                            ..Camera::default()
                        },
                        render: RenderConfig {
                            exposure: if dark { 0.75 } else { 1.15 },
                            background: V3::new(0.002, 0.0017, 0.003),
                            key_strength: 0.0,
                            rim_strength: 0.0,
                            fill_strength: 0.0,
                            bloom_strength: if dark { 0.06 } else { 0.015 },
                            ..RenderConfig::default()
                        },
                        ..StudyConfig::default()
                    }
                }
                "aurora" => StudyConfig {
                    kind: "aurora".into(),
                    calligraphy: None,
                    aurora: Some(aurora::AuroraConfig::default()),
                    camera: Camera {
                        position: three_body_problem::atelier::V3::new(2.0, 1.4, 12.0),
                        target: three_body_problem::atelier::V3::new(0.0, 1.3, 0.0),
                        orthographic_height: 7.0,
                        ..Camera::default()
                    },
                    render: RenderConfig {
                        exposure: 0.7,
                        background: three_body_problem::atelier::V3::new(0.0008, 0.0014, 0.003),
                        key_strength: 0.035,
                        rim_strength: 0.025,
                        fill_strength: 0.015,
                        bloom_strength: 0.09,
                        ..RenderConfig::default()
                    },
                    ..StudyConfig::default()
                },
                "loom" | "loom-panel" | "loom-spindle" => {
                    let parameters = loom::LoomConfig {
                        panels: if preset == "loom-panel" {
                            [true, false, false]
                        } else {
                            [true; 3]
                        },
                        profile: if preset == "loom-spindle" {
                            loom::LoomProfile::Spindle
                        } else {
                            loom::LoomProfile::Conch
                        },
                        ..loom::LoomConfig::default()
                    };
                    StudyConfig {
                        kind: "loom".into(),
                        calligraphy: None,
                        loom: Some(parameters),
                        camera: Camera {
                            position: three_body_problem::atelier::V3::new(3.4, 2.0, 10.0),
                            orthographic_height: 4.8,
                            ..Camera::default()
                        },
                        ..StudyConfig::default()
                    }
                }
                "default" | "gossamer" | "fans" => StudyConfig {
                    calligraphy: Some(match preset.as_str() {
                        "gossamer" => calligraphy::CalligraphyConfig::gossamer(),
                        "fans" => calligraphy::CalligraphyConfig::silk_fans(),
                        _ => calligraphy::CalligraphyConfig::default(),
                    }),
                    ..StudyConfig::default()
                },
                _ => return Err("Unknown art preset".into()),
            };
            json(&output, &config)?;
        }
        Action::Render {
            orbit,
            config,
            output,
            frame,
            start,
            end,
            every,
            width,
            height,
            aa,
            overwrite,
            progress,
        } => {
            let mut config: StudyConfig = if let Some(path) = config {
                serde_json::from_slice(&fs::read(path)?)?
            } else {
                StudyConfig::default()
            };
            if let Some(width) = width {
                config.render.width = width;
            }
            if let Some(height) = height {
                config.render.height = height;
            }
            if let Some(aa) = aa {
                config.render.aa = aa;
            }
            match config.kind.as_str() {
                "eclipse" => {
                    config.eclipse.get_or_insert_with(eclipse::EclipseConfig::default);
                    config.calligraphy = None;
                    config.loom = None;
                    config.aurora = None;
                    config.light = None;
                    config.engraving = None;
                }
                "calligraphy" => {
                    config.calligraphy.get_or_insert_with(calligraphy::CalligraphyConfig::default);
                    config.loom = None;
                    config.aurora = None;
                    config.light = None;
                }
                "loom" => {
                    config.loom.get_or_insert_with(loom::LoomConfig::default);
                    config.calligraphy = None;
                    config.aurora = None;
                    config.light = None;
                }
                "aurora" => {
                    config.aurora.get_or_insert_with(aurora::AuroraConfig::default);
                    config.calligraphy = None;
                    config.loom = None;
                    config.light = None;
                }
                "light" => {
                    config.light.get_or_insert_with(light::LightConfig::default);
                    config.calligraphy = None;
                    config.loom = None;
                    config.aurora = None;
                }
                "engraving" => {
                    config.engraving.get_or_insert_with(engraving::EngravingConfig::default);
                    config.calligraphy = None;
                    config.loom = None;
                    config.aurora = None;
                    config.light = None;
                }
                _ => return Err(format!("Unsupported study: {}", config.kind).into()),
            }
            if config.kind != "eclipse" {
                config.eclipse = None;
            }
            if config.kind != "engraving" {
                config.engraving = None;
            }
            if config.frames < 2 || config.fps == 0 || every == 0 || config.temporal_samples == 0 {
                return Err("Require at least two frames, positive fps and positive stride".into());
            }
            if !config.shutter_fraction.is_finite()
                || !(0.0..=1.0).contains(&config.shutter_fraction)
            {
                return Err("Shutter fraction must be finite and between zero and one".into());
            }
            if !config.prelude_fraction.is_finite() || config.prelude_fraction < 0.0 {
                return Err("Prelude fraction must be finite and nonnegative".into());
            }
            let frames: Vec<usize> = if let Some(frame) = frame {
                if frame >= config.frames {
                    return Err("Frame is outside the film".into());
                }
                vec![frame]
            } else {
                let first = start.unwrap_or(0);
                let last = end.unwrap_or(config.frames);
                if first >= last || last > config.frames {
                    return Err("Invalid frame interval".into());
                }
                (first..last).step_by(every).collect()
            };
            let data = cache::read_orbit(&orbit)?;
            let source = if config.prelude_fraction > 0.0 {
                OrbitSeries::with_prelude(&data, config.prelude_fraction)?
            } else {
                OrbitSeries::new(&data)?
            };
            let mut manifest = Manifest {
                schema_version: 1,
                seed: data.seed.clone(),
                orbit_sha256: cache::file_hash(&orbit)?,
                executable_sha256: cache::file_hash(&std::env::current_exe()?)?,
                source_history_start_fraction: source.history_start_fraction(),
                config,
                rendered_frames: frames,
                complete: false,
            };
            fs::create_dir_all(&output)?;
            let manifest_path = output.join("render.json");
            let hash = recipe_hash(&manifest)?;
            if manifest_path.exists() && !overwrite {
                let previous: Manifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
                if recipe_hash(&previous)? != hash {
                    return Err("Output has a different recipe or executable; use a new directory or --overwrite".into());
                }
            } else if !overwrite
                && manifest.rendered_frames.iter().any(|&f| frame_path(&output, f).exists())
            {
                return Err("Existing images have no matching manifest".into());
            }
            json(&manifest_path, &manifest)?;
            for &frame in &manifest.rendered_frames {
                let path = frame_path(&output, frame);
                if !overwrite && path.exists() && sidecar(&path).exists() {
                    let receipt: Receipt = serde_json::from_slice(&fs::read(sidecar(&path))?)?;
                    if receipt.recipe_sha256 == hash
                        && receipt.png_sha256 == cache::file_hash(&path)?
                        && verify_light_samples(&manifest.config, frame, &receipt).is_ok()
                        && verify_engraving_samples(&manifest.config, frame, &receipt).is_ok()
                        && verify_eclipse_samples(&manifest.config, frame, &receipt).is_ok()
                    {
                        let dims = image::image_dimensions(&path)?;
                        if dims == (manifest.config.render.width, manifest.config.render.height) {
                            eprintln!("Reusing frame {frame}");
                            continue;
                        }
                    }
                }
                let instant = Instant::now();
                let fraction = frame as f64 / (manifest.config.frames - 1) as f64;
                let c = &manifest.config;
                let mut accumulated = Vec::new();
                let mut geometry = [0; 3];
                let mut light_samples = Vec::new();
                let mut engraving_samples = Vec::new();
                let mut eclipse_samples = Vec::new();
                let mut shutter_start = fraction;
                let mut shutter_end = fraction;
                for sample in 0..c.temporal_samples {
                    let time = expected_sample_time(c, frame, sample);
                    if sample == 0 {
                        shutter_start = time;
                    }
                    shutter_end = time;
                    let product =
                        art_frame(&source, time, expected_sample_interval(c, frame, sample), c)?;
                    geometry = product.geometry;
                    if let Some(diagnostics) = product.light {
                        light_samples.push(diagnostics);
                    }
                    if let Some(diagnostics) = product.engraving {
                        engraving_samples.push(diagnostics);
                    }
                    if let Some(diagnostics) = product.eclipse {
                        eclipse_samples.push(diagnostics);
                    }
                    let linear = product.pixels;
                    let weight = 1.0 / c.temporal_samples as f64;
                    if accumulated.is_empty() {
                        accumulated = linear;
                        accumulated.par_iter_mut().for_each(|v| *v *= weight);
                    } else {
                        if linear.len() != accumulated.len() {
                            return Err("Sub-frame image dimensions changed".into());
                        }
                        accumulated
                            .par_iter_mut()
                            .zip(linear.par_iter())
                            .for_each(|(a, b)| *a += *b * weight);
                    }
                    if progress
                        && (sample == 0
                            || (sample + 1).is_multiple_of(16)
                            || sample + 1 == c.temporal_samples)
                    {
                        write_progress_preview(&output, frame, sample + 1, c, &hash, &accumulated)?;
                    }
                    if matches!(c.kind.as_str(), "light" | "engraving" | "eclipse") {
                        eprintln!(
                            "{} frame {frame}: shutter sample {}/{}, {:.1}s elapsed",
                            c.kind,
                            sample + 1,
                            c.temporal_samples,
                            instant.elapsed().as_secs_f64()
                        );
                    }
                }
                let pixels = render::finish_linear(accumulated, &c.render)?;
                let temporary = path.with_extension("png.partial");
                pixels.save_with_format(&temporary, image::ImageFormat::Png)?;
                fs::rename(temporary, &path)?;
                json(
                    &sidecar(&path),
                    &Receipt {
                        recipe_sha256: hash.clone(),
                        png_sha256: cache::file_hash(&path)?,
                        source_fraction: fraction,
                        shutter_start_fraction: shutter_start,
                        shutter_end_fraction: shutter_end,
                        vertices: geometry[0],
                        triangles: geometry[1],
                        strands: geometry[2],
                        seconds: instant.elapsed().as_secs_f64(),
                        light_samples: (!light_samples.is_empty()).then_some(light_samples),
                        engraving_samples: (!engraving_samples.is_empty())
                            .then_some(engraving_samples),
                        eclipse_samples: (!eclipse_samples.is_empty()).then_some(eclipse_samples),
                    },
                )?;
                eprintln!(
                    "{} frame {frame}/{}: {:.2}s",
                    manifest.config.kind,
                    manifest.config.frames - 1,
                    instant.elapsed().as_secs_f64()
                );
            }
            manifest.complete = true;
            json(&manifest_path, &manifest)?;
        }
        Action::Encode { input, output, hq, encoder_threads, overwrite } => {
            encode(&input, &output, hq, encoder_threads, overwrite)?;
        }
        Action::Assemble { input, output } => assemble(&input, &output)?,
    }
    eprintln!("Completed in {:.2}s", started.elapsed().as_secs_f64());
    Ok(())
}

struct AssemblyChunk {
    directory: PathBuf,
    manifest: Manifest,
    manifest_sha256: String,
    recipe_sha256: String,
}

struct AssemblyFrame {
    frame: usize,
    chunk: usize,
    source: PathBuf,
    source_receipt: PathBuf,
    receipt: Receipt,
    receipt_sha256: String,
}

fn assembly_identity(manifest: &Manifest) -> SilkResult<String> {
    let mut identity = manifest.clone();
    identity.rendered_frames.clear();
    // Keep the original typed serialization and field order used by v03.
    recipe_hash(&identity)
}

fn resolve_output_location(path: &Path) -> SilkResult<PathBuf> {
    let absolute =
        if path.is_absolute() { path.to_owned() } else { std::env::current_dir()?.join(path) };
    let mut existing = absolute.as_path();
    let mut missing = Vec::new();
    while !existing.exists() {
        missing.push(existing.file_name().ok_or("Cannot resolve assembly output path")?.to_owned());
        existing = existing.parent().ok_or("Cannot resolve assembly output parent")?;
    }
    let mut resolved = fs::canonicalize(existing)?;
    for part in missing.into_iter().rev() {
        resolved.push(part);
    }
    Ok(resolved)
}

fn read_assembly_chunks(input: &[PathBuf]) -> SilkResult<Vec<AssemblyChunk>> {
    if input.is_empty() {
        return Err("Assembly requires at least one completed chunk".into());
    }
    let mut seen = BTreeSet::new();
    let mut chunks = Vec::with_capacity(input.len());
    for directory in input {
        let directory = fs::canonicalize(directory)?;
        if !directory.is_dir() || !seen.insert(directory.clone()) {
            return Err("Assembly inputs must be distinct directories".into());
        }
        if directory.to_str().is_none() {
            return Err("Assembly source paths must be UTF-8 for provenance".into());
        }
        let bytes = fs::read(directory.join("render.json"))?;
        let manifest: Manifest = serde_json::from_slice(&bytes)?;
        let config = &manifest.config;
        if manifest.schema_version != 1 || !manifest.complete || manifest.rendered_frames.is_empty()
        {
            return Err(format!(
                "Chunk {} is incomplete or has an unsupported manifest",
                directory.display()
            )
            .into());
        }
        if config.frames < 2
            || config.fps == 0
            || config.temporal_samples == 0
            || config.render.width == 0
            || config.render.height == 0
            || !config.shutter_fraction.is_finite()
            || !(0.0..=1.0).contains(&config.shutter_fraction)
            || !config.prelude_fraction.is_finite()
            || config.prelude_fraction < 0.0
            || !manifest.source_history_start_fraction.is_finite()
        {
            return Err(format!("Chunk {} has invalid film settings", directory.display()).into());
        }
        if manifest.rendered_frames.windows(2).any(|frames| frames[0] >= frames[1]) {
            return Err(format!(
                "Chunk {} frame indices are not strictly increasing",
                directory.display()
            )
            .into());
        }
        let recipe_sha256 = recipe_hash(&manifest)?;
        chunks.push(AssemblyChunk {
            directory,
            manifest,
            manifest_sha256: hex::encode(Sha256::digest(&bytes)),
            recipe_sha256,
        });
    }
    chunks.sort_by(|a, b| {
        a.manifest.rendered_frames[0]
            .cmp(&b.manifest.rendered_frames[0])
            .then_with(|| a.directory.cmp(&b.directory))
    });
    let identity = assembly_identity(&chunks[0].manifest)?;
    for chunk in &chunks[1..] {
        if assembly_identity(&chunk.manifest)? != identity {
            return Err(format!(
                "Chunk {} has a different recipe, renderer, orbit, seed, or source history",
                chunk.directory.display()
            )
            .into());
        }
    }
    Ok(chunks)
}

fn expected_frame_timing(config: &StudyConfig, frame: usize) -> [f64; 3] {
    let fraction = frame as f64 / (config.frames - 1) as f64;
    [
        fraction,
        expected_sample_time(config, frame, 0),
        expected_sample_time(config, frame, config.temporal_samples - 1),
    ]
}

fn expected_sample_time(config: &StudyConfig, frame: usize, sample: usize) -> f64 {
    let fraction = frame as f64 / (config.frames - 1) as f64;
    let offset = ((sample as f64 + 0.5) / config.temporal_samples as f64 - 0.5)
        * config.shutter_fraction
        / (config.frames - 1) as f64;
    (fraction + offset).clamp(0.0, 1.0)
}

fn expected_sample_interval(config: &StudyConfig, frame: usize, sample: usize) -> [f64; 2] {
    let fraction = frame as f64 / (config.frames - 1) as f64;
    [sample, sample + 1].map(|edge| {
        fraction
            + (edge as f64 / config.temporal_samples as f64 - 0.5) * config.shutter_fraction
                / (config.frames - 1) as f64
    })
}

fn verify_eclipse_samples(config: &StudyConfig, frame: usize, receipt: &Receipt) -> SilkResult<()> {
    if config.kind != "eclipse" {
        return if receipt.eclipse_samples.is_none() {
            Ok(())
        } else {
            Err("Non-Eclipse frame has unexpected Eclipse diagnostics".into())
        };
    }
    let samples = receipt.eclipse_samples.as_ref().ok_or("Missing Eclipse exposure diagnostics")?;
    if samples.len() != config.temporal_samples {
        return Err("Incomplete Eclipse exposure diagnostics".into());
    }
    let style = config.eclipse.as_ref().ok_or("Missing Eclipse parameters")?;
    let expected_curves = if style.corona_fraction == 0.0 {
        0
    } else {
        style.petals.iter().filter(|p| p.enabled && p.light_gain > 0.0).count() * style.corona_hairs
    };
    for (index, sample) in samples.iter().enumerate() {
        sample.validate()?;
        if sample.spatial_subcells != config.render.aa
            || sample.integration_tolerance.to_bits() != style.integration_tolerance.to_bits()
            || sample.max_spatial_depth > style.max_spatial_depth
        {
            return Err("Eclipse diagnostics do not match the recipe's integration settings".into());
        }
        if sample.curves != expected_curves
            || sample.segments != expected_curves * style.corona_segments
        {
            return Err("Eclipse diagnostics have inconsistent corona geometry".into());
        }
        if sample.source_fraction.to_bits() != expected_sample_time(config, frame, index).to_bits()
        {
            return Err("Mistimed Eclipse exposure diagnostics".into());
        }
    }
    Ok(())
}

fn verify_engraving_samples(
    config: &StudyConfig,
    frame: usize,
    receipt: &Receipt,
) -> SilkResult<()> {
    if config.kind != "engraving" {
        return if receipt.engraving_samples.is_none() {
            Ok(())
        } else {
            Err("Non-engraving frame has unexpected engraving diagnostics".into())
        };
    }
    let samples = receipt.engraving_samples.as_ref().ok_or("Missing engraving diagnostics")?;
    if samples.len() != config.temporal_samples {
        return Err("Incomplete engraving exposure cells".into());
    }
    for (index, sample) in samples.iter().enumerate() {
        sample.validate()?;
        let expected = expected_sample_interval(config, frame, index);
        if sample.source_fraction.to_bits() != expected_sample_time(config, frame, index).to_bits()
            || sample
                .raw_cell_interval
                .iter()
                .zip(expected)
                .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err("Engraving diagnostics have inconsistent exposure timing".into());
        }
    }
    Ok(())
}

fn verify_light_samples(config: &StudyConfig, frame: usize, receipt: &Receipt) -> SilkResult<()> {
    if config.kind != "light" {
        return if receipt.light_samples.is_none() {
            Ok(())
        } else {
            Err("Non-optical frame has unexpected light-transport diagnostics".into())
        };
    }
    let samples = receipt.light_samples.as_ref().ok_or("Missing light-transport diagnostics")?;
    if samples.len() != config.temporal_samples {
        return Err("Incomplete light-transport shutter samples".into());
    }
    for (index, sample) in samples.iter().enumerate() {
        sample.validate()?;
        if sample.source_fraction.to_bits() != expected_sample_time(config, frame, index).to_bits()
        {
            return Err("Light-transport diagnostics have inconsistent source timing".into());
        }
    }
    Ok(())
}

fn verify_assembly_frame(
    chunk: &AssemblyChunk,
    chunk_index: usize,
    frame: usize,
) -> SilkResult<AssemblyFrame> {
    let source = frame_path(&chunk.directory, frame);
    let source_receipt = sidecar(&source);
    let receipt_bytes = fs::read(&source_receipt)?;
    let receipt: Receipt = serde_json::from_slice(&receipt_bytes)?;
    verify_light_samples(&chunk.manifest.config, frame, &receipt)?;
    verify_engraving_samples(&chunk.manifest.config, frame, &receipt)?;
    verify_eclipse_samples(&chunk.manifest.config, frame, &receipt)?;
    if receipt.recipe_sha256 != chunk.recipe_sha256 {
        return Err(format!("Frame {frame} receipt does not match its source chunk recipe").into());
    }
    let actual_timing =
        [receipt.source_fraction, receipt.shutter_start_fraction, receipt.shutter_end_fraction];
    let expected_timing = expected_frame_timing(&chunk.manifest.config, frame);
    if actual_timing.into_iter().zip(expected_timing).any(|(a, b)| a.to_bits() != b.to_bits())
        || !receipt.seconds.is_finite()
        || receipt.seconds < 0.0
    {
        return Err(
            format!("Frame {frame} receipt has inconsistent source timing or diagnostics").into()
        );
    }
    let dimensions = image::image_dimensions(&source)?;
    if dimensions != (chunk.manifest.config.render.width, chunk.manifest.config.render.height) {
        return Err(format!("Frame {frame} dimensions do not match the source recipe").into());
    }
    if receipt.png_sha256 != cache::file_hash(&source)? {
        return Err(format!("Frame {frame} PNG hash does not match its source receipt").into());
    }
    Ok(AssemblyFrame {
        frame,
        chunk: chunk_index,
        source,
        source_receipt,
        receipt,
        receipt_sha256: hex::encode(Sha256::digest(&receipt_bytes)),
    })
}

fn install_assembly_frame(
    source: &AssemblyFrame,
    output: &Path,
    width: u32,
    height: u32,
) -> SilkResult<&'static str> {
    let destination = frame_path(output, source.frame);
    if destination.exists() {
        if cache::file_hash(&destination)? != source.receipt.png_sha256
            || image::image_dimensions(&destination)? != (width, height)
        {
            return Err(format!(
                "Existing assembled frame {} differs from the verified source",
                source.frame
            )
            .into());
        }
        return Ok("reused");
    }
    let temporary =
        destination.with_extension(format!("assembly-{}.partial.png", std::process::id()));
    let method = match fs::hard_link(&source.source, &temporary) {
        Ok(()) => "hard_link",
        Err(error) if error.kind() == ErrorKind::CrossesDevices => {
            // Copy only for a genuine cross-device destination; unrelated link
            // failures are surfaced instead of silently changing their meaning.
            let mut created = false;
            let copy_result = (|| -> std::io::Result<()> {
                let mut input = fs::File::open(&source.source)?;
                let mut output =
                    fs::OpenOptions::new().write(true).create_new(true).open(&temporary)?;
                created = true;
                std::io::copy(&mut input, &mut output)?;
                output.sync_all()
            })();
            if let Err(error) = copy_result {
                if created {
                    let _ = fs::remove_file(&temporary);
                }
                return Err(error.into());
            }
            "copy"
        }
        Err(error) => return Err(error.into()),
    };
    let result = (|| -> SilkResult<()> {
        if cache::file_hash(&temporary)? != source.receipt.png_sha256
            || image::image_dimensions(&temporary)? != (width, height)
        {
            return Err(
                format!("Frame {} changed while it was being assembled", source.frame).into()
            );
        }
        fs::rename(&temporary, &destination)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result?;
    Ok(method)
}

fn assemble(input: &[PathBuf], output: &Path) -> SilkResult<()> {
    let chunks = read_assembly_chunks(input)?;
    let output = resolve_output_location(output)?;
    if output.to_str().is_none() {
        return Err("Assembly output path must be UTF-8 for provenance".into());
    }
    if chunks
        .iter()
        .any(|chunk| output.starts_with(&chunk.directory) || chunk.directory.starts_with(&output))
    {
        return Err("Assembly output must be separate from every source chunk directory".into());
    }
    let frame_count = chunks[0].manifest.config.frames;
    let mut coverage = BTreeMap::new();
    for (chunk_index, chunk) in chunks.iter().enumerate() {
        for &frame in &chunk.manifest.rendered_frames {
            if frame >= frame_count {
                return Err(format!("Frame {frame} is outside the configured film").into());
            }
            if coverage.insert(frame, chunk_index).is_some() {
                return Err(format!("Duplicate source frame {frame}").into());
            }
        }
    }
    if coverage.len() != frame_count {
        let missing = (0..frame_count)
            .find(|frame| !coverage.contains_key(frame))
            .ok_or("Invalid assembly coverage")?;
        return Err(format!(
            "Assembly is missing frame {missing}; have {} of {frame_count}",
            coverage.len()
        )
        .into());
    }
    let plan: Vec<_> = coverage.into_iter().collect();
    // Work in parallel, but collect errors and frames in source-index order.
    let checked: Vec<SilkResult<AssemblyFrame>> = plan
        .par_iter()
        .map(|&(frame, chunk)| verify_assembly_frame(&chunks[chunk], chunk, frame))
        .collect();
    let verified = checked.into_iter().collect::<SilkResult<Vec<_>>>()?;
    let mut manifest = chunks[0].manifest.clone();
    manifest.rendered_frames = (0..frame_count).collect();
    manifest.complete = false;
    let full_hash = recipe_hash(&manifest)?;
    let assembler_hash = cache::file_hash(&std::env::current_exe()?)?;
    fs::create_dir_all(&output)?;
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(output.join(".assembly.lock"))?;
    lock.try_lock()
        .map_err(|error| format!("Another assembly owns this output directory: {error}"))?;
    let manifest_path = output.join("render.json");
    if manifest_path.exists() {
        let previous: Manifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
        if recipe_hash(&previous)? != full_hash {
            return Err(
                "Assembly output contains a different recipe; use a separate directory".into()
            );
        }
    } else if fs::read_dir(&output)?
        .any(|entry| entry.map_or(true, |entry| entry.file_name() != ".assembly.lock"))
    {
        return Err("Assembly output is not empty and has no matching manifest".into());
    }
    let sources:Vec<_>=chunks.iter().enumerate().map(|(index,chunk)|serde_json::json!({
        "index":index,"directory":chunk.directory,"manifest_sha256":chunk.manifest_sha256,
        "recipe_sha256":chunk.recipe_sha256,"rendered_frames":chunk.manifest.rendered_frames,
    })).collect();
    let mut proof = serde_json::json!({
        "schema_version":1,"kind":"verified-frame-assembly","complete":false,
        "renderer_executable_sha256":manifest.executable_sha256,
        "assembler_executable_sha256":assembler_hash,
        "source_identity_sha256":assembly_identity(&manifest)?,
        "assembled_recipe_sha256":full_hash,"frame_count":frame_count,
        "pixel_operation":"none; PNG bytes are linked or copied without decoding/re-encoding",
        "source_files_unchanged":true,"sources":sources,"frames":[],
    });
    json(&manifest_path, &manifest)?;
    json(&output.join("assembly.json"), &proof)?;
    let mut mapping = Vec::with_capacity(frame_count);
    for source in &verified {
        let method = install_assembly_frame(
            source,
            &output,
            manifest.config.render.width,
            manifest.config.render.height,
        )?;
        let destination = frame_path(&output, source.frame);
        let mut receipt = source.receipt.clone();
        receipt.recipe_sha256.clone_from(&full_hash);
        json(&sidecar(&destination), &receipt)?;
        mapping.push(serde_json::json!({
            "frame":source.frame,"source_manifest_index":source.chunk,
            "source_frame":source.frame,"source_path":source.source,
            "source_receipt_sha256":source.receipt_sha256,"png_sha256":source.receipt.png_sha256,
            "output_path":format!("frame_{:06}.png",source.frame),"output_receipt_sha256":cache::file_hash(&sidecar(&destination))?,
            "transfer":method,
        }));
    }
    // Recheck after installation, before either completion flag is published.
    for chunk in &chunks {
        if cache::file_hash(&chunk.directory.join("render.json"))? != chunk.manifest_sha256 {
            return Err("A source manifest changed during assembly".into());
        }
    }
    for source in &verified {
        if cache::file_hash(&source.source_receipt)? != source.receipt_sha256
            || cache::file_hash(&source.source)? != source.receipt.png_sha256
            || cache::file_hash(&frame_path(&output, source.frame))? != source.receipt.png_sha256
        {
            return Err(format!(
                "Frame {} or its source receipt changed during assembly",
                source.frame
            )
            .into());
        }
    }
    proof["frames"] = serde_json::to_value(mapping)?;
    proof["complete"] = true.into();
    json(&output.join("assembly.json"), &proof)?;
    manifest.complete = true;
    json(&manifest_path, &manifest)?;
    eprintln!(
        "Assembled {frame_count} verified frames from {} chunks into {}",
        chunks.len(),
        output.display()
    );
    Ok(())
}

fn encode(
    input: &Path,
    output: &Path,
    hq: bool,
    threads: usize,
    overwrite: bool,
) -> SilkResult<()> {
    if threads == 0 || (output.exists() && !overwrite) {
        return Err("Encoder threads must be positive; existing output requires --overwrite".into());
    }
    let manifest: Manifest = serde_json::from_slice(&fs::read(input.join("render.json"))?)?;
    if !manifest.complete || manifest.rendered_frames.is_empty() {
        return Err("Render is incomplete".into());
    }
    let stride = if manifest.rendered_frames.len() > 1 {
        manifest.rendered_frames[1]
            .checked_sub(manifest.rendered_frames[0])
            .ok_or("Invalid frame order")?
    } else {
        1
    };
    if stride == 0
        || manifest.rendered_frames.windows(2).any(|w| w[1].checked_sub(w[0]) != Some(stride))
    {
        return Err("Frames must be strictly increasing at a fixed interval".into());
    }
    let c = &manifest.config;
    if !c.render.width.is_multiple_of(2) || !c.render.height.is_multiple_of(2) {
        return Err("Movies require even pixel dimensions".into());
    }
    let hash = recipe_hash(&manifest)?;
    for &frame in &manifest.rendered_frames {
        let path = frame_path(input, frame);
        let receipt: Receipt = serde_json::from_slice(&fs::read(sidecar(&path))?)?;
        verify_light_samples(c, frame, &receipt)?;
        verify_engraving_samples(c, frame, &receipt)?;
        verify_eclipse_samples(c, frame, &receipt)?;
        if receipt.recipe_sha256 != hash || receipt.png_sha256 != cache::file_hash(&path)? {
            return Err(format!("Frame {frame} failed its integrity check").into());
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
        &format!("{}x{}", c.render.width, c.render.height),
        "-framerate",
        &format!("{}/{}", c.fps, stride),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        if hq { "libx265" } else { "libx264" },
        "-preset",
        "slow",
        "-crf",
        if hq { "12" } else { "16" },
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
            &format!("pools={threads}:frame-threads=3:log-level=error"),
        ]);
    } else {
        command.args(["-x264-params", "cpu-independent=1"]);
    }
    let mut child = command.arg(&temporary).stdin(Stdio::piped()).spawn()?;
    let write_result = (|| -> SilkResult<()> {
        let mut pipe = child.stdin.take().ok_or("Encoder stdin unavailable")?;
        for &frame in &manifest.rendered_frames {
            let image = image::open(frame_path(input, frame))?.to_rgb16();
            if image.dimensions() != (c.render.width, c.render.height) {
                return Err("Frame dimensions changed".into());
            }
            let bytes: Vec<u8> = image.as_raw().iter().flat_map(|v| v.to_le_bytes()).collect();
            pipe.write_all(&bytes)?;
        }
        Ok(())
    })();
    if let Err(error) = write_result {
        let _ = child.kill();
        let _ = child.wait();
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    if !child.wait()?.success() {
        let _ = fs::remove_file(&temporary);
        return Err("FFmpeg encoding failed".into());
    }
    fs::rename(temporary, output)?;
    let version = Command::new("ffmpeg").arg("-version").output()?;
    let encoder =
        String::from_utf8_lossy(&version.stdout).lines().next().unwrap_or("unknown").to_owned();
    json(
        &sidecar(output),
        &serde_json::json!({
            "schema_version":1,"source":manifest,"recipe_sha256":hash,
            "encoder":encoder,"encoder_threads":threads,"codec":if hq{"hevc10"}else{"h264"},
            "duration_seconds":manifest.rendered_frames.len() as f64 * stride as f64 / f64::from(c.fps),
            "sha256":cache::file_hash(output)?,
        }),
    )?;
    eprintln!("Film: {}", output.display());
    Ok(())
}

#[cfg(test)]
mod assembly_tests {
    use super::*;

    #[test]
    fn eclipse_receipts_require_complete_correctly_timed_geometry_records() {
        let config = StudyConfig {
            kind: "eclipse".into(),
            calligraphy: None,
            eclipse: Some(eclipse::EclipseConfig::default()),
            temporal_samples: 2,
            ..StudyConfig::default()
        };
        let diagnostic = |time| eclipse::EclipseDiagnostics {
            source_fraction: time,
            centers: [[0.0; 2]; 3],
            axis_factors: [1.0; 3],
            curves: 4608,
            segments: 294_912,
            spatial_subcells: config.render.aa,
            accepted_cells: 1,
            refinements: 0,
            max_spatial_depth: 0,
            max_accepted_error_indicator: 0.0,
            integration_tolerance: 1e-4,
            max_chord_error_pixels: 0.002,
            estimated_corona_luminance: 0.1,
            minimum_linear_channel: 0.0,
            maximum_linear_channel: 1.0,
        };
        let mut receipt = Receipt {
            recipe_sha256: String::new(),
            png_sha256: String::new(),
            source_fraction: 0.0,
            shutter_start_fraction: 0.0,
            shutter_end_fraction: 0.0,
            vertices: 0,
            triangles: 0,
            strands: 0,
            seconds: 0.0,
            light_samples: None,
            engraving_samples: None,
            eclipse_samples: None,
        };
        assert!(verify_eclipse_samples(&StudyConfig::default(), 0, &receipt).is_ok());
        assert!(verify_eclipse_samples(&config, 0, &receipt).is_err());
        receipt.eclipse_samples =
            Some((0..2).map(|i| diagnostic(expected_sample_time(&config, 0, i))).collect());
        verify_eclipse_samples(&config, 0, &receipt).unwrap();
        receipt.eclipse_samples.as_mut().unwrap()[0].integration_tolerance = 0.002;
        assert!(verify_eclipse_samples(&config, 0, &receipt).is_err());
        receipt.eclipse_samples.as_mut().unwrap()[0].integration_tolerance = 1e-4;
        assert!(verify_eclipse_samples(&StudyConfig::default(), 0, &receipt).is_err());
        receipt.eclipse_samples.as_mut().unwrap()[1].source_fraction = 0.4;
        assert!(verify_eclipse_samples(&config, 0, &receipt).is_err());
        receipt.eclipse_samples.as_mut().unwrap()[1] =
            diagnostic(expected_sample_time(&config, 0, 1));
        receipt.eclipse_samples.as_mut().unwrap()[1].segments -= 1;
        assert!(verify_eclipse_samples(&config, 0, &receipt).is_err());
        receipt.eclipse_samples.as_mut().unwrap().pop();
        assert!(verify_eclipse_samples(&config, 0, &receipt).is_err());
    }

    #[test]
    fn archived_v03_manifest_keeps_its_original_recipe_hash() {
        let manifest: Manifest =
            serde_json::from_str(include_str!("../../tests/fixtures/atelier-v03-manifest.json"))
                .unwrap();
        assert!(manifest.config.calligraphy.is_some());
        assert!(manifest.config.loom.is_none());
        assert!(manifest.config.aurora.is_none());
        assert!(manifest.config.light.is_none());
        assert_eq!(
            recipe_hash(&manifest).unwrap(),
            "f40ec9d064ca6c7711f1955998c26ac183c9f3b6cc7bd477cb41e8578b1b9748"
        );
    }

    #[test]
    fn archived_v08_loom_keeps_its_original_recipe_hash() {
        let manifest: Manifest = serde_json::from_str(include_str!(
            "../../tests/fixtures/atelier-v08-loom-manifest.json"
        ))
        .unwrap();
        assert!(manifest.config.calligraphy.is_none());
        assert!(manifest.config.loom.is_some());
        assert!(manifest.config.aurora.is_none());
        assert!(manifest.config.light.is_none());
        assert_eq!(
            recipe_hash(&manifest).unwrap(),
            "dac3b725dd71d3accbb0042d61ebde3174c8956c95720c2c4702eb669ba2a0b7"
        );
    }

    #[test]
    fn archived_v11_aurora_keeps_its_original_recipe_hash() {
        let manifest: Manifest = serde_json::from_str(include_str!(
            "../../tests/fixtures/atelier-v11-aurora-manifest.json"
        ))
        .unwrap();
        assert!(manifest.config.aurora.is_some());
        assert!(manifest.config.light.is_none());
        assert_eq!(
            recipe_hash(&manifest).unwrap(),
            "bf17b61e3f8f304bc4eedb52d1fa4a8a0f72b42d6ed4d213f2901c3c91272acd"
        );
    }

    #[test]
    fn archived_v12_light_keeps_its_original_recipe_hash() {
        let manifest: Manifest = serde_json::from_str(include_str!(
            "../../tests/fixtures/atelier-v12-light-manifest.json"
        ))
        .unwrap();
        assert!(manifest.config.light.is_some());
        assert_eq!(
            recipe_hash(&manifest).unwrap(),
            "6a9d84428443f4138865dad8d2eea7fa504b71ea24a3ad7a61b6e023faef3abd"
        );
    }

    #[test]
    fn archived_v16_engraving_keeps_its_original_recipe_hash() {
        let manifest: Manifest = serde_json::from_str(include_str!(
            "../../tests/fixtures/atelier-v16-engraving-manifest.json"
        ))
        .unwrap();
        assert!(manifest.config.engraving.is_some());
        assert_eq!(
            recipe_hash(&manifest).unwrap(),
            "994f194a5f5fd722aba2dc6178b5aa87d24715725860c58a0742df466f975b29"
        );
    }

    #[test]
    fn light_receipts_require_complete_optical_samples_and_legacy_receipts_do_not() {
        let config = StudyConfig {
            kind: "light".into(),
            calligraphy: None,
            light: Some(light::LightConfig::default()),
            ..StudyConfig::default()
        };
        let mut receipt = Receipt {
            recipe_sha256: String::new(),
            png_sha256: String::new(),
            source_fraction: 0.0,
            shutter_start_fraction: 0.0,
            shutter_end_fraction: 0.0,
            vertices: 0,
            triangles: 0,
            strands: 0,
            seconds: 0.0,
            light_samples: None,
            engraving_samples: None,
            eclipse_samples: None,
        };
        assert!(verify_light_samples(&StudyConfig::default(), 0, &receipt).is_ok());
        assert!(verify_light_samples(&config, 0, &receipt).is_err());
        receipt.light_samples = Some(Vec::new());
        assert!(verify_light_samples(&config, 0, &receipt).is_err());
        assert!(verify_light_samples(&StudyConfig::default(), 0, &receipt).is_err());
    }

    #[test]
    fn exposure_cells_cover_the_complete_unclamped_shutter_without_gaps() {
        let config = StudyConfig { frames: 3, temporal_samples: 16, ..StudyConfig::default() };
        for (frame, expected) in [(0, [-0.125, 0.125]), (1, [0.375, 0.625]), (2, [0.875, 1.125])] {
            let cells: Vec<_> =
                (0..16).map(|index| expected_sample_interval(&config, frame, index)).collect();
            assert_eq!(cells[0][0], expected[0]);
            assert_eq!(cells[15][1], expected[1]);
            assert!(cells.windows(2).all(|pair| pair[0][1].to_bits() == pair[1][0].to_bits()));
        }
        assert_eq!(expected_sample_time(&config, 0, 0), 0.0);
        assert_eq!(expected_sample_time(&config, 2, 15), 1.0);
    }

    #[test]
    fn engraving_receipts_require_their_exposure_cell_records() {
        let config = StudyConfig {
            kind: "engraving".into(),
            calligraphy: None,
            engraving: Some(engraving::EngravingConfig::default()),
            ..StudyConfig::default()
        };
        let mut receipt = Receipt {
            recipe_sha256: String::new(),
            png_sha256: String::new(),
            source_fraction: 0.0,
            shutter_start_fraction: 0.0,
            shutter_end_fraction: 0.0,
            vertices: 0,
            triangles: 0,
            strands: 0,
            seconds: 0.0,
            light_samples: None,
            engraving_samples: None,
            eclipse_samples: None,
        };
        assert!(verify_engraving_samples(&StudyConfig::default(), 0, &receipt).is_ok());
        assert!(verify_engraving_samples(&config, 0, &receipt).is_err());
        receipt.engraving_samples = Some(Vec::new());
        assert!(verify_engraving_samples(&config, 0, &receipt).is_err());
        assert!(verify_engraving_samples(&StudyConfig::default(), 0, &receipt).is_err());
    }

    fn manifest(frames: &[usize]) -> Manifest {
        Manifest {
            schema_version: 1,
            seed: "0x01".into(),
            orbit_sha256: "a".repeat(64),
            executable_sha256: "b".repeat(64),
            source_history_start_fraction: -0.5,
            config: StudyConfig {
                frames: 4,
                fps: 60,
                render: RenderConfig { width: 4, height: 4, ..RenderConfig::default() },
                ..StudyConfig::default()
            },
            rendered_frames: frames.to_vec(),
            complete: true,
        }
    }

    fn chunk(directory: &Path, manifest: &Manifest) {
        fs::create_dir_all(directory).unwrap();
        json(&directory.join("render.json"), manifest).unwrap();
        let hash = recipe_hash(manifest).unwrap();
        for &frame in &manifest.rendered_frames {
            let path = frame_path(directory, frame);
            image::ImageBuffer::from_pixel(
                manifest.config.render.width,
                manifest.config.render.height,
                image::Rgb([1000 + frame as u16, 20_000, 60_000]),
            )
            .save(&path)
            .unwrap();
            let [source_fraction, shutter_start_fraction, shutter_end_fraction] =
                expected_frame_timing(&manifest.config, frame);
            json(
                &sidecar(&path),
                &Receipt {
                    recipe_sha256: hash.clone(),
                    png_sha256: cache::file_hash(&path).unwrap(),
                    source_fraction,
                    shutter_start_fraction,
                    shutter_end_fraction,
                    vertices: 12,
                    triangles: 10,
                    strands: 5,
                    seconds: 0.25,
                    light_samples: None,
                    engraving_samples: None,
                    eclipse_samples: None,
                },
            )
            .unwrap();
        }
    }

    #[test]
    fn assembly_preserves_pixels_and_source_receipts_but_retags_full_recipe() {
        let temp = tempfile::tempdir().unwrap();
        let a = temp.path().join("a");
        let b = temp.path().join("b");
        let output = temp.path().join("assembled");
        chunk(&a, &manifest(&[0, 1]));
        chunk(&b, &manifest(&[2, 3]));
        let original_receipts: Vec<_> = [(&a, 0), (&a, 1), (&b, 2), (&b, 3)]
            .into_iter()
            .map(|(directory, frame)| fs::read(sidecar(&frame_path(directory, frame))).unwrap())
            .collect();
        assemble(&[b.clone(), a.clone()], &output).unwrap();
        let full: Manifest =
            serde_json::from_slice(&fs::read(output.join("render.json")).unwrap()).unwrap();
        assert!(full.complete);
        assert_eq!(full.rendered_frames, vec![0, 1, 2, 3]);
        assert_eq!(full.executable_sha256, "b".repeat(64));
        let full_hash = recipe_hash(&full).unwrap();
        for (frame, original) in original_receipts.iter().enumerate() {
            let directory = if frame < 2 { &a } else { &b };
            let source_path = frame_path(directory, frame);
            let output_path = frame_path(&output, frame);
            assert_eq!(fs::read(&output_path).unwrap(), fs::read(&source_path).unwrap());
            assert_eq!(fs::read(sidecar(&source_path)).unwrap(), *original);
            let receipt: Receipt =
                serde_json::from_slice(&fs::read(sidecar(&output_path)).unwrap()).unwrap();
            assert_eq!(receipt.recipe_sha256, full_hash);
            let source_receipt: Receipt = serde_json::from_slice(original).unwrap();
            assert_ne!(receipt.recipe_sha256, source_receipt.recipe_sha256);
            assert_eq!(receipt.png_sha256, source_receipt.png_sha256);
            assert_eq!(receipt.source_fraction.to_bits(), source_receipt.source_fraction.to_bits());
        }
        let proof: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("assembly.json")).unwrap()).unwrap();
        assert_eq!(proof["complete"], true);
        assert_eq!(proof["renderer_executable_sha256"], "b".repeat(64));
        assert_eq!(proof["frames"].as_array().unwrap().len(), 4);
        assert_eq!(proof["frames"][0]["source_manifest_index"], 0);
        assert_eq!(proof["frames"][2]["source_manifest_index"], 1);
        assert_eq!(
            proof["sources"][0]["manifest_sha256"],
            cache::file_hash(&a.join("render.json")).unwrap()
        );
        // Repeated verification may reuse files but must preserve all pixels.
        assemble(&[a, b], &output).unwrap();
        assert_eq!(
            full_hash,
            recipe_hash(
                &serde_json::from_slice(&fs::read(output.join("render.json")).unwrap()).unwrap()
            )
            .unwrap()
        );
    }

    #[test]
    fn missing_duplicate_and_out_of_range_frames_fail_before_output_creation() {
        for issue in ["missing", "duplicate", "out-of-range"] {
            let temp = tempfile::tempdir().unwrap();
            let a = temp.path().join("a");
            let b = temp.path().join("b");
            let output = temp.path().join("assembled");
            chunk(&a, &manifest(&[0, 1]));
            let frames = match issue {
                "missing" => vec![3],
                "duplicate" => vec![1, 2, 3],
                _ => vec![2, 4],
            };
            chunk(&b, &manifest(&frames));
            assert!(assemble(&[a, b], &output).is_err(), "accepted {issue}");
            assert!(!output.exists());
        }
    }

    #[test]
    fn mixed_renderer_or_source_or_config_is_rejected() {
        for issue in ["renderer", "orbit", "seed", "history", "config"] {
            let temp = tempfile::tempdir().unwrap();
            let a = temp.path().join("a");
            let b = temp.path().join("b");
            let output = temp.path().join("assembled");
            chunk(&a, &manifest(&[0, 1]));
            let mut different = manifest(&[2, 3]);
            match issue {
                "renderer" => different.executable_sha256 = "c".repeat(64),
                "orbit" => different.orbit_sha256 = "c".repeat(64),
                "seed" => different.seed = "0x02".into(),
                "history" => different.source_history_start_fraction = -0.25,
                _ => different.config.render.exposure = 1.2,
            }
            chunk(&b, &different);
            let error = assemble(&[a, b], &output).unwrap_err();
            assert!(
                error.to_string().contains("different recipe"),
                "wrong error for {issue}: {error}"
            );
            assert!(!output.exists());
        }
    }

    #[test]
    fn corrupt_png_receipt_or_dimensions_are_rejected() {
        for issue in ["png", "receipt", "timing", "dimensions"] {
            let temp = tempfile::tempdir().unwrap();
            let a = temp.path().join("a");
            let b = temp.path().join("b");
            let output = temp.path().join("assembled");
            chunk(&a, &manifest(&[0, 1]));
            chunk(&b, &manifest(&[2, 3]));
            let path = frame_path(&b, 3);
            match issue {
                "png" => image::ImageBuffer::from_pixel(4, 4, image::Rgb([0_u16; 3]))
                    .save(&path)
                    .unwrap(),
                "dimensions" => image::ImageBuffer::from_pixel(2, 8, image::Rgb([0_u16; 3]))
                    .save(&path)
                    .unwrap(),
                _ => {
                    let mut receipt: Receipt =
                        serde_json::from_slice(&fs::read(sidecar(&path)).unwrap()).unwrap();
                    if issue == "receipt" {
                        receipt.recipe_sha256 = "bad".into();
                    } else {
                        receipt.source_fraction = 0.5;
                    }
                    json(&sidecar(&path), &receipt).unwrap();
                }
            }
            assert!(assemble(&[a, b], &output).is_err(), "accepted {issue}");
            assert!(!output.exists());
        }
    }

    #[test]
    fn incomplete_chunks_and_aliasing_output_are_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let a = temp.path().join("a");
        let output = temp.path().join("assembled");
        let mut incomplete = manifest(&[0, 1, 2, 3]);
        incomplete.complete = false;
        chunk(&a, &incomplete);
        assert!(assemble(std::slice::from_ref(&a), &output).is_err());
        assert!(!output.exists());
        incomplete.complete = true;
        chunk(&a, &incomplete);
        assert!(assemble(std::slice::from_ref(&a), &a).is_err());
        assert!(assemble(std::slice::from_ref(&a), &a.join("nested")).is_err());
    }

    #[test]
    fn interrupted_assembly_reuses_verified_pixels_with_missing_output_receipts() {
        let temp = tempfile::tempdir().unwrap();
        let a = temp.path().join("a");
        let b = temp.path().join("b");
        let output = temp.path().join("assembled");
        chunk(&a, &manifest(&[0, 1]));
        chunk(&b, &manifest(&[2, 3]));
        fs::create_dir_all(&output).unwrap();
        let mut full = manifest(&[0, 1, 2, 3]);
        full.complete = false;
        json(&output.join("render.json"), &full).unwrap();
        fs::copy(frame_path(&a, 0), frame_path(&output, 0)).unwrap();
        assemble(&[a, b], &output).unwrap();
        let finished: Manifest =
            serde_json::from_slice(&fs::read(output.join("render.json")).unwrap()).unwrap();
        assert!(finished.complete);
        assert!(sidecar(&frame_path(&output, 0)).exists());
        let proof: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("assembly.json")).unwrap()).unwrap();
        assert_eq!(proof["frames"][0]["transfer"], "reused");
    }
}
