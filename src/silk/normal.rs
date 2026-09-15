//! Replay the original accumulating artwork from a frozen physical orbit.
//!
//! The recorded winner and all main-stream random draws are reconstructed without
//! rerunning candidate simulations. Projection, viewing orientation, drift, color,
//! histogram analysis and video accumulation remain the production operations.

use std::fs;
use std::io;
use std::path::Path;
use std::sync::atomic::Ordering;

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tracing::info;

use super::orbit::InitialBody;
use super::{OrbitData, SilkResult, V3, cache};
use crate::generation_log::GenerationRecord;
use crate::render::visual_profile::ResolvedVisualProfile;
use crate::sim::{self, Body, Sha3RandomByteStream};
use crate::{app, render};

/// Output settings independent of the original source-selection resolution.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NormalConfig {
    /// Output width in pixels; must be positive and even.
    pub width: u32,
    /// Output height in pixels; must be positive and even.
    pub height: u32,
    /// Use the production fast encoder for the secondary video variant.
    pub fast_encode: bool,
}

impl Default for NormalConfig {
    fn default() -> Self {
        Self { width: 1280, height: 828, fast_encode: false }
    }
}

/// Inclusive source-step checkpoint of each original normal-video frame.
///
/// This reproduces `render::checkpoint_steps` with the production target-frame
/// interval. One million samples produce 1,802 frames, not exactly 1,800.
#[must_use]
pub fn frame_schedule(total_steps: usize) -> Vec<usize> {
    if total_steps == 0 {
        return Vec::new();
    }
    let interval = (total_steps / render::constants::DEFAULT_TARGET_FRAMES as usize).max(1);
    let mut checkpoints: Vec<usize> = (interval..total_steps).step_by(interval).collect();
    let last = total_steps - 1;
    if checkpoints.last().copied() != Some(last) {
        checkpoints.push(last);
    }
    checkpoints
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn nearly_equal(first: f64, second: f64) -> bool {
    first.is_finite()
        && second.is_finite()
        && (first - second).abs() <= 8.0 * f64::EPSILON * first.abs().max(second.abs()).max(1.0)
}

fn verify_number(name: &str, actual: f64, recorded: f64) -> SilkResult<()> {
    if !nearly_equal(actual, recorded) {
        return Err(invalid(format!(
            "Recorded {name} differs from the current generator: {recorded} versus {actual}"
        ))
        .into());
    }
    Ok(())
}

fn sample_candidate(rng: &mut Sha3RandomByteStream) -> Vec<Body> {
    let mut bodies = (0..3)
        .map(|_| {
            Body::new(
                rng.random_mass(),
                Vector3::new(rng.random_location(), rng.random_location(), rng.random_location()),
                Vector3::new(rng.random_velocity(), rng.random_velocity(), rng.random_velocity()),
            )
        })
        .collect::<Vec<_>>();
    sim::shift_bodies_to_com(&mut bodies);
    bodies
}

/// Advance an entire batch, retaining only the requested candidate's states.
fn consume_candidates(
    rng: &mut Sha3RandomByteStream,
    count: usize,
    selected: Option<usize>,
) -> Option<Vec<Body>> {
    let mut result = None;
    for index in 0..count {
        if selected == Some(index) {
            result = Some(sample_candidate(rng));
        } else {
            for _ in 0..21 {
                let _ = rng.next_f64();
            }
        }
    }
    result
}

fn resolve_recorded_stack(
    preferred: render::LayerStack,
    chosen: &str,
) -> SilkResult<render::LayerStack> {
    // The full seeded stack retains its exact layer alpha values. Parsing its
    // rounded display label would lose information (for example 0.0777 -> 0.08).
    let candidates = [
        preferred,
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
    ];
    candidates
        .into_iter()
        .find(|stack| stack.label() == chosen)
        .ok_or_else(|| invalid(format!("Unsupported recorded structure: {chosen}")).into())
}

fn validate_inputs(
    orbit: &OrbitData,
    record: &GenerationRecord,
    config: &NormalConfig,
) -> SilkResult<()> {
    crate::error::validation::validate_dimensions(config.width, config.height)?;
    if !config.width.is_multiple_of(2) || !config.height.is_multiple_of(2) {
        return Err(invalid("Normal video output dimensions must be even").into());
    }
    let source = &record.simulation_config;
    if record.render_config.visual_profile != "cosmic_signature"
        || record.render_config.width == 0
        || record.render_config.height == 0
        || record.render_config.alpha_denom == 0
    {
        return Err(
            invalid("Generation record has an unsupported or invalid visual profile").into()
        );
    }
    if !(2..=100_000_000).contains(&source.num_steps_sim)
        || source.num_sims == 0
        || source.num_sims > 10_000_000
        || record.orbit_info.selected_index >= source.num_sims
        || record.orbit_info.retry_count > app::AESTHETIC_MAX_RETRIES
    {
        return Err(
            invalid("Generation record has invalid simulation or selection settings").into()
        );
    }
    if ![source.min_mass, source.max_mass, source.location, source.velocity]
        .iter()
        .all(|value| value.is_finite())
        || source.min_mass <= 0.0
        || source.max_mass < source.min_mass
        || source.location <= 0.0
        || source.velocity < 0.0
    {
        return Err(invalid("Generation record has invalid physical sampling bounds").into());
    }
    if app::parse_seed(&orbit.seed)? != app::parse_seed(&record.seed)? {
        return Err(invalid("Orbit cache and generation record refer to different seeds").into());
    }
    if orbit.samples.len() != source.num_steps_sim
        || orbit.dt.to_bits() != render::constants::DEFAULT_DT.to_bits()
    {
        return Err(invalid(
            "Normal replay requires every recorded sample at dt=0.001; use an uncropped sample_stride=1 orbit cache",
        )
        .into());
    }
    if orbit.samples.iter().flatten().any(|position| !position.is_finite()) {
        return Err(invalid("Orbit cache contains non-finite physical positions").into());
    }
    if orbit.provenance["warmup_steps"].as_u64() != Some(source.num_steps_sim as u64)
        || orbit.provenance["recording_steps"].as_u64() != Some(source.num_steps_sim as u64)
        || orbit.provenance["sample_stride"].as_u64() != Some(1)
    {
        return Err(
            invalid("Orbit provenance does not describe the complete recorded source").into()
        );
    }
    Ok(())
}

fn verify_initial_conditions(orbit: &OrbitData, expected: &[Body]) -> SilkResult<()> {
    let initial: [InitialBody; 3] =
        serde_json::from_value(orbit.provenance["initial_conditions"].clone()).map_err(
            |error| invalid(format!("Orbit cache needs its source initial conditions: {error}")),
        )?;
    for (index, (actual, expected)) in initial.iter().zip(expected).enumerate() {
        if actual.mass.to_bits() != expected.mass.to_bits()
            || orbit.masses[index].to_bits() != expected.mass.to_bits()
        {
            return Err(
                invalid(format!("Body {index} mass does not match the recorded winner")).into()
            );
        }
        let actual_coordinates = [
            actual.position.x,
            actual.position.y,
            actual.position.z,
            actual.velocity.x,
            actual.velocity.y,
            actual.velocity.z,
        ];
        let expected_coordinates = [
            expected.position.x,
            expected.position.y,
            expected.position.z,
            expected.velocity.x,
            expected.velocity.y,
            expected.velocity.z,
        ];
        if actual_coordinates
            .into_iter()
            .zip(expected_coordinates)
            .any(|(a, b)| !nearly_equal(a, b))
        {
            return Err(invalid(format!(
                "Body {index} source states do not match the generation record's selected candidate",
            ))
            .into());
        }
    }
    Ok(())
}

struct Prepared {
    positions: Vec<Vec<Vector3<f64>>>,
    colors: Vec<Vec<render::OklabColor>>,
    alphas: Vec<f64>,
    profile: ResolvedVisualProfile,
    view: (f64, f64, f64),
    palette: (String, String),
    closest_approach: Value,
}

fn closest_event(positions: &[Vec<Vector3<f64>>], dt: f64) -> SilkResult<Value> {
    let event = crate::traits_analysis::closest_approach(positions)
        .ok_or("Cannot measure closest approach without three complete trajectories")?;
    if !event.distance.is_finite() {
        return Err(invalid("Raw closest-approach distance is not finite").into());
    }
    let steps = positions[0].len();
    let schedule = frame_schedule(steps);
    let (frame, &checkpoint) = schedule
        .iter()
        .enumerate()
        .min_by_key(|(_, checkpoint)| checkpoint.abs_diff(event.step))
        .ok_or("Normal video has no frame checkpoints")?;
    let body_ids = ["A", "B", "C"];
    Ok(json!({
        "source_index":event.step,
        "source_fraction":event.step as f64/(steps-1) as f64,
        "source_time_seconds":event.step as f64*dt,
        "pair":[event.pair.0,event.pair.1],
        "body_ids":[body_ids[event.pair.0],body_ids[event.pair.1]],
        "distance":event.distance,
        "coordinate_space":"raw-Newtonian-positions",
        "normal_frame":frame,
        "normal_checkpoint_source_index":checkpoint,
        "normal_video_seconds":frame as f64/f64::from(render::constants::DEFAULT_VIDEO_FPS),
    }))
}

fn prepare(
    orbit: &OrbitData,
    record: &GenerationRecord,
    config: &NormalConfig,
    enhancements: &app::Enhancements,
) -> SilkResult<Prepared> {
    validate_inputs(orbit, record, config)?;
    let source = &record.simulation_config;
    let mut rng = Sha3RandomByteStream::new(
        &app::parse_seed(&record.seed)?,
        source.min_mass,
        source.max_mass,
        source.location,
        source.velocity,
    );
    // Source resolution belongs to the historical profile, even when this replay
    // requests a smaller output. Only change the render dimensions after resolving.
    let seeded = ResolvedVisualProfile::cosmic_signature(
        &mut rng,
        record.render_config.width,
        record.render_config.height,
    );
    if seeded.parameters.stack.label() != record.orbit_info.preferred_structure {
        return Err(invalid("Recorded preferred structure differs from this generator").into());
    }
    let chosen =
        resolve_recorded_stack(seeded.parameters.stack, &record.orbit_info.chosen_structure)?;
    let mut profile = seeded.with_stack(chosen);
    for (name, actual, expected) in [
        ("hdr_scale", profile.effect_config.hdr_scale, record.render_config.hdr_scale),
        ("clip_black", profile.effect_config.clip_black, record.render_config.clip_black),
        ("clip_white", profile.effect_config.clip_white, record.render_config.clip_white),
    ] {
        verify_number(name, actual, expected)?;
    }
    if let Some(projection) = record.randomization_log.as_ref().and_then(|log| {
        log.effects
            .iter()
            .flat_map(|effect| &effect.parameters)
            .find(|parameter| parameter.name == "projection")
    }) && projection.value.parse::<usize>()? != profile.parameters.projection.log_index()
    {
        return Err(invalid("Recorded projection differs from this generator").into());
    }
    if source.weights_randomized {
        let _ = rng.next_f64();
    }
    let candidate_index = record.orbit_info.selected_index;
    let selected_in_main = (record.orbit_info.retry_count == 0).then_some(candidate_index);
    let mut selected = consume_candidates(&mut rng, source.num_sims, selected_in_main);
    if record.orbit_info.retry_count > 0 {
        let mut retry_rng =
            rng.fork(format!("cosmic-retry/v1/{}", record.orbit_info.retry_count).as_bytes());
        let _ = retry_rng.next_f64();
        selected = consume_candidates(&mut retry_rng, candidate_index + 1, Some(candidate_index));
    }
    verify_initial_conditions(
        orbit,
        &selected.ok_or("Recorded candidate could not be reconstructed")?,
    )?;
    info!(
        candidate_index,
        candidates = source.num_sims,
        "Verified frozen winner; candidate simulations skipped"
    );

    let raw_positions: Vec<Vec<Vector3<f64>>> = (0..3)
        .map(|body| {
            orbit
                .samples
                .iter()
                .map(|sample| {
                    let p = sample[body];
                    Vector3::new(p.x, p.y, p.z)
                })
                .collect()
        })
        .collect();
    let closest_approach = closest_event(&raw_positions, orbit.dt)?;
    info!(event=%closest_approach,"Measured closest approach on the unprojected physical orbit");
    let mut positions = app::apply_projection(&raw_positions, profile.parameters.projection);
    drop(raw_positions);
    let view = app::apply_view_orientation(&mut positions, &rng, chosen);
    if record.drift_config.enabled {
        let drift = &record.drift_config;
        let (scale, arc, eccentricity) = if drift.randomized {
            (None, None, None)
        } else {
            (Some(drift.scale), Some(drift.arc_fraction), Some(drift.orbit_eccentricity))
        };
        let actual = app::apply_drift_transformation(
            &mut positions,
            &drift.mode,
            scale,
            arc,
            eccentricity,
            &mut rng,
        )?
        .ok_or("Recorded drift could not be reconstructed")?;
        verify_number("drift scale", actual.scale, drift.scale)?;
        verify_number("drift arc", actual.arc_fraction, drift.arc_fraction)?;
        verify_number("drift eccentricity", actual.orbit_eccentricity, drift.orbit_eccentricity)?;
    }
    let (colors, alphas) = app::generate_colors(
        &mut rng,
        source.num_steps_sim,
        record.render_config.alpha_denom,
        enhancements,
        profile.parameters.palette_phase,
    );
    let palette = render::color::current_palette_metadata();
    if palette.0 != record.render_config.palette_fingerprint
        || palette.1 != record.render_config.palette_gate
    {
        return Err(invalid(format!(
            "Recorded palette does not match replay: {} / {} versus {} / {}",
            record.render_config.palette_fingerprint,
            record.render_config.palette_gate,
            palette.0,
            palette.1
        ))
        .into());
    }
    profile.effect_config.width = config.width;
    profile.effect_config.height = config.height;
    Ok(Prepared { positions, colors, alphas, profile, view, palette, closest_approach })
}

fn orbit_digest(orbit: &OrbitData) -> String {
    let mut hash = Sha256::new();
    hash.update(orbit.dt.to_le_bytes());
    for mass in orbit.masses {
        hash.update(mass.to_le_bytes());
    }
    for sample in &orbit.samples {
        for &V3 { x, y, z } in sample {
            for value in [x, y, z] {
                hash.update(value.to_le_bytes());
            }
        }
    }
    hex::encode(hash.finalize())
}

fn marker_data(
    prepared: &Prepared,
    orbit: &OrbitData,
    config: &NormalConfig,
    aspect_correction: bool,
    schedule: &[usize],
) -> Value {
    let context = render::context::RenderContext::new(
        config.width,
        config.height,
        &prepared.positions,
        aspect_correction,
    );
    let frames: Vec<Value> = schedule
        .iter()
        .enumerate()
        .map(|(frame, &source_index)| {
            let bodies: [[f64; 2]; 3] = std::array::from_fn(|body| {
                let point = prepared.positions[body][source_index];
                let (x, y) = context.to_pixel(point.x, point.y);
                [f64::from(x) / f64::from(config.width), f64::from(y) / f64::from(config.height)]
            });
            json!({
                "frame":frame,
                "source_index":source_index,
                "source_fraction":source_index as f64/(orbit.samples.len()-1) as f64,
                "bodies":bodies,
            })
        })
        .collect();
    json!({
        "schema_version":1,
        "seed":orbit.seed,
        "fps":render::constants::DEFAULT_VIDEO_FPS,
        "width":config.width,
        "height":config.height,
        "coordinate_system":"normalized_xy_top_left",
        "body_ids":["A","B","C"],
        "closest_approach":prepared.closest_approach,
        "frames":frames,
    })
}

fn write_json(path: &Path, value: &Value) -> SilkResult<()> {
    let temporary = path.with_extension("json.partial");
    fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

/// Render the original normal video and its final still from a verified source.
///
/// Outputs are `normal.mp4`, `normal-hq.mp4`, `master.png`, `full.webp`,
/// `preview.webp`, `body-markers.json`, and `normal.json`. No search, spectral
/// gallery, spectral sweep, or public NFT metadata generation is performed. Like
/// the original executable, this entry point uses process-global render settings.
pub fn render_from_record(
    orbit: &OrbitData,
    record: &GenerationRecord,
    config: &NormalConfig,
    output: &Path,
) -> SilkResult<Value> {
    let enhancements = app::Enhancements::default();
    let prepared = prepare(orbit, record, config, &enhancements)?;
    crate::spectrum_simd::SAT_BOOST_ENABLED.store(enhancements.sat_boost, Ordering::Relaxed);
    render::ACES_TWEAK_ENABLED.store(enhancements.aces_tweak, Ordering::Relaxed);
    render::drawing::DISPERSION_BOOST_ENABLED
        .store(enhancements.dispersion_boost, Ordering::Relaxed);
    fs::create_dir_all(output)?;
    let files = ["normal.mp4", "normal-hq.mp4", "master.png", "full.webp", "preview.webp"];
    let paths: Vec<String> = files
        .iter()
        .map(|name| {
            output
                .join(name)
                .to_str()
                .map(str::to_owned)
                .ok_or_else(|| invalid("Normal output paths must be UTF-8"))
        })
        .collect::<Result<_, _>>()?;
    let render_config = render::RenderConfig {
        hdr_scale: prepared.profile.effect_config.hdr_scale,
        bloom_mode: render::BloomMode::Dog,
    };
    let levels = app::build_histogram_and_levels(
        &prepared.positions,
        &prepared.colors,
        &prepared.alphas,
        &prepared.profile.effect_config,
        &render_config,
        enhancements.aspect_correction,
        &prepared.profile.parameters,
    )?;
    let schedule = frame_schedule(orbit.samples.len());
    let markers = marker_data(&prepared, orbit, config, enhancements.aspect_correction, &schedule);
    // A failed replacement must not leave a previous successful manifest
    // certifying media or markers that this invocation has already overwritten.
    write_json(
        &output.join("normal.json"),
        &json!({
            "schema_version":1,"kind":"normal-accumulation-replay","complete":false,
            "seed":orbit.seed,"configuration":config,
            "source_sample_count":orbit.samples.len(),
            "frame_count":schedule.len(),"fps":render::constants::DEFAULT_VIDEO_FPS,
        }),
    )?;
    write_json(&output.join("body-markers.json"), &markers)?;
    let settings = render::SpectralRenderSettings::new(
        &prepared.profile.effect_config,
        &render_config,
        enhancements.aspect_correction,
    )
    .with_traits(prepared.profile.parameters.scene_traits());
    let accum = app::render_video(
        render::SpectralScene::new(&prepared.positions, &prepared.colors, &prepared.alphas),
        &levels,
        settings,
        app::VideoOutputPaths { web: &paths[0], high_quality: &paths[1] },
        app::ImageOutputPaths {
            master_png: &paths[2],
            full_webp: &paths[3],
            preview_webp: &paths[4],
        },
        config.fast_encode,
    )?;
    drop(accum);
    let fps = render::constants::DEFAULT_VIDEO_FPS;
    let artifacts=files.iter().zip(&paths).map(|(name,path)|{
        Ok(json!({"path":name,"bytes":fs::metadata(path)?.len(),"sha256":cache::file_hash(Path::new(path))?}))
    }).collect::<SilkResult<Vec<_>>>()?;
    let metadata = json!({
        "schema_version":1,"kind":"normal-accumulation-replay","complete":true,
        "seed":orbit.seed,"configuration":config,
        "source_generation_record":record,"source_orbit_provenance":orbit.provenance,
        "source_samples_sha256":orbit_digest(orbit),
        "source_sample_count":orbit.samples.len(),"source_dt":orbit.dt,
        "source_first_step":0,"source_last_step":orbit.samples.len()-1,
        "frame_count":schedule.len(),"fps":fps,"duration_seconds":schedule.len() as f64/f64::from(fps),
        "frame_interval":(orbit.samples.len()/render::constants::DEFAULT_TARGET_FRAMES as usize).max(1),
        "frame_checkpoint_indices":schedule,
        "body_markers":markers,
        "closest_approach":prepared.closest_approach,
        "checkpoint_semantics":"inclusive accumulated source index; ribbon segments may read the following sample",
        "projection":prepared.profile.parameters.projection.label(),
        "structure":prepared.profile.parameters.stack.label(),
        "view_random_triple":[prepared.view.0,prepared.view.1,prepared.view.2],
        "palette_fingerprint":prepared.palette.0,"palette_gate":prepared.palette.1,
        "candidate_simulations_run":0,"main_rng_candidate_draws":record.simulation_config.num_sims*21,
        "artifacts":artifacts,
    });
    write_json(&output.join("normal.json"), &metadata)?;
    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_schedule_includes_full_legacy_million_step_span() {
        let schedule = frame_schedule(1_000_000);
        assert_eq!(schedule.len(), 1802);
        assert_eq!(&schedule[..3], &[555, 1110, 1665]);
        assert_eq!(&schedule[1800..], &[999_555, 999_999]);
        assert!(schedule.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn small_schedules_keep_the_final_source_sample_once() {
        assert!(frame_schedule(0).is_empty());
        assert_eq!(frame_schedule(1), vec![0]);
        assert_eq!(frame_schedule(2), vec![1]);
        assert_eq!(frame_schedule(4), vec![1, 2, 3]);
        assert_eq!(frame_schedule(1800).len(), 1799);
    }

    #[test]
    fn full_batch_consumption_matches_sequential_candidate_generation() {
        let make_rng = || Sha3RandomByteStream::new(&[0xCA, 0xFE], 100.0, 300.0, 300.0, 1.0);
        let mut sequential = make_rng();
        let mut expected = None;
        for index in 0..12 {
            let bodies = sample_candidate(&mut sequential);
            if index == 3 {
                expected = Some(bodies);
            }
        }
        let mut consumed = make_rng();
        let actual = consume_candidates(&mut consumed, 12, Some(3)).unwrap();
        for (actual, expected) in actual.iter().zip(expected.unwrap()) {
            assert_eq!(actual.mass.to_bits(), expected.mass.to_bits());
            assert_eq!(actual.position, expected.position);
            assert_eq!(actual.velocity, expected.velocity);
        }
        assert_eq!(sequential.next_f64().to_bits(), consumed.next_f64().to_bits());
    }

    #[test]
    fn seeded_stack_keeps_exact_alpha_instead_of_parsing_rounded_label() {
        let preferred = render::LayerStack {
            primary: render::StructureMode::OrbitRibbons,
            underlay: None,
            accent: Some(render::StackLayer {
                vocabulary: render::StructureMode::StippleConstellation,
                alpha: 0.077_712_345,
            }),
        };
        let resolved = resolve_recorded_stack(preferred, &preferred.label()).unwrap();
        assert_eq!(resolved, preferred);
        assert!(resolve_recorded_stack(preferred, "unknown-structure").is_err());
    }

    #[test]
    fn markers_use_source_body_order_and_actual_image_y_direction() {
        let mut rng = Sha3RandomByteStream::new(&[1, 2], 100.0, 300.0, 300.0, 1.0);
        let config = NormalConfig { width: 64, height: 48, ..NormalConfig::default() };
        let points = [V3::new(-1.0, -1.0, 0.0), V3::new(0.0, 0.0, 0.0), V3::new(1.0, 1.0, 0.0)];
        let orbit = OrbitData {
            seed: "0x0102".into(),
            dt: 0.001,
            masses: [1.0; 3],
            samples: vec![points; 4],
            provenance: json!({}),
        };
        let prepared = Prepared {
            positions: points
                .map(|point| vec![Vector3::new(point.x, point.y, point.z); 4])
                .to_vec(),
            colors: Vec::new(),
            alphas: Vec::new(),
            profile: ResolvedVisualProfile::cosmic_signature(&mut rng, 64, 48),
            view: (0.0, 0.0, 0.0),
            palette: (String::new(), String::new()),
            closest_approach: Value::Null,
        };
        let markers = marker_data(&prepared, &orbit, &config, true, &[1, 3]);
        assert_eq!(markers["coordinate_system"], "normalized_xy_top_left");
        assert_eq!(markers["body_ids"], json!(["A", "B", "C"]));
        assert_eq!(markers["frames"][0]["source_index"], 1);
        assert_eq!(markers["frames"][1]["source_fraction"], 1.0);
        let bodies = markers["frames"][0]["bodies"].as_array().unwrap();
        assert!(bodies[0][1].as_f64().unwrap() < bodies[1][1].as_f64().unwrap());
        assert!(bodies[1][1].as_f64().unwrap() < bodies[2][1].as_f64().unwrap());
        assert!((bodies[1][0].as_f64().unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn closest_approach_uses_raw_distances_and_nearest_normal_checkpoint() {
        let raw = vec![
            vec![Vector3::zeros(); 4],
            [1.0, 0.8, 0.1, 0.4].map(|x| Vector3::new(x, 0.0, 0.0)).to_vec(),
            vec![Vector3::new(5.0, 5.0, 0.0); 4],
        ];
        let event = closest_event(&raw, 0.001).unwrap();
        assert_eq!(event["source_index"], 2);
        assert_eq!(event["pair"], json!([0, 1]));
        assert_eq!(event["body_ids"], json!(["A", "B"]));
        assert_eq!(event["distance"], 0.1);
        assert_eq!(event["normal_frame"], 1);
        assert_eq!(event["normal_checkpoint_source_index"], 2);
        assert!((event["normal_video_seconds"].as_f64().unwrap() - 1.0 / 60.0).abs() < 1e-12);
    }
}
